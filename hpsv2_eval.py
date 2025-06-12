import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
import hpsv2
import huggingface_hub
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import root_path, hps_version_map
import time
from torch.utils.data import Dataset, DataLoader

# Habana-specific imports, though the script logic will be device-agnostic.
try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    print("Habana frameworks not found, running on CPU/GPU.")

class ImagePromptDataset(Dataset):
    """
    A custom PyTorch Dataset to handle loading images and prompts.
    This allows us to use DataLoader for efficient, parallel data prefetching.
    """
    def __init__(self, prompts, img_dir, transform):
        self.prompts = prompts
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        prompt = self.prompts[i]
        
        preprocessed_images = []
        img_paths_for_prompt = []
        for j in range(4):
            img_idx = i * 4 + j + 1
            img_path = os.path.join(self.img_dir, f'image_{img_idx}.png')
            if os.path.exists(img_path):
                try:
                    # Loading and transforming the image
                    image = Image.open(img_path)
                    preprocessed_images.append(self.transform(image))
                    img_paths_for_prompt.append(img_path)
                except Exception as e:
                    tqdm.write(f"\nWarning: Could not load or process image {img_path}: {e}")
        
        if not preprocessed_images:
            # Return empty tensors if no images were found or loaded for this prompt
            return torch.empty(0), prompt, []

        return torch.stack(preprocessed_images), prompt, img_paths_for_prompt


def analyze_hps_scores(img_dir, prompt_csv_path, output_dir, exp_name):
    """
    Calculates HPSv2 scores for generated images, saves them to a CSV file,
    prints statistics, and creates a visualization of the score distribution.
    This version is optimized to load the model only once and process images in batches.
    """
    # Define paths
    IMAGE_DIR = img_dir
    OUTPUT_CSV_PATH = os.path.join(output_dir, 'hpsv2_scores.csv')
    OUTPUT_FIG_PATH = os.path.join(output_dir, 'hpsv2_scores_histogram.png')
    OUTPUT_STATS_PATH = os.path.join(output_dir, 'hpsv2_top_bottom_scores.txt')
    OUTPUT_TOP_BOTTOM_FIG_PATH = os.path.join(output_dir, 'hpsv2_top_bottom_images.png')

    # Check for device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.hpu.is_available():
        device = 'hpu'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # --- 1. Initialize Model and Tokenizer once ---
    print("Initializing HPSv2 model...")
    start_time = time.time()
    hps_version = "v2.1"
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    
    # Download checkpoint
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    time_model_init = time.time()
    print(f"Model initialized in {time_model_init - start_time:.4f} seconds")

    if device == 'hpu':
        print("Compiling model for HPU...")
        model = torch.compile(model, backend="hpu_backend")
        time_model_compile = time.time()
        print(f"Model compiled successfully in {time_model_compile - time_model_init:.4f} seconds")
    print("Model initialized successfully.")


    # 2. Read prompts and create DataLoader
    try:
        prompts_df = pd.read_csv(prompt_csv_path, header=None)
        prompts_df.rename(columns={0: 'prompt'}, inplace=True)
        prompts_df.dropna(subset=['prompt'], inplace=True)
        prompts_df = prompts_df[prompts_df['prompt'].str.strip() != '']
        prompts = prompts_df['prompt'].tolist()
        print(f"Successfully loaded {len(prompts)} prompts from {prompt_csv_path}")

        dataset = ImagePromptDataset(prompts, IMAGE_DIR, preprocess_val)
        
        # Use multiple worker processes to load data in the background.
        # This allows the HPU to compute on one batch while the next are being loaded by the CPU.
        num_workers = min(os.cpu_count(), 8)
        print(f"Using {num_workers} workers for data loading.")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_csv_path}")
        return
    except Exception as e:
        print(f"An error occurred while setting up the dataloader: {e}")
        return

    # 3. Calculate HPSv2 scores with batching and prefetching
    results = []
    latencies = {
        'data_loading_wait': [],
        'data_transfer_to_device': [],
        'model_inference': [],
        'result_post_processing': [],
        'total_loop_time': []
    }
    
    print("Calculating HPSv2 scores for images...")
    
    loop_iter_start_time = time.time()
    for i, (image_batch, prompt_tuple, img_paths_list) in enumerate(tqdm(dataloader, desc="Processing prompts")):
        
        # --- Component 1: Data Loading Wait Time ---
        # This measures how long the main process waited for the dataloader workers.
        # With effective prefetching, this should be very low after the initial startup.
        data_ready_time = time.time()
        latency_data_wait = data_ready_time - loop_iter_start_time

        if image_batch.nelement() == 0: # Handle cases where a prompt had no valid images
            loop_iter_start_time = time.time()
            continue

        # Dataloader with batch_size=1 adds an extra dimension, so we remove it.
        image_batch = image_batch.squeeze(0)
        prompt = prompt_tuple[0]
        img_paths_for_prompt = img_paths_list[0]
        
        try:
            # --- Component 2: Data transfer to device ---
            time_start_transfer = time.time()
            image_batch_device = image_batch.to(device=device, non_blocking=True)
            text_batch = tokenizer([prompt] * len(image_batch_device)).to(device=device, non_blocking=True)
            if device == 'hpu': htcore.mark_step()
            time_end_transfer = time.time()
            latency_transfer = time_end_transfer - time_start_transfer

            # --- Component 3: Model inference ---
            time_start_inference = time.time()
            with torch.no_grad(), torch.amp.autocast(device_type=device):
                outputs = model(image_batch_device, text_batch)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                scores = torch.diagonal(logits_per_image)
            if device == 'hpu': htcore.mark_step()
            time_end_inference = time.time()
            latency_inference = time_end_inference - time_start_inference

            # --- Component 4: Post-processing (includes HPU->CPU transfer) ---
            time_start_post = time.time()
            scores_cpu = scores.cpu()
            scores_float = [s.float().item() for s in scores_cpu]

            for path, score_val in zip(img_paths_for_prompt, scores_float):
                results.append({
                    'prompt_index': i,
                    'prompt': prompt,
                    'image_path': path,
                    'hps_score': score_val
                })
            time_end_post = time.time()
            latency_post = time_end_post - time_start_post
            
            loop_end_time = time.time()
            latency_total = loop_end_time - data_ready_time # Total time for this loop's work

            # Store latencies
            latencies['data_loading_wait'].append(latency_data_wait)
            latencies['data_transfer_to_device'].append(latency_transfer)
            latencies['model_inference'].append(latency_inference)
            latencies['result_post_processing'].append(latency_post)
            latencies['total_loop_time'].append(latency_total)

            if i < 3:
                tqdm.write(f"\n--- Latency for prompt {i} ---")
                tqdm.write(f"  1. Data Loading Wait:       {latency_data_wait:.4f}s")
                tqdm.write(f"  2. Data Transfer to Device: {latency_transfer:.4f}s")
                tqdm.write(f"  3. Model Inference:         {latency_inference:.4f}s")
                tqdm.write(f"  4. Result Post-processing:  {latency_post:.4f}s")
                tqdm.write(f"  -----------------------------")
                tqdm.write(f"  Total Active Loop Time:     {latency_total:.4f}s")

        except Exception as e:
            tqdm.write(f"\nError scoring images for prompt {i}: {e}")
            tqdm.write(f"Prompt: {prompt}")

        # Reset timer for the next iteration's wait time measurement
        loop_iter_start_time = time.time()


    if not results:
        print("No results were generated. Please check image paths and prompts.")
        return

    # Print average latencies at the end
    print("\n\n--- Average Latencies Across All Prompts ---")
    for name, values in latencies.items():
        if values:
            # Skip first N iterations for a more stable average, as they can be outliers
            warmup_iterations = 5
            stable_values = values[warmup_iterations:]
            if stable_values:
                mean_val = np.mean(stable_values)
                print(f"  Average {name.replace('_', ' ').title()}: {mean_val:.4f}s")
    print("------------------------------------------\n")

    # 4. Save scores to CSV and print statistics
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nScores saved to {OUTPUT_CSV_PATH}")

    scores = results_df['hps_score']
    mean_score = scores.mean()
    variance_score = scores.var()
    min_score = scores.min()
    max_score = scores.max()

    print("\n--- HPSv2 Score Statistics ---")
    print(f"Average Score: {mean_score:.4f}")
    print(f"Variance:      {variance_score:.4f}")
    print(f"Minimum Score: {min_score:.4f}")
    print(f"Maximum Score: {max_score:.4f}")
    print("------------------------------")

    # 5. Find and save top/bottom scored prompts and images
    top_5 = results_df.nlargest(5, 'hps_score')
    bottom_5 = results_df.nsmallest(5, 'hps_score')

    with open(OUTPUT_STATS_PATH, 'w') as f:
        f.write("--- Top 5 Highest HPSv2 Scores ---\n")
        for _, row in top_5.iterrows():
            image_number = os.path.basename(row['image_path']).split('_')[1].split('.')[0]
            f.write(f"Score: {row['hps_score']:.4f}\n")
            f.write(f"Image Number: {image_number}\n")
            f.write(f"Prompt: {row['prompt']}\n\n")

        f.write("\n--- Top 5 Lowest HPSv2 Scores ---\n")
        for _, row in bottom_5.iterrows():
            image_number = os.path.basename(row['image_path']).split('_')[1].split('.')[0]
            f.write(f"Score: {row['hps_score']:.4f}\n")
            f.write(f"Image Number: {image_number}\n")
            f.write(f"Prompt: {row['prompt']}\n\n")

    print(f"Top/bottom score details saved to {OUTPUT_STATS_PATH}")


    # 6. Create visualization for top/bottom images
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Top and Bottom 5 HPSv2 Scored Images', fontsize=24, y=1.02)

    # --- Consolidate plotting logic to avoid errors ---
    datasets_to_plot = [(top_5, 0), (bottom_5, 1)]

    for df, row_idx in datasets_to_plot:
        # Use reset_index to ensure we have a continuous index for enumeration
        df_reset = df.reset_index(drop=True)
        for col_idx in range(5):
            ax = axs[row_idx, col_idx]
            ax.axis('off')
            # Check if a corresponding image exists in the dataframe
            if col_idx < len(df_reset):
                row = df_reset.iloc[col_idx]
                try:
                    img = Image.open(row['image_path'])
                    ax.imshow(img)
                    filename = os.path.basename(row['image_path'])
                    score = row['hps_score']
                    ax.set_title(f"{filename}\nScore: {score:.4f}", fontsize=12)
                except FileNotFoundError:
                    ax.set_title(f"Image not found\n{os.path.basename(row['image_path'])}", fontsize=10)
                except Exception as e:
                    ax.set_title(f"Error plotting", fontsize=10)
                    tqdm.write(f"Error plotting image {row['image_path']}: {e}")
            else:
                # If there are fewer than 5 images, leave the subplot blank
                pass
    
    # Add row titles to the left of the grid
    axs[0, 0].text(-0.15, 0.5, 'Top 5\nHighest Scores', transform=axs[0, 0].transAxes,
                   ha='center', va='center', rotation='vertical', fontsize=18)
    axs[1, 0].text(-0.15, 0.5, 'Top 5\nLowest Scores', transform=axs[1, 0].transAxes,
                   ha='center', va='center', rotation='vertical', fontsize=18)

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=2.0)
    plt.savefig(OUTPUT_TOP_BOTTOM_FIG_PATH, bbox_inches='tight')
    print(f"Top/bottom images visualization saved to {OUTPUT_TOP_BOTTOM_FIG_PATH}")


    # 7. Visualize the scores (Histogram)
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 7))
    plt.hist(scores, bins=50, alpha=0.75, color='royalblue', edgecolor='black')
    plt.title(f'Distribution of HPSv2 Scores for {exp_name}', fontsize=18)
    plt.xlabel('HPSv2 Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.4f}')
    plt.legend()
    
    plt.savefig(OUTPUT_FIG_PATH)
    print(f"Score distribution histogram saved to {OUTPUT_FIG_PATH}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--img_dir', type=str, default='/workspace/jh/flux/outputs/250prompts', help='image directory')
    parser.add_argument('--prompt_csv_path', type=str, default='../prompts_250.csv', help='prompt csv path')
    args = parser.parse_args()
    
    exp_name = args.exp
    # mkdir if not exists
    if not os.path.exists(f'../{exp_name}'):
        os.makedirs(f'../{exp_name}')
    img_dir = f'{args.img_dir}/{exp_name}'
    prompt_csv_path = args.prompt_csv_path
    if not os.path.exists(prompt_csv_path):
        print(f"Error: Prompt file not found at {prompt_csv_path}")
        exit(1)
    output_dir = f'../{exp_name}/hpsv2_eval'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    analyze_hps_scores(img_dir, prompt_csv_path, output_dir, exp_name)
