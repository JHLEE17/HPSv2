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
# PickScore needs to be imported if used
# from transformers import AutoProcessor, AutoModel
import huggingface_hub


# Habana-specific imports, though the script logic will be device-agnostic.
try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    print("Habana frameworks not found, running on CPU/GPU.")

class ImagePromptDataset(Dataset):
    """
    A custom PyTorch Dataset to handle loading images and prompts.
    This version returns PIL images, and transformation is done in the main loop.
    """
    def __init__(self, prompts, img_dir):
        self.prompts = prompts
        self.img_dir = img_dir

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        prompt = self.prompts[i]
        
        images = []
        img_paths_for_prompt = []
        for j in range(4):
            img_idx = i * 4 + j + 1
            img_path = os.path.join(self.img_dir, f'image_{img_idx}.png')
            if os.path.exists(img_path):
                try:
                    # Loading the image as a PIL object
                    image = Image.open(img_path)
                    images.append(image)
                    img_paths_for_prompt.append(img_path)
                except Exception as e:
                    tqdm.write(f"\nWarning: Could not load image {img_path}: {e}")
        
        if not images:
            # Return empty lists if no images were found or loaded for this prompt
            return [], prompt, []

        return images, prompt, img_paths_for_prompt


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of PIL images, which the default collate_fn cannot handle.
    """
    images_list = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    paths_list = [item[2] for item in batch]
    return images_list, prompts, paths_list


def analyze_scores(img_dir, prompt_csv_path, output_dir, exp_name, scorer):
    """
    Calculates HPSv2 or PickScore for generated images, saves them to a CSV file,
    prints statistics, and creates a visualization of the score distribution.
    """
    # Define paths based on the scorer
    if scorer == 'hpsv2':
        SCORE_NAME = 'hps_score'
    elif scorer == 'pickscore':
        SCORE_NAME = 'pickscore_score'
    elif scorer == 'imagereward':
        SCORE_NAME = 'imagereward_score'
    OUTPUT_CSV_PATH = os.path.join(output_dir, f'{scorer}_scores.csv')
    OUTPUT_FIG_PATH = os.path.join(output_dir, f'{scorer}_scores_histogram.png')
    OUTPUT_STATS_PATH = os.path.join(output_dir, f'{scorer}_top_bottom_scores.txt')
    OUTPUT_TOP_BOTTOM_FIG_PATH = os.path.join(output_dir, f'{scorer}_top_bottom_images.png')

    # Check for device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.hpu.is_available():
        device = 'hpu'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # --- 1. Initialize Model and Processor based on scorer ---
    print(f"Initializing {scorer.upper()} model...")
    start_time = time.time()
    
    if scorer == 'hpsv2':
        import hpsv2
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import root_path, hps_version_map
        hps_version = "v2.1"
        model, _, processor = create_model_and_transforms(
            'ViT-H-14', 'laion2B-s32B-b79K', precision='amp', device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
            force_image_size=None, pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True, with_score_predictor=False,
            with_region_predictor=False
        )
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
    
    elif scorer == 'pickscore':
        from transformers import AutoProcessor, CLIPModel
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        model = CLIPModel.from_pretrained(model_pretrained_name_or_path)
        tokenizer = None # PickScore's processor handles text
    
    elif scorer == 'imagereward':
        import ImageReward as RM
        model = RM.load("ImageReward-v1.0", device=device)
        # ImageReward handles its own tokenization and processing
        processor = None
        tokenizer = None
    
    else:
        raise ValueError(f"Unknown scorer: {scorer}")

    if scorer != 'imagereward': # ImageReward model is not a standard nn.Module
        model = model.to(device).eval()
        time_model_init = time.time()
        print(f"Model initialized in {time_model_init - start_time:.4f} seconds")

        if device == 'hpu':
            print("Compiling model for HPU...")
            model = torch.compile(model, backend="hpu_backend")
            time_model_compile = time.time()
            print(f"Model compiled successfully in {time_model_compile - time_model_init:.4f} seconds")
    else:
        time_model_init = time.time()
        print(f"Model initialized in {time_model_init - start_time:.4f} seconds")

    print("Model initialized successfully.")


    # 2. Read prompts and create DataLoader
    try:
        prompts_df = pd.read_csv(prompt_csv_path, header=None)
        prompts_df.rename(columns={0: 'prompt'}, inplace=True)
        prompts_df.dropna(subset=['prompt'], inplace=True)
        prompts_df = prompts_df[prompts_df['prompt'].str.strip() != '']
        prompts = prompts_df['prompt'].tolist()
        print(f"Successfully loaded {len(prompts)} prompts from {prompt_csv_path}")

        dataset = ImagePromptDataset(prompts, img_dir)
        
        num_workers = min(os.cpu_count(), 8)
        print(f"Using {num_workers} workers for data loading.")
        # We need a custom collate_fn to handle PIL images, which default_collate can't batch.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_csv_path}")
        return
    except Exception as e:
        print(f"An error occurred while setting up the dataloader: {e}")
        return

    # 3. Calculate scores with batching and prefetching
    results = []
    latencies = {
        'data_loading_wait': [],
        'data_transfer_to_device': [],
        'model_inference': [],
        'result_post_processing': [],
        'total_loop_time': []
    }
    
    print(f"Calculating {scorer.upper()} scores for images...")
    
    loop_iter_start_time = time.time()
    # Dataloader with batch_size=1 returns a batch of 1 item, so we unpack it.
    for i, (image_list_batch, prompt_tuple, img_paths_list_batch) in enumerate(tqdm(dataloader, desc="Processing prompts")):
        
        # --- Component 1: Data Loading Wait Time ---
        data_ready_time = time.time()
        latency_data_wait = data_ready_time - loop_iter_start_time

        # Unpack the batch of size 1
        image_list = image_list_batch[0]
        prompt = prompt_tuple[0]
        img_paths_for_prompt = img_paths_list_batch[0]

        # Skip if no images were found for this prompt
        if not image_list:
            loop_iter_start_time = time.time()
            continue

        try:
            time_start_transfer = time.time()
            # --- Score-specific preprocessing and inference ---
            with torch.no_grad(), torch.amp.autocast(device_type=device):
                if scorer == 'hpsv2':
                    # HPSv2: transform PIL images to tensors
                    image_batch_device = torch.stack([processor(img) for img in image_list]).to(device)
                    text_batch = tokenizer([prompt] * len(image_batch_device)).to(device=device, non_blocking=True)
                    if device == 'hpu': htcore.mark_step()
                    time_end_transfer = time.time()

                    time_start_inference = time.time()
                    outputs = model(image_batch_device, text_batch)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    scores = torch.diagonal(logits_per_image)

                elif scorer == 'pickscore':
                    # PickScore: use its processor for both images and text
                    image_inputs = processor(
                        images=image_list, padding=True, truncation=True, max_length=77, return_tensors="pt"
                    ).to(device)
                    text_inputs = processor(
                        text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt"
                    ).to(device)
                    if device == 'hpu': htcore.mark_step()
                    time_end_transfer = time.time()

                    time_start_inference = time.time()
                    image_embs = model.get_image_features(**image_inputs)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                
                    text_embs = model.get_text_features(**text_inputs)
                    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                
                    # score
                    scores = (model.logit_scale.exp() * (text_embs @ image_embs.T))[0]

                elif scorer == 'imagereward':
                    # ImageReward is handled differently as it doesn't return a standard tensor
                    pass
            
            if scorer != 'imagereward':
                if device == 'hpu': htcore.mark_step()
                time_end_inference = time.time()
            
                latency_transfer = time_end_transfer - time_start_transfer
                latency_inference = time_end_inference - time_start_inference
                
                # --- Component 4: Post-processing (includes HPU->CPU transfer) ---
                time_start_post = time.time()
                scores_cpu = scores.cpu()
                scores_float = [s.float().item() for s in scores_cpu]
                time_end_post = time.time()
                latency_post = time_end_post - time_start_post
            
            else: # ImageReward specific path
                time_start_transfer = time.time()
                # ImageReward uses file paths directly, so no device transfer is measured here.
                time_end_transfer = time.time()
                latency_transfer = time_end_transfer - time_start_transfer

                time_start_inference = time.time()
                with torch.no_grad():
                     # Use model.score in a loop for robustness, as inference_rank can have inconsistent return types.
                     scores_float = [model.score(prompt, img_path) for img_path in img_paths_for_prompt]
                time_end_inference = time.time()
                latency_inference = time_end_inference - time_start_inference

                time_start_post = time.time()
                # scores_float is already the list of floats we need.
                time_end_post = time.time()
                latency_post = time_end_post - time_start_post

            for path, score_val in zip(img_paths_for_prompt, scores_float):
                results.append({
                    'prompt_index': i,
                    'prompt': prompt,
                    'image_path': path,
                    SCORE_NAME: score_val
                })
            
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
    print(f"\n\n--- Average Latencies Across All Prompts ({scorer.upper()}) ---")
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

    scores = results_df[SCORE_NAME]
    mean_score = scores.mean()
    variance_score = scores.var()
    min_score = scores.min()
    max_score = scores.max()

    print(f"\n--- {scorer.upper()} Score Statistics ---")
    print(f"Average Score: {mean_score:.4f}")
    print(f"Variance:      {variance_score:.4f}")
    print(f"Minimum Score: {min_score:.4f}")
    print(f"Maximum Score: {max_score:.4f}")
    print("------------------------------")

    # 5. Find and save top/bottom scored prompts and images
    top_5 = results_df.nlargest(5, SCORE_NAME)
    bottom_5 = results_df.nsmallest(5, SCORE_NAME)

    with open(OUTPUT_STATS_PATH, 'w') as f:
        f.write(f"--- Top 5 Highest {scorer.upper()} Scores ---\n")
        for _, row in top_5.iterrows():
            image_number = os.path.basename(row['image_path']).split('_')[1].split('.')[0]
            f.write(f"Score: {row[SCORE_NAME]:.4f}\n")
            f.write(f"Image Number: {image_number}\n")
            f.write(f"Prompt: {row['prompt']}\n\n")

        f.write(f"\n--- Top 5 Lowest {scorer.upper()} Scores ---\n")
        for _, row in bottom_5.iterrows():
            image_number = os.path.basename(row['image_path']).split('_')[1].split('.')[0]
            f.write(f"Score: {row[SCORE_NAME]:.4f}\n")
            f.write(f"Image Number: {image_number}\n")
            f.write(f"Prompt: {row['prompt']}\n\n")

    print(f"Top/bottom score details saved to {OUTPUT_STATS_PATH}")


    # 6. Create visualization for top/bottom images
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f'Top and Bottom 5 {scorer.upper()} Scored Images', fontsize=24, y=1.02)

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
                    score = row[SCORE_NAME]
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
    plt.title(f'Distribution of {scorer.upper()} Scores for {exp_name}', fontsize=18)
    plt.xlabel(f'{scorer.upper()} Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.4f}')
    plt.legend()
    
    plt.savefig(OUTPUT_FIG_PATH)
    print(f"Score distribution histogram saved to {OUTPUT_FIG_PATH}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/workspace/jh/flux/outputs/250prompts', help='image directory')
    parser.add_argument('--exp', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--prompt_file', type=str, default='../prompts_250.csv', help='prompt csv path')
    parser.add_argument('--scorer', type=str, default='hpsv2', choices=['hpsv2', 'pickscore', 'imagereward'], help='Scoring model to use: hpsv2, pickscore, or imagereward.')
    args = parser.parse_args()
    
    exp_name = args.exp
    img_dir = f'{args.img_dir}/{exp_name}'
    prompt_file = args.prompt_file
    scorer = args.scorer

    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file not found at {prompt_file}")
        exit(1)
    output_dir = f'../hpsv2_eval/{exp_name}' # Keeping the parent output directory the same for consistency
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    analyze_scores(img_dir, prompt_file, output_dir, exp_name, scorer)
