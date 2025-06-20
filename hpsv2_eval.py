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
import argparse


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


def analyze_scores(img_dir, prompt_csv_path, output_dir, scorer):
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
        
        # Determine precision based on device capabilities
        if device == 'hpu':
            # HPU supports amp with bfloat16
            precision = 'fp32'
        elif device == 'cuda' and torch.cuda.is_bf16_supported():
            # Use amp if bfloat16 is supported on CUDA
            precision = 'fp32'
        else:
            # Fall back to fp32 to avoid BFloat16 issues
            precision = 'fp32'
        
        print(f"Using precision: {precision} for HPSv2 model")
        
        model, _, processor = create_model_and_transforms(
            'ViT-H-14', 'laion2B-s32B-b79K', precision=precision, device=device, jit=False,
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

        time_start_transfer = time.time()
        # --- Score-specific preprocessing and inference ---
        # Disable autocast to avoid BFloat16 issues for now
        with torch.no_grad():
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
                latency_transfer = time_end_transfer - time_start_transfer  

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
                latency_transfer = time_end_transfer - time_start_transfer

            elif scorer == 'imagereward':
                # ImageReward is handled differently as it doesn't return a standard tensor
                latency_transfer = -1.0

        if scorer != 'imagereward':
            if device == 'hpu': htcore.mark_step()
            time_end_inference = time.time()
            
            latency_inference = time_end_inference - time_start_inference

            time_start_post = time.time()
            
            # --- Result post-processing ---
            scores_for_prompt = scores.cpu().numpy()

            # Save score for each image
            for j, score in enumerate(scores_for_prompt):
                result = {
                    'prompt_index': i + 1,
                    'image_path': img_paths_for_prompt[j],
                    SCORE_NAME: score
                }
                results.append(result)

            latency_post = time.time() - time_start_post

        elif scorer == 'imagereward':
            # For ImageReward, we process one image at a time
            time_start_inference = time.time()
            
            for j, img in enumerate(image_list):
                # The model.infer method returns a single score
                score = model.score(prompt, [img])
                result = {
                    'prompt_index': i + 1,
                    'image_path': img_paths_for_prompt[j],
                    SCORE_NAME: score
                }
                results.append(result)

            if device == 'hpu': htcore.mark_step()
            time_end_inference = time.time()
            latency_inference = time_end_inference - time_start_inference
            time_start_post = time.time()
            latency_post = time.time() - time_start_post

        # --- Latency calculation ---
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
                avg_latency = sum(stable_values) / len(stable_values)
                tqdm.write(f"  Average {name.replace('_', ' ').title()}: {avg_latency:.4f}s")
    
    # --- 4. Process and Save Results ---
    if results:
        df = pd.DataFrame(results)
        
        # Save all scores to a single CSV
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nScores saved to {OUTPUT_CSV_PATH}")

        # --- 5. Statistics and Visualization ---
        print(f"\n--- Statistics for {SCORE_NAME} ---")
        print(df[SCORE_NAME].describe())
        
        # Create histogram with statistics
        plt.figure(figsize=(12, 8))
        
        # Calculate statistics
        scores_data = df[SCORE_NAME]
        mean_val = scores_data.mean()
        std_val = scores_data.std()
        median_val = scores_data.median()
        min_val = scores_data.min()
        max_val = scores_data.max()
        count_val = len(scores_data)
        
        # Create histogram
        n, bins, patches = plt.hist(scores_data, bins=50, color='blue', alpha=0.7, edgecolor='black')
        
        # Add vertical lines for statistics
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        
        # Add statistics text box
        stats_text = f'''Statistics:
Count: {count_val}
Mean: {mean_val:.4f}
Median: {median_val:.4f}
Std: {std_val:.4f}
Min: {min_val:.4f}
Max: {max_val:.4f}'''
        
        # Position text box in upper right corner
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.title(f'Distribution of {SCORE_NAME}', fontsize=14, fontweight='bold')
        plt.xlabel(SCORE_NAME, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histogram with statistics saved to {OUTPUT_FIG_PATH}")
        
        # Find and save top 5 and bottom 5 scores
        top_5 = df.nlargest(5, SCORE_NAME)
        bottom_5 = df.nsmallest(5, SCORE_NAME)
        
        with open(OUTPUT_STATS_PATH, 'w') as f:
            f.write(f"--- Top 5 {SCORE_NAME} ---\n")
            f.write(top_5.to_string())
            f.write("\n\n")
            f.write(f"--- Bottom 5 {SCORE_NAME} ---\n")
            f.write(bottom_5.to_string())
        print(f"Top and bottom scores saved to {OUTPUT_STATS_PATH}")
        
        # Save top and bottom images
        try:
            fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            fig.suptitle(f'Top and Bottom 5 Images based on {SCORE_NAME}', fontsize=16)
            for i, row in enumerate(top_5.itertuples()):
                # Use getattr to access score column dynamically
                score_val = getattr(row, SCORE_NAME)
                img = Image.open(row.image_path)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Top {i+1}\nScore: {score_val:.4f}")
                axes[0, i].axis('off')
            for i, row in enumerate(bottom_5.itertuples()):
                # Use getattr to access score column dynamically
                score_val = getattr(row, SCORE_NAME)
                img = Image.open(row.image_path)
                axes[1, i].imshow(img)
                axes[1, i].set_title(f"Bottom {i+1}\nScore: {score_val:.4f}")
                axes[1, i].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(OUTPUT_TOP_BOTTOM_FIG_PATH)
            plt.close()
            print(f"Top and bottom images visualization saved to {OUTPUT_TOP_BOTTOM_FIG_PATH}")
        except Exception as e:
            print(f"Could not save top/bottom images visualization: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate HPSv2, PickScore, or ImageReward for a directory of images against a list of prompts.')
    
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Directory containing the generated images.')
    parser.add_argument('--prompt-csv-path', type=str, 
                        default='/workspace/jh/flux/eval/prompts_250.csv',
                        help='Path to the CSV file containing prompts.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where the output CSV and plots will be saved.')
    parser.add_argument('--scorer', type=str, default='hpsv2',
                        choices=['hpsv2', 'pickscore', 'imagereward'],
                        help='The scoring model to use.')

    args = parser.parse_args()
    
    if not os.path.isdir(args.img_dir):
        print(f"Error: Image directory not found at {args.img_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyze_scores(args.img_dir, args.prompt_csv_path, args.output_dir, args.scorer)


if __name__ == '__main__':
    main()
