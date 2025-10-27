# python workspace/jun/nec_lat/rad_dino/rad_dino_zeroshot/zero_shot.py
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    recall_score
)

# ─── Add RAD-DINO code directory to path ────────────────────────────────────────
# Assumes the rad_dino_code directory is one level up and then inside rad_dino_code
rad_dino_code_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../rad_dino_code")
)
if rad_dino_code_path not in sys.path:
    sys.path.insert(0, rad_dino_code_path)

# Make sure RADDINO_LateralDatasetPNG can be imported
# If it's not directly in rad_dino_code, adjust the path or ensure it's importable
try:
    from rdino_dataset import RADDINO_LateralDatasetPNG # your dataset class
except ImportError:
    print(f"Error: Could not import RADDINO_LateralDatasetPNG.")
    print(f"Attempted to add {rad_dino_code_path} to sys.path")
    print(f"Ensure 'rdino_dataset.py' containing 'RADDINO_LateralDatasetPNG' exists there.")
    sys.exit(1)

# ─── Device and processor ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino", do_rescale=False)
except Exception as e:
    print(f"Error loading AutoImageProcessor: {e}")
    print("Please ensure you have internet connectivity or the model is cached.")
    sys.exit(1)

# ─── Helpers for confidence intervals ───────────────────────────────────────────
def calculate_ci(metric, n, z=1.96): # Wilson Score Interval for proportions
    if n == 0: # Avoid division by zero if a class has no samples
        return 0.0, 0.0
    # Ensure metric is within [0, 1]
    metric = np.clip(metric, 0.0, 1.0)
    den = 1 + z**2 / n
    center = (metric + z**2 / (2*n)) / den
    term_under_sqrt = metric*(1-metric)/n + z**2/(4*n**2)
    # Handle potential negative values due to floating point errors very close to 0 or 1
    if term_under_sqrt < 0:
         term_under_sqrt = 0
    half = z * np.sqrt(term_under_sqrt) / den
    lower = center - half
    upper = center + half
    # Ensure bounds are within [0, 1]
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def calculate_auc_ci(y_true, y_probs, seed=42, n_bootstraps=1000, alpha=0.95):
    bootstrapped = []
    # Ensure y_true and y_probs are numpy arrays for consistent indexing
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    
    rng = np.random.RandomState(seed)
    n = len(y_probs)
    
    # Check if there's only one class present in y_true initially
    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class present in y_true. AUC is not defined.")
        # Return NaN or some indicator that AUC is undefined.
        # Returning point estimate might be misleading. For CI, maybe return (NaN, NaN) or (0.5, 0.5)?
        # Let's return (np.nan, np.nan) as CI cannot be calculated.
        try:
            auc_score = roc_auc_score(y_true, y_probs)
            return auc_score, (np.nan, np.nan) # Return point estimate if calculable, but CI as NaN
        except ValueError:
             return np.nan, (np.nan, np.nan) # Return NaN if point estimate also fails


    for _ in range(n_bootstraps):
        idxs = rng.randint(0, n, n)
        # Ensure indices are valid if using lists, not needed for numpy arrays
        sample_true = y_true[idxs]
        sample_scores = y_probs[idxs]
        
        # Check if the bootstrap sample has both classes
        if len(np.unique(sample_true)) < 2:
            # Skip this bootstrap sample as AUC is not defined
            continue
            
        try:
             bootstrapped.append(roc_auc_score(sample_true, sample_scores))
        except ValueError:
             # This might happen in rare cases with specific bootstrap samples
             continue

    if not bootstrapped:
        # This can happen if all bootstrap samples had only one class
        print("Warning: Could not compute bootstrap AUC CIs; samples might lack class diversity.")
        try:
            val = roc_auc_score(y_true, y_probs)
            return val, (np.nan, np.nan) # Return point estimate, but NaN CI
        except ValueError:
             return np.nan, (np.nan, np.nan) # Return NaN if point estimate also fails

    sorted_scores = np.sort(bootstrapped)
    lower_pct = (1.0 - alpha) / 2.0 * 100
    upper_pct = (1.0 + alpha) / 2.0 * 100
    lower_bound = np.percentile(sorted_scores, lower_pct)
    upper_bound = np.percentile(sorted_scores, upper_pct)
    
    # Calculate the point estimate on the original data
    try:
         point_estimate = roc_auc_score(y_true, y_probs)
    except ValueError:
         point_estimate = np.nan # Handle case where original data might also fail

    return point_estimate, (lower_bound, upper_bound)

# ─── Metric computation with CIs ───────────────────────────────────────────────
def compute_metrics(true_labels, pred_labels, scores):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    scores = np.asarray(scores)

    n     = len(true_labels)
    n_pos = int(np.sum(true_labels == 1))
    n_neg = n - n_pos

    acc  = accuracy_score(true_labels, pred_labels)
    # Use recall_score for sensitivity (TPR) and specificity (TNR)
    # Sensitivity = Recall of positive class (label 1)
    sens = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    # Specificity = Recall of negative class (label 0)
    spec = recall_score(true_labels, pred_labels, pos_label=0, zero_division=0)

    # Get AUC point estimate and CI
    auc_score, (auc_lo, auc_hi) = calculate_auc_ci(true_labels, scores)
    if np.isnan(auc_score): # Handle case where AUC couldn't be calculated
         auc_score_str = "Undefined"
         auc_ci_str = "Undefined"
    else:
         auc_score_str = f"{auc_score:.4f}"
         if np.isnan(auc_lo):
              auc_ci_str = "Could not compute CI"
         else:
              auc_ci_str = f"{auc_lo:.4f}–{auc_hi:.4f}"


    acc_lo, acc_hi   = calculate_ci(acc,  n)
    sens_lo, sens_hi = calculate_ci(sens, n_pos) # CI based on number of positives
    spec_lo, spec_hi = calculate_ci(spec, n_neg) # CI based on number of negatives

    # Handle potential UndefinedMetricWarning in classification_report
    # It uses zero_division='warn' by default
    try:
        report = classification_report(true_labels, pred_labels, digits=4, zero_division=0)
    except Exception as e:
        report = f"Could not generate classification report: {e}"

    # Format CI strings carefully, especially for sensitivity/specificity if n_pos/n_neg is 0
    sens_ci_str = f"{sens_lo:.4f}–{sens_hi:.4f}" if n_pos > 0 else "N/A (0 pos samples)"
    spec_ci_str = f"{spec_lo:.4f}–{spec_hi:.4f}" if n_neg > 0 else "N/A (0 neg samples)"

    return (
        f"Accuracy:    {acc:.4f} (95% CI: {acc_lo:.4f}–{acc_hi:.4f})\n"
        f"Sensitivity: {sens:.4f} (95% CI: {sens_ci_str})\n"
        f"Specificity: {spec:.4f} (95% CI: {spec_ci_str})\n"
        f"ROC AUC:     {auc_score_str} (95% CI: {auc_ci_str})\n\n"
        f"Classification Report (zero_division=0):\n{report}\n"
    )

# ─── Argument parser ───────────────────────────────────────────────────────────
def get_args_zero_shot():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Inference on Internal and External Datasets"
    )
    parser.add_argument(
        '--internal_path', type=str, # Make paths required
        default='/workspace/yeonsu/0.Projects/Pneumoperitoneum/data/internal_data/Internal6.csv', # Example default
        help='Path to CSV with internal data (must include data_split_kfold column)'
    )
    parser.add_argument(
        '--external_path', type=str, # Make paths required
        default='/workspace/yeonsu/0.Projects/Pneumoperitoneum/data/external_data/External5.csv', # Example default
        help='Path to CSV with external data'
    )
    parser.add_argument('--gpu', type=str, default='0', help="GPU id to use (e.g., '0', '1', '0,1')")
    parser.add_argument('--batch', type=int, default=16, help='Batch size for inference') # Increase default batch
    parser.add_argument('--size',  type=int, default=518, help='Target image size for processor')
    # Gamma percentage seems unused in this script, maybe remove or clarify?
    # parser.add_argument('--gamma_percentage', type=float, default=0.5)
    # Add output directory argument
    parser.add_argument('--output_dir', type=str, default='/workspace/jun/nec_lat/rad_dino/rad_dino_zeroshot/zero_shot_metrics_with_youden', help='Directory to save outputs')
    # Add argument for base path if needed for image paths in CSV
    parser.add_argument('--base_image_path', type=str, default='', help='Base path to prepend to image paths in CSV if they are relative')

    return parser.parse_args()

# ─── Custom Zero-Shot Dataset ──────────────────────────────────────────────────
class ZeroShotDataset(RADDINO_LateralDatasetPNG):
    def __init__(self, df, args, processor, training=False):
        # Pass necessary args to superclass; adapt based on RADDINO_LateralDatasetPNG's __init__
        # Assuming it needs df, args (for size maybe?), and potentially base_path
        # You might need to modify this based on the actual signature of RADDINO_LateralDatasetPNG
        
        # If RADDINO_LateralDatasetPNG handles image loading and basic transforms:
        # super().__init__(df, args, training) # Or however it's called

        # --- If RADDINO_LateralDatasetPNG only gives paths/labels, handle loading here ---
        # This example assumes RADDINO_LateralDatasetPNG provides the image data structure needed by the processor
        # and potentially the label. Let's store df and args needed.
        self.df = df
        self.args = args
        self.processor = processor
        self.base_image_path = args.base_image_path if args.base_image_path else ''

        # --- If RADDINO_LateralDatasetPNG needs to be instantiated differently ---
        # Example: Maybe it needs the path directly
        # super().__init__(csv_path=None, df=df, args=args, training=training) # Adjust as needed
        
        # Simplified Example: Let's assume we just need df, args, processor
        # And that RADDINO_LateralDatasetPNG itself isn't strictly needed if we reimplement __getitem__ fully
        print(f"ZeroShotDataset created with {len(df)} samples.")


    def __len__(self):
         return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label from dataframe
        row = self.df.iloc[idx]
        # Construct full image path if necessary
        img_path = row['png_path'] # Assuming 'png_path' column exists
        if self.base_image_path and not os.path.isabs(img_path):
             img_path = os.path.join(self.base_image_path, img_path)
             
        label = row['Binary_Label'] # Assuming 'Binary_Label' column exists

        # Load image - Adapt this based on how RADDINO_LateralDatasetPNG loads images
        # Option A: If RADDINO_LateralDatasetPNG has a method to load/process
        # sample = super().__getitem__(idx) # Get processed sample from parent
        # image_for_processor = sample['image'] # Assuming parent returns tensor/numpy

        # Option B: Load image directly here (using PIL example)
        try:
            from PIL import Image
            # Ensure image is RGB for most vision transformers
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
             print(f"ERROR: Image not found at {img_path}")
             # Return dummy data or raise error, here returning None to be handled later
             return None # Needs handling in the inference loop
        except Exception as e:
             print(f"ERROR: Could not load image {img_path}: {e}")
             return None # Needs handling

        # Process image using HuggingFace processor
        # The processor typically handles resizing, normalization, and tensor conversion
        try:
            # Processor expects a PIL image or numpy array
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed['pixel_values'][0] # Remove batch dim added by processor
        except Exception as e:
            print(f"ERROR: Could not process image {img_path}: {e}")
            return None # Needs handling

        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.long) # Ensure label is a tensor
        }

# ─── Zero-Shot Inference (embeddings) ─────────────────────────────────────────
def zero_shot_inference(model, data_loader, device):
    model.eval()
    embeddings = []
    true_labels_list = [] # Collect labels during inference
    
    print(f"Starting zero-shot inference on {len(data_loader.dataset)} samples...")
    batch_num = 0
    with torch.no_grad():
        for batch in data_loader:
            # Handle potential None items from dataset __getitem__
            if batch is None or 'pixel_values' not in batch or batch['pixel_values'] is None:
                 print(f"Warning: Skipping problematic batch or item.")
                 # If batch size is > 1, this might skip the whole batch.
                 # Consider modifying DataLoader's collate_fn to handle Nones if needed.
                 continue # Skip this batch/item

            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'] # Keep labels on CPU or move as needed

            try:
                outputs = model(pixel_values=pixel_values)
                # Typically use pooler_output or last_hidden_state[:, 0] (CLS token)
                cls_emb = outputs.pooler_output.cpu().numpy()
                embeddings.extend(cls_emb)
                true_labels_list.extend(labels.cpu().numpy()) # Collect labels from batch
                batch_num += 1
                if batch_num % 50 == 0: # Print progress
                     print(f"  Processed {batch_num * data_loader.batch_size}/{len(data_loader.dataset)} samples...")
            except Exception as e:
                 print(f"Error during model inference on a batch: {e}")
                 # Decide how to handle: skip batch, store NaNs, etc.
                 # For now, we just print and continue; results might be incomplete.
                 continue

    print("Inference complete.")
    return embeddings, true_labels_list

# ─── Latency measurement ──────────────────────────────────────────────────────
def measure_latency(model, device, processor, input_size=518, repeats=30, warmup=10):
    model.eval()
    # Create a dummy input matching the processor's expected output format
    # Usually (batch_size, num_channels, height, width)
    try:
         # Generate a dummy PIL image, process it, then use its shape
         from PIL import Image
         dummy_pil = Image.new('RGB', (input_size, input_size))
         dummy_input_processed = processor(images=dummy_pil, return_tensors="pt")
         dummy_tensor = dummy_input_processed['pixel_values'].to(device)
         input_shape = dummy_tensor.shape
         print(f"Measuring latency using dummy input shape: {input_shape}")
    except Exception as e:
         print(f"Could not create dummy input for latency measurement: {e}")
         # Fallback to a common shape if processor failed
         input_shape=(1, 3, input_size, input_size)
         dummy_tensor = torch.randn(input_shape, device=device)
         print(f"Warning: Using fallback dummy input shape: {input_shape}")


    timings = []
    with torch.no_grad():
        # Warmup runs
        print(f"Warming up for {warmup} iterations...")
        for _ in range(warmup):
            _ = model(dummy_tensor)
        
        # Timing runs
        print(f"Measuring latency over {repeats} iterations...")
        for i in range(repeats):
            if device.type == 'cuda':
                 start = torch.cuda.Event(enable_timing=True)
                 end   = torch.cuda.Event(enable_timing=True)
                 start.record()
                 _ = model(dummy_tensor)
                 end.record()
                 torch.cuda.synchronize() # Wait for the operation to complete
                 timings.append(start.elapsed_time(end)) # Time in milliseconds
            else: # CPU timing
                 import time
                 start_time = time.perf_counter()
                 _ = model(dummy_tensor)
                 end_time = time.perf_counter()
                 timings.append((end_time - start_time) * 1000) # Time in milliseconds
            if (i+1) % 10 == 0: print(f"  Completed {i+1}/{repeats} timing runs...")

    if not timings:
         print("Warning: No timings recorded for latency.")
         return np.nan
         
    avg_latency = np.mean(timings)
    print(f"Latency measurement done. Average: {avg_latency:.2f} ms")
    return avg_latency

# ─── Run one pass (DF → y_true, scores, embeddings) ──────────────────────────
def run_zero_shot_on_df(model, df, args, processor, device):
    df = df.copy()
    # Path modification like replacing '/home/brody9512' might be needed here if not handled elsewhere
    # df['png_path'] = df['png_path'].apply(lambda x: x.replace('/home/brody9512', args.base_image_path or '')) # Example adjustment

    # Pass processor to the dataset
    dataset = ZeroShotDataset(df, args, processor, training=False)
    if not dataset: # Check if dataset initialization failed
        print("Error: Dataset could not be initialized.")
        return None, None, None

    # Handle cases where dataset might be empty
    if len(dataset) == 0:
         print("Warning: Dataset is empty. Skipping inference.")
         return [], [], []

    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    embeddings, true_labels = zero_shot_inference(model, loader, device)

    if not embeddings: # Check if inference returned empty lists
        print("Warning: Inference did not produce embeddings or labels.")
        return [], [], []

    # Use the first element of the embedding as the score (as per original code)
    # This assumes the first element has discriminative power for the binary task.
    # Consider alternative ways to get a score if needed (e.g., linear probe head if available, cosine similarity to text prompts, etc.)
    scores = [emb[0] for emb in embeddings]

    return true_labels, scores, embeddings

# ─── Save predictions & embeddings ────────────────────────────────────────────
def save_predictions(out_dir, prefix, true_labels, scores, embeddings):
    os.makedirs(out_dir, exist_ok=True)
    # Ensure data is available before trying to save
    if not true_labels or not scores:
         print(f"Warning: No true labels or scores to save for prefix '{prefix}'. Skipping prediction file.")
    else:
        try:
            with open(os.path.join(out_dir, f"{prefix}_predictions.txt"), "w") as f:
                f.write("y_true:\n" + str(list(true_labels)) + "\n") # Convert numpy arrays if needed
                f.write("y_prob (raw scores used):\n" + str([round(float(s), 5) for s in scores]) + "\n")
            print(f"Saved predictions to {os.path.join(out_dir, f'{prefix}_predictions.txt')}")
        except Exception as e:
             print(f"Error saving predictions for prefix '{prefix}': {e}")

    if embeddings is None or len(embeddings) == 0:
         print(f"Warning: No embeddings to save for prefix '{prefix}'. Skipping embedding files.")
    else:
        try:
            embeddings_array = np.array(embeddings)
            labels_array = np.array(true_labels)
            np.savez(os.path.join(out_dir, f"{prefix}_embeddings.npz"),
                    embeddings=embeddings_array,
                    labels=labels_array)
            print(f"Saved embeddings (binary) to {os.path.join(out_dir, f'{prefix}_embeddings.npz')}")

            # Optionally save embeddings to text (can be very large)
            # with open(os.path.join(out_dir, f"{prefix}_embeddings.txt"), "w") as f:
            #     f.write("Embeddings:\n")
            #     f.write(np.array2string(embeddings_array, separator=", ") + "\n")
            #     f.write("Labels:\n")
            #     f.write(np.array2string(labels_array, separator=", ") + "\n")
            # print(f"Saved embeddings (text) to {os.path.join(out_dir, f'{prefix}_embeddings.txt')}")

        except Exception as e:
            print(f"Error saving embeddings for prefix '{prefix}': {e}")


# ─── Save metrics with Youden threshold ─────────────────────────────────────────
def save_metrics(out_dir, prefix, model, device, processor, model_info, true_labels, scores, threshold=None, input_size=518):
    os.makedirs(out_dir, exist_ok=True)

    # Ensure scores and labels are available and valid
    if not scores or not true_labels or len(np.unique(true_labels)) < 2:
        print(f"Warning: Not enough data or class diversity to compute metrics for prefix '{prefix}'.")
        metrics_txt = model_info + "Metrics could not be computed (insufficient data or class diversity).\n"
        auc_value = np.nan
        pred_labels = [] # No predictions can be made
    else:
        true_labels = np.asarray(true_labels)
        scores = np.asarray(scores)
        # Determine prediction labels based on threshold
        if threshold is not None:
            pred_labels = (scores >= threshold).astype(int)
            header = f"Threshold (Youden): {threshold:.4f}\n\n"
        else:
            # Default thresholding if none provided (e.g., 0.0 based on original code's logic if no threshold calc needed)
            # Let's make it explicit: If no threshold is passed, maybe just report AUC?
            # Or use a default like 0.0 (carefully!)
            # Current implementation REQUIRES a threshold to be passed if threshold-dependent metrics are needed.
            # If only AUC is desired, the calling code should handle that.
            # For this function, let's assume threshold is usually provided (like from Youden).
            # If threshold is None, we maybe shouldn't calculate threshold-based metrics.
            # Reverting to original logic: Use 0.0 if threshold is None
            # THIS IS RISKY - The scale of raw embeddings[0] might not make 0.0 meaningful.
            # BETTER: Calculate metrics only if threshold is not None?
            # Let's stick to the code's apparent intent: calculate metrics IF threshold is passed.

            # If threshold is None, maybe we only calculate AUC? Let's refine.
            # For now, assume threshold IS provided when this function is called for full metrics.
            # If threshold is None, we perhaps just skip the thresholded metrics?
            # The original code had a fallback if threshold was None. Let's keep it for consistency, but warn.
             if threshold is None:
                  print("Warning: No threshold provided to save_metrics. Using 0.0 as default threshold for Acc/Sens/Spec.")
                  print("         This might not be meaningful for raw embedding scores.")
                  threshold = 0.0 # Fallback, use with caution!
                  pred_labels = (scores >= threshold).astype(int)
                  header = f"Threshold (Default Fallback): {threshold:.4f}\n\n"


        metrics_results = compute_metrics(true_labels, pred_labels, scores)
        metrics_txt = header + model_info + metrics_results
        
        # Extract AUC score for efficiency metrics (handle potential NaN)
        auc_value = roc_auc_score(true_labels, scores) if len(np.unique(true_labels)) > 1 else np.nan

    # --- Efficiency Metrics ---
    latency_ms = np.nan
    peak_mem_mb = np.nan
    try:
         latency_ms = measure_latency(model, device, processor, input_size=input_size)
    except Exception as e:
         print(f"Could not measure latency: {e}")
         
    if device.type == 'cuda':
        try:
            # Reset peak memory stats if possible (depends on PyTorch version/CUDA context)
            # torch.cuda.reset_peak_memory_stats(device) # Might be needed before inference if measuring there
            peak_mem_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem_mb = peak_mem_bytes / (1024**2)
        except Exception as e:
            print(f"Could not measure peak GPU memory: {e}")
            
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Avoid division by zero for efficiency metrics
    auc_per_total_mparam = (auc_value / (total_params / 1e6)) if total_params > 0 and not np.isnan(auc_value) else np.nan
    auc_per_trainable_mparam = (auc_value / (trainable_params / 1e6)) if trainable_params > 0 and not np.isnan(auc_value) else np.nan

    metrics_txt += (
        "\nEfficiency Metrics:\n"
        f"AUC per million total params:     {auc_per_total_mparam:.4f}\n"
        f"AUC per million trainable params: {auc_per_trainable_mparam:.4f}\n"
        f"Avg inference latency:            {latency_ms:.2f} ms/image\n"
    )
    if device.type == 'cuda':
         metrics_txt += f"Peak GPU memory usage:            {peak_mem_mb:.2f} MB\n"
    else:
         metrics_txt += "Peak GPU memory usage:            N/A (CPU execution)\n"

    # --- Save ---
    try:
        filepath = os.path.join(out_dir, f"{prefix}_metrics.txt")
        with open(filepath, "w") as f:
            f.write(metrics_txt)
        print(f"Saved metrics to {filepath}")
    except Exception as e:
        print(f"Error saving metrics for prefix '{prefix}': {e}")

    torch.cuda.reset_peak_memory_stats(device)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args_zero_shot()
    # Ensure the GPU environment variable is set based on args.gpu BEFORE loading model to device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Update device based on actual availability AFTER setting CUDA_VISIBLE_DEVICES
    global device # Modify the global device variable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor once
    global processor
    try:
        processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino", do_rescale=False)
    except Exception as e:
        print(f"Error loading AutoImageProcessor: {e}")
        sys.exit(1)

    print("Loading model...")
    try:
        model = AutoModel.from_pretrained("microsoft/rad-dino").to(device)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model from HuggingFace: {e}")
        sys.exit(1)

    # Model info header
    total_params       = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_bytes      = sum(p.numel() * p.element_size() for p in model.parameters())
    model_mb         = model_bytes / (1024**2)
    model_info = (
        f"Model: microsoft/rad-dino\n"
        f"Total parameters:           {total_params:,}\n"
        f"Trainable parameters:       {trainable_params:,}\n"
        f"Model size (parameters):    {model_mb:.2f} MB\n\n"
    )
    print(model_info)

    out_dir = args.output_dir
    print(f"Output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    thr_val = None
    
    # --- Step 1: Process Internal Test Dataset to get scores ---
    print(f"\n--- Processing Internal Test Dataset for Scores: {args.internal_path} ---")
    y_int, s_int, emb_int = None, None, None # Initialize
    thr_int = None # Initialize threshold to None
    internal_results_valid = False # Flag to check if we got valid internal results

    try:
        df_int = pd.read_csv(args.internal_path)
        if 'Binary_Label' not in df_int.columns or 'png_path' not in df_int.columns:
             raise ValueError("Internal CSV must contain 'Binary_Label' and 'png_path' columns.")
        if 'data_split_kfold' not in df_int.columns:
            raise ValueError("Internal CSV must contain 'data_split_kfold' column.")

        val_df = df_int[df_int['data_split_kfold'] == 'valid'].copy()
        test_df = df_int[df_int['data_split_kfold'] == 'test'].copy()
        if test_df.empty:
            print("Warning: No samples found with data_split_kfold == 'test' in internal data.")
        else:
            print(f"Found {len(test_df)} samples in the internal 'test' split.")
            y_val, s_val, emb_val = run_zero_shot_on_df(model, val_df, args, processor, device)
            # 3) compute Youden threshold ON VAL
            fpr_v, tpr_v, th_v = roc_curve(y_val, s_val)
            youden_v = np.argmax(tpr_v - fpr_v)
            thr_val  = th_v[youden_v]
            print(f"🔥 Youden threshold (from VAL): {thr_val:.4f}")
            
            y_int, s_int, emb_int = run_zero_shot_on_df(model, test_df, args, processor, device) # Pass processor/device

            if y_int is not None and s_int is not None and len(y_int) > 0:
                 internal_results_valid = True
                 save_predictions(out_dir, "internal_test_zero_shot", y_int, s_int, emb_int) # Save predictions here

                 # # --- Step 2: Calculate Youden threshold FROM INTERNAL data ---
                 # if len(np.unique(y_int)) > 1:
                 #     try:
                 #         fpr_i, tpr_i, ths_i = roc_curve(y_int, s_int)
                 #         youden_i            = np.argmax(tpr_i - fpr_i)
                 #         thr_int             = ths_i[youden_i] # Store the internal threshold
                 #         print(f"Determined Internal Youden Threshold: {thr_int:.4f}")
                 #     except Exception as e:
                 #         print(f"Could not calculate Youden threshold from internal data: {e}")
                 #         thr_int = None # Ensure it's None if calculation fails
                 # else:
                 #     print("Warning: Only one class in internal test dataset labels. Cannot compute Youden threshold.")
            else:
                 print("Skipping threshold calculation and metrics for internal dataset due to lack of results.")

    except FileNotFoundError:
        print(f"Error: Internal data file not found at {args.internal_path}")
    except ValueError as ve:
         print(f"Error processing internal data: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during internal data processing: {e}")


    # --- Step 3: Process External Dataset to get scores ---
    print(f"\n--- Processing External Dataset for Scores: {args.external_path} ---")
    y_ext, s_ext, emb_ext = None, None, None # Initialize
    external_results_valid = False # Flag

    try:
        df_ext = pd.read_csv(args.external_path)
        if 'Binary_Label' not in df_ext.columns or 'png_path' not in df_ext.columns:
             raise ValueError("External CSV must contain 'Binary_Label' and 'png_path' columns.")

        print(f"Loaded external data: {len(df_ext)} samples.")
        y_ext, s_ext, emb_ext = run_zero_shot_on_df(model, df_ext, args, processor, device) # Pass processor/device

        if y_ext is not None and s_ext is not None and len(y_ext) > 0:
            external_results_valid = True
            save_predictions(out_dir, "external_zero_shot", y_ext, s_ext, emb_ext) # Save predictions here
        else:
            print("Skipping metrics saving for external dataset due to lack of results.")

    except FileNotFoundError:
        print(f"Error: External data file not found at {args.external_path}")
    except ValueError as ve:
         print(f"Error processing external data: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during external data processing: {e}")


    # --- Step 4: Save Metrics using the SINGLE internal threshold ---

    # --- Save Internal Metrics ---
    if internal_results_valid:
        print(f"\n--- Saving Internal Metrics (using threshold: {thr_val}) ---")
        # Reset memory before this metric calculation if desired
        if device.type == 'cuda':
            print("Resetting peak GPU memory stats before internal metrics calculation...")
            torch.cuda.reset_peak_memory_stats(device)
        # Use thr_val (which might be None if calculation failed)
        save_metrics(out_dir, "internal_test_zero_shot", model, device, processor, model_info, y_int, s_int, threshold=thr_val, input_size=args.size)
    else:
        print("\nSkipping internal metrics saving as internal results were not valid.")


    # --- Save External Metrics ---
    if external_results_valid:
        print(f"\n--- Saving External Metrics (using **internal** threshold: {thr_val}) ---")
        if thr_val is None:
             print("Warning: Internal threshold was not determined. External threshold-based metrics cannot be calculated using the internal threshold.")
             # Option: Calculate external metrics only with AUC (threshold=None lets save_metrics handle it, likely falling back to 0.0 or skipping parts)
             # save_metrics(out_dir, "external_zero_shot", model, device, processor, model_info, y_ext, s_ext, threshold=None, input_size=args.size)
             # Option: Skip saving metrics entirely if threshold is missing
             print("Skipping external metrics saving as required internal threshold is missing.")
        else:
            # Reset memory before this metric calculation if desired (measures peak since last reset)
            if device.type == 'cuda':
                print("Resetting peak GPU memory stats before external metrics calculation...")
                torch.cuda.reset_peak_memory_stats(device)
            # Use thr_val for external evaluation
            save_metrics(out_dir, "external_zero_shot", model, device, processor, model_info, y_ext, s_ext, threshold=thr_val, input_size=args.size)
    else:
        print("\nSkipping external metrics saving as external results were not valid.")


    print("\n--- Zero-shot evaluation script finished ---")


if __name__ == "__main__":
    main()