import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor):
    # tensor has shape (B, N, C) where N includes the CLS token.
    B, N, C = tensor.shape
    # Remove the CLS token.
    token_seq = tensor[:, 1:, :]
    # Compute the spatial size: assume square grid.
    spatial_dim = int((N - 1) ** 0.5)
    assert spatial_dim * spatial_dim == (N - 1), "Number of tokens (excluding CLS) is not a perfect square"
    result = token_seq.reshape(B, spatial_dim, spatial_dim, C)
    # Bring channels to first dimension: result shape (B, C, spatial_dim, spatial_dim)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def process_gradcam_batch(model, inputs, grad_cam, DEVICE, threshold, rounding_precision=5):
    """
    Run inference on one batch and compute Grad-CAM for classification.
    """
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        cls_pred = model(inputs)
        cls_pred = torch.sigmoid(cls_pred)
        cls_pred_bin = (cls_pred > threshold).float()
        cls_pred_rounded = round(cls_pred.item(), rounding_precision)

    # Remove the batch dimension and prepare the input image.
    inputs_squeezed = inputs.squeeze(0)
    inputs_np = np.transpose(inputs_squeezed.cpu().numpy(), (1, 2, 0))

    # Compute Grad-CAM mask.
    grayscale_cam = grad_cam(input_tensor=inputs)
    min_val = np.min(grayscale_cam)
    max_val = np.max(grayscale_cam)
    if max_val - min_val != 0:
        grayscale_cam = np.uint8(255 * (grayscale_cam - min_val) / (max_val - min_val))
    else:
        grayscale_cam = np.zeros_like(grayscale_cam, dtype=np.uint8)
    grayscale_cam = np.squeeze(grayscale_cam)

    # Overlay heatmap using OpenCV's COLORMAP_JET.
    colormap = cv2.COLORMAP_JET
    visualization_g = show_cam_on_image(inputs_np, grayscale_cam / 255, use_rgb=True, colormap=colormap)
    
    # Ensure the input image is RGB.
    if inputs_np.ndim == 2 or (inputs_np.ndim == 3 and inputs_np.shape[2] == 1):
        inputs_np = cv2.cvtColor(inputs_np, cv2.COLOR_GRAY2RGB)

    # Inside process_gradcam_batch:
    visualization_g = show_cam_on_image(inputs_np, grayscale_cam / 255, use_rgb=True, colormap=colormap)
    print(f"\n--- GradCAM Output ---")
    print(f"visualization_g - dtype: {visualization_g.dtype}, shape: {visualization_g.shape}")
    print(f"visualization_g - min: {np.min(visualization_g)}, max: {np.max(visualization_g)}")
    # Check a sample background pixel (e.g., top-left corner)
    print(f"visualization_g - sample pixel [0,0]: {visualization_g[0, 0, :]}")
        
    return {
        'cls_pred_bin': cls_pred_bin,
        'cls_pred_rounded': cls_pred_rounded,
        'inputs_np': inputs_np,
        'visualization_g': visualization_g
    }

############################# ViT-ReciproCAM #################################
class ViTReciproCAM:
    """
    Implementation of ViT-ReciproCAM for Vision Transformers.
    This version masks patches in the input image rather than tokens.
    MODIFIED: Clips negative importance scores before normalization.
              Patch-aware thresholding is disabled.
              Includes debugging prints.
    """
    def __init__(self, model, target_layers, reshape_transform=None, batch_size=16,
                 use_scorecam=True, score_threshold=0.33, patch_size=14):
        """
        Initializes the ViTReciproCAM instance.

        Args:
            model: The target PyTorch model.
            target_layers: List of target layers within the model for CAM methods
                           (used by ScoreCAM if enabled).
            reshape_transform: A function to reshape the model's output if needed
                               (common for ViTs).
            batch_size: Batch size used internally for ScoreCAM if enabled.
            use_scorecam: Boolean flag to enable/disable ScoreCAM pre-selection.
            score_threshold: Threshold used with ScoreCAM if enabled.
            patch_size: The size of the patches in the Vision Transformer.
        """
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.batch_size = batch_size # Used by ScoreCAM if enabled
        self.use_scorecam = use_scorecam
        self.score_threshold = score_threshold
        self.device = next(model.parameters()).device
        self.patch_size = patch_size
        self.model.eval() # Ensure model is in eval mode

        # Initialize ScoreCAM for patch pre-selection if required
        if self.use_scorecam:
            try:
                from pytorch_grad_cam import ScoreCAM # Lazy import
                self.score_cam = ScoreCAM(
                    model=self.model,
                    target_layers=self.target_layers,
                    reshape_transform=self.reshape_transform,
                    batch_size=self.batch_size # Pass batch_size here
                )
                print("ViTReciproCAM: ScoreCAM pre-selection enabled.")
            except ImportError:
                print("ViTReciproCAM Warning: pytorch_grad_cam not found, cannot use ScoreCAM.")
                self.use_scorecam = False
            except Exception as e:
                 print(f"ViTReciproCAM Error initializing ScoreCAM: {e}. Disabling ScoreCAM.")
                 self.use_scorecam = False
        else:
            print("ViTReciproCAM: ScoreCAM pre-selection disabled.")


    def __call__(self, input_tensor, targets=None):
        """
        Generates the ReciproCAM heatmap.

        Args:
            input_tensor (torch.Tensor): Input tensor batch (usually B=1).
            targets (list or None): Optional list containing target class indices or criteria.
                                     If None, defaults based on model prediction.

        Returns:
            np.ndarray: The generated CAM map, normalized approx [0, 1], shape [B, h, w].
        """
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        batch_size = input_tensor.shape[0]

        # Get original predictions
        with torch.no_grad():
            original_preds = self.model(input_tensor)

            # --- Determine Target Scores ---
            # Handle binary classification output (single logit or shape [B, 1])
            if len(original_preds.shape) == 1 or original_preds.shape[1] == 1:
                original_scores_prob = torch.sigmoid(original_preds).cpu().numpy().flatten()
                if targets is None:
                    # Default target for binary: use the class with score >= 0.5
                    # Ensure targets is shaped correctly for later indexing (list of lists/arrays)
                    targets = [[int(score >= 0.5)] for score in original_scores_prob] # List of single-element lists
                original_scores = original_scores_prob # Use sigmoid probability as the score

            # Handle multi-class classification output (shape [B, num_classes])
            else:
                original_scores_softmax = torch.nn.functional.softmax(original_preds, dim=1)
                if targets is None:
                    # Default target for multi-class: use the class with the highest score
                    targets = [original_scores_softmax.argmax(dim=1).cpu().tolist()] # List containing list of targets for the batch
                # Select the score corresponding to the target class for each item in the batch
                # Assuming targets[0] contains the list/array of target indices for the batch
                try:
                    original_scores = np.array([original_scores_softmax[i, target_idx].item() for i, target_idx in enumerate(targets[0])])
                except IndexError:
                    print(f"Error: Mismatch between batch size ({batch_size}) and targets provided ({targets}). Check target format.")
                    # Handle error - maybe raise exception or default target
                    raise # Re-raise for now
            # --- End Target Scores ---

        # Calculate patch grid dimensions based on image size
        img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]
        h = img_h // self.patch_size
        w = img_w // self.patch_size
        if h * self.patch_size != img_h or w * self.patch_size != img_w:
             print(f"Warning: Image dimensions ({img_h}x{img_w}) not perfectly divisible by patch size ({self.patch_size}). Grid size: {h}x{w}")


        # --- ScoreCAM Pre-selection (if enabled) ---
        if self.use_scorecam and hasattr(self, 'score_cam'):
            try:
                import cv2 # Lazy import for resize
                print("ViTReciproCAM: Running ScoreCAM for pre-selection...")
                # Pass targets to score_cam if appropriate for its implementation
                # Check if score_cam takes targets; assume it does based on GradCAM library patterns
                score_cam_targets = [t[0] for t in targets] # Assuming targets is list of lists, take first element
                score_maps = self.score_cam(input_tensor=input_tensor, targets=score_cam_targets)

                # Resize ScoreCAM maps to patch grid dimensions [B, h, w]
                resized_score_maps = []
                for b in range(batch_size):
                    map_to_resize = np.squeeze(score_maps[b])
                    if map_to_resize.ndim != 2:
                        print(f"Warning: ScoreCAM map for item {b} is not 2D (shape: {map_to_resize.shape}). Skipping resize.")
                        resized_map = np.ones((h, w)) # Fallback: Assume all patches are candidates
                    else:
                         resized_map = cv2.resize(map_to_resize, (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_score_maps.append(resized_map)
                score_maps_resized = np.array(resized_score_maps)

                # Normalize each resized ScoreCAM map individually to [0, 1]
                for b in range(batch_size):
                    map_min = score_maps_resized[b].min()
                    map_max = score_maps_resized[b].max()
                    if map_max > map_min:
                        score_maps_resized[b] = (score_maps_resized[b] - map_min) / (map_max - map_min)

                # Create mask of candidate patches based on threshold
                candidate_patches = score_maps_resized >= self.score_threshold
                print(f"ViTReciproCAM: Using ScoreCAM pre-selection. Threshold={self.score_threshold}")
            except Exception as e:
                 print(f"ViTReciproCAM Error during ScoreCAM pre-selection: {e}. Falling back to using all patches.")
                 candidate_patches = np.ones((batch_size, h, w), dtype=bool)
                 self.use_scorecam = False # Disable for safety if it failed
        else:
            # If not using ScoreCAM, ablate all patches
            candidate_patches = np.ones((batch_size, h, w), dtype=bool)
        # --- End ScoreCAM Pre-selection ---


        # Initialize CAM storage
        cams = np.zeros((batch_size, h, w), dtype=np.float32)

        # --- Ablate each patch and measure prediction change ---
        print(f"\n--- ViTReciproCAM: Starting Patch Ablation (grid={h}x{w}, use_scorecam={self.use_scorecam}) ---")
        patch_counter = 0 # Counter for processed patches
        skipped_patches = 0 # Counter for skipped patches

        for y in range(h):
            for x in range(w):
                # Check if ANY image in the batch requires this patch (based on ScoreCAM)
                # If using ScoreCAM, and this patch is below threshold for ALL images in batch, skip.
                if self.use_scorecam and not candidate_patches[:, y, x].any():
                    skipped_patches += 1
                    continue # Skip if no image needs this patch based on pre-selection

                patch_counter += 1
                masked_input = self._mask_patch(input_tensor.clone(), y, x)

                # Get prediction with masked input
                with torch.no_grad():
                    masked_preds = self.model(masked_input)

                    # --- Get Masked Scores (matching original score format) ---
                    if len(masked_preds.shape) == 1 or masked_preds.shape[1] == 1:
                        # Binary case
                        masked_scores = torch.sigmoid(masked_preds).cpu().numpy().flatten()
                    else:
                        # Multi-class case
                        masked_scores_softmax = torch.nn.functional.softmax(masked_preds, dim=1)
                        try:
                             # Select score for the original target class for each item
                             masked_scores = np.array([masked_scores_softmax[i, target_idx].item() for i, target_idx in enumerate(targets[0])])
                        except IndexError:
                            print(f"Error: Mismatch processing masked scores. Batch size {batch_size}, targets {targets}")
                            raise # Re-raise
                    # --- End Masked Scores ---

                # Calculate importance score: (S_orig - S_masked) / S_orig
                importance = (original_scores - masked_scores) / (np.maximum(original_scores, 1e-10))

                # --- DEBUG PRINT: Raw Importance ---
                if batch_size > 0 and patch_counter <= 4 : # Print for first few *processed* patches
                     print(f"DEBUG ReciproCAM: Patch(y={y}, x={x})[Item 0] -> Importance: {importance[0]:.6f} (OrigScore: {original_scores[0]:.4f}, MaskedScore: {masked_scores[0]:.4f})")
                # ------------------------------------

                # Update CAM map for each image in the batch
                for b in range(batch_size):
                    # Only update if this patch was a candidate for this specific image OR if not using scorecam
                    if not self.use_scorecam or candidate_patches[b, y, x]:
                        cams[b, y, x] = importance[b]

        print(f"--- ViTReciproCAM: Processed {patch_counter} patch locations (skipped {skipped_patches} based on ScoreCAM) ---")
        # --- End Patch Ablation ---


        # --- Process Final CAM Maps ---
        # --- Clip negative importance scores to zero ---
        if batch_size > 0:
             print(f"\nDEBUG ReciproCAM: Raw cams[0] before clip/norm - min: {np.min(cams[0]):.6f}, max: {np.max(cams[0]):.6f}, mean: {np.mean(cams[0]):.6f}")
        cams = np.maximum(cams, 0) # ReLU equivalent: set negative values to 0
        if batch_size > 0:
             print(f"DEBUG ReciproCAM: Raw cams[0] after clip(>=0) - min: {np.min(cams[0]):.6f}, max: {np.max(cams[0]):.6f}, mean: {np.mean(cams[0]):.6f}")
        # ----------------------------------------------

        # Normalize the now non-negative CAMs to [0, 1] range for each image
        for b in range(batch_size):
            cam_min = cams[b].min() # Guaranteed >= 0
            cam_max = cams[b].max()
            # Check if map has variation
            if cam_max > cam_min: # Effectively if cam_max > 0 now (since min >= 0)
                # Normalize [cam_min, cam_max] -> [0, 1].
                cams[b] = (cams[b] - cam_min) / (cam_max - cam_min)
            # else: map remains constant zero (min==max==0)

        # --- DEBUG PRINT: Normalized Cams Map ---
        if batch_size > 0:
             print(f"DEBUG ReciproCAM: cams[0] after final normalization - min: {np.min(cams[0]):.6f}, max: {np.max(cams[0]):.6f}, mean: {np.mean(cams[0]):.6f}")
        # ----------------------------------

        # --- Patch-Aware Thresholding - KEEP COMMENTED OUT ---
        # patch_std = np.std(cams, axis=(1,2))
        # cams = np.where(cams < patch_std[:,None,None], 0, cams)
        # -----------------------------------------------------

        print(f"--- ViTReciproCAM: Finished CAM processing ---")
        # Return final map(s), shape [B, h, w], float32 [0, 1]
        return cams


    def _mask_patch(self, input_tensor, y, x):
        """
        Masks a specific patch in the input tensor by setting its values to 0.

        Args:
            input_tensor (torch.Tensor): The input tensor (should be a clone).
            y (int): Row index of the patch.
            x (int): Column index of the patch.

        Returns:
            torch.Tensor: The tensor with the specified patch masked.
        """
        # Calculate pixel coordinates of the patch
        start_y = y * self.patch_size
        end_y = start_y + self.patch_size
        start_x = x * self.patch_size
        end_x = start_x + self.patch_size

        # Mask the patch (works for [B, C, H, W])
        input_tensor[:, :, start_y:end_y, start_x:end_x] = 0
        return input_tensor
        
def create_vit_reciprocam_scorecam(model, DEVICE):
    """
    Create a ViT-ReciproCAM instance for the model using ScoreCAM.
    """
    model.to(DEVICE)
    model.eval()

    # If model is wrapped in DataParallel, get the underlying model
    if hasattr(model, "module"):
        base_model = model.module
    else:
        base_model = model

    # Target the final norm1 layer of the last encoder layer
    if hasattr(base_model, "rad_dino"):
        target_layer = [base_model.rad_dino.encoder.layer[-1].norm1]
    else:
        raise AttributeError("Model does not have attribute 'rad_dino'")

    print("#########Target Layer of RADDINO for ViT-ReciproCAM with ScoreCAM#########")
    print(target_layer)

    # Create and return the ViT-ReciproCAM instance with ScoreCAM
    reciprocam = ViTReciproCAM(
        model=model, 
        target_layers=target_layer, 
        reshape_transform=reshape_transform,
        batch_size=6,  # Adjust based on available GPU memory
        use_scorecam=False,  # Enable ScoreCAM pre-selection for efficiency
        score_threshold=0.1  # prev 0.5 ######### Adjust threshold as needed
    )
    
    return reciprocam

def process_reciprocam_batch(model, inputs, reciprocam, DEVICE, threshold, rounding_precision=5, alpha=0.5):
    """
    Run inference on one batch and compute ReciproCAM for classification.
    ATTEMPT 7: More debugging for Manual Blending.
    """
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        cls_pred = model(inputs)
        cls_pred = torch.sigmoid(cls_pred)
        cls_pred_bin = (cls_pred > threshold).float()
        cls_pred_rounded = round(cls_pred.item(), rounding_precision)

    # --- Prepare Base Image ---
    inputs_squeezed = inputs.squeeze(0)
    inputs_np_float_hw1 = np.transpose(inputs_squeezed.cpu().numpy(), (1, 2, 0))
    base_image_rgb_float = cv2.cvtColor(inputs_np_float_hw1.squeeze(-1), cv2.COLOR_GRAY2RGB)
    if base_image_rgb_float.dtype != np.float32:
        base_image_rgb_float = base_image_rgb_float.astype(np.float32) / 255.0
    base_image_rgb_float = np.clip(base_image_rgb_float, 0, 1)
    # --- >>> ADDED DEBUG PRINT <<< ---
    print(f"\nDEBUG ReciproCAM: Base image value at [0,0] (float RGB): {base_image_rgb_float[0, 0, :]}")
    # ------------------------------

    # --- Prepare Heatmap ---
    grayscale_cam = reciprocam(input_tensor=inputs)
    img_height, img_width = base_image_rgb_float.shape[:2]
    grayscale_cam = np.squeeze(grayscale_cam)
    if grayscale_cam.shape[:2] != (img_height, img_width):
        grayscale_cam = cv2.resize(grayscale_cam, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    grayscale_cam_float32 = np.clip(grayscale_cam.astype(np.float32), 0, 1)
    # --- >>> ADDED DEBUG PRINT <<< ---
    print(f"DEBUG ReciproCAM: Heatmap value at [0,0] (float): {grayscale_cam_float32[0, 0]:.6f}")
    # ------------------------------
    print(f"DEBUG ReciproCAM: Heatmap overall min/max (float): min={np.min(grayscale_cam_float32):.6f}, max={np.max(grayscale_cam_float32):.6f}")


    # --- Manual Blending ---
    # 1. Apply colormap
    # --- >>> ADDED DEBUG PRINT <<< ---
    heatmap_input_val_at_00 = np.uint8(255 * grayscale_cam_float32[0, 0])
    print(f"DEBUG ReciproCAM: Input value to applyColorMap for [0,0]: {heatmap_input_val_at_00}")
    # ------------------------------
    heatmap_bgr_uint8 = cv2.applyColorMap(np.uint8(255 * grayscale_cam_float32), cv2.COLORMAP_JET)
    # --- >>> ADDED DEBUG PRINT <<< ---
    print(f"DEBUG ReciproCAM: Colormapped Heatmap value at [0,0] (uint8 BGR): {heatmap_bgr_uint8[0, 0, :]}")
    # ------------------------------

    # 2. Convert colormapped heatmap to RGB float [0, 1]
    colored_heatmap_float = cv2.cvtColor(heatmap_bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    print(f"DEBUG ReciproCAM: Colormapped Heatmap value at [0,0] (float RGB): {colored_heatmap_float[0, 0, :]}")

    # 3. Perform alpha blending
    blended_float = (alpha * colored_heatmap_float) + ((1 - alpha) * base_image_rgb_float)
    print(f"DEBUG ReciproCAM: Blended value at [0,0] (float RGB): {blended_float[0, 0, :]}")

    # 4. Clip final result and convert to uint8
    visualization_r = np.uint8(255 * np.clip(blended_float, 0, 1))
    # -----------------------

    # --- DEBUG PRINT: Check final visualization output ---
    print(f"\n--- ReciproCAM Output (Manual Blend) ---")
    print(f"visualization_r - dtype: {visualization_r.dtype}, shape: {visualization_r.shape}")
    print(f"visualization_r - min: {np.min(visualization_r)}, max: {np.max(visualization_r)}")
    print(f"visualization_r - sample pixel [0,0]: {visualization_r[0, 0, :]}") # This is the final result
    # ----------------------------------------------------

    return {
        'cls_pred_bin': cls_pred_bin,
        'cls_pred_rounded': cls_pred_rounded,
        'inputs_np': base_image_rgb_float,
        'visualization_g': visualization_r
    }
    
def generate_vit_reciprocam_visualizations_test(model, test_loader, DEVICE, threshold, save_dir, reader_study:bool=False, text=False):
    """
    Generate and save ViT-ReciproCAM visualizations using ScoreCAM for testing.
    """
    reciprocam = create_vit_reciprocam_scorecam(model, DEVICE)
    
    # Ensure the output directory exists
    if text and reader_study:
        subdir = "vit_reciprocam_readerstudy_text"
    elif reader_study:
        subdir = "vit_reciprocam_readerstudy"
    else:
        subdir = "vit_reciprocam"
    # subdir = "vit_reciprocam_readerstudy" if reader_study else "vit_reciprocam"
    reciprocam_dir = os.path.join(save_dir, subdir)
    if not os.path.exists(reciprocam_dir):
        os.makedirs(reciprocam_dir, exist_ok=True)

    for i, data in enumerate(test_loader):
        inputs = data['image']
        png_path = data['png_name']

        # get ground truth label from the batch & convert ground-truth value to human-readable string.
        actual_label_val = data['label'][0].item() if torch.is_tensor(data['label'][0]) else data['label'][0]
        actual_label = 'Pneumoperitoneum' if int(actual_label_val) == 1 else 'No Pneumoperitoneum'

        # Use the dedicated processing function
        outputs = process_reciprocam_batch(model, inputs, reciprocam, DEVICE, threshold, rounding_precision=4)
        
        # Determine predicted class label.
        clspred = 'Pneumoperitoneum' if int(outputs['cls_pred_bin'].item()) == 1 else 'Non Pneumoperitoneum'
        
        plt.figure(figsize=(7,7), dpi=114.1)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        # title_text = (
        #     f"Predicted: {clspred}\n"
        #     f"Likelihood: {outputs['cls_pred_rounded']}   Threshold: {threshold:.4f}"
        #     if reader_study
        #     else
        #     f"Predicted: {clspred}\n"
        #     f"Actual: {actual_label}\n"
        #     f"Likelihood: {outputs['cls_pred_rounded']}   Threshold: {threshold:.4f}"
        # )
        if text and reader_study:
            title_text = (f"Likelihood: {outputs['cls_pred_rounded']}")
            plt.title(title_text, fontsize=17)
            
        
        # plt.title(f"Predicted: {clspred}\nActual: {actual_label}\nLikelihood: {outputs['cls_pred_rounded']}   Threshold: {threshold}", fontsize=17)
        
        file_name = f"{png_path[0].split('.')[0]}_vitreciprocam.png"
        plt.savefig(os.path.join(reciprocam_dir, file_name), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
    print("ViT-ReciproCAM with ScoreCAM visualizations saved successfully!")

################################# ScoreCAM ##################################
def create_scorecam(model, DEVICE):
    # Wrap the model if not already wrapped
    model.to(DEVICE)
    model.eval()

    # If model is wrapped in DataParallel, get the underlying model
    if hasattr(model, "module"):
        base_model = model.module
    else:
        base_model = model

    # Target the final norm1 layer of the last encoder layer
    if hasattr(base_model, "rad_dino"):
        target_layer = [base_model.rad_dino.encoder.layer[-1].norm1]
    else:
        raise AttributeError("Model does not have attribute 'rad_dino'")

    print("#########Target Layer of RADDINO for ScoreCAM#########")
    print(target_layer)

    # Create and return the Score-CAM instance
    score_cam = ScoreCAM(
        model=model, 
        target_layers=target_layer, 
        reshape_transform=reshape_transform,
    )
    
    return score_cam

def generate_scorecam_visualizations_test(model, test_loader, DEVICE, threshold, save_dir):
    """
    Generate and save Score-CAM visualizations for testing.
    """
    score_cam = create_scorecam(model, DEVICE)
    
    # Ensure the output directory exists
    scorecam_dir = os.path.join(save_dir, "scorecam")
    if not os.path.exists(scorecam_dir):
        os.makedirs(scorecam_dir, exist_ok=True)

    for i, data in enumerate(test_loader):
        inputs = data['image']
        png_path = data['png_name']

        # get ground truth label from the batch & convert ground-truth value to human-readable string.
        actual_label_val = data['label'][0].item() if torch.is_tensor(data['label'][0]) else data['label'][0]
        actual_label = 'Pneumoperitoneum' if int(actual_label_val) == 1 else 'Non Pneumoperitoneum'

        outputs = process_gradcam_batch(model, inputs, score_cam, DEVICE, threshold, rounding_precision=4)
        
        # Determine predicted class label.
        clspred = 'Pneumoperitoneum' if int(outputs['cls_pred_bin'].item()) == 1 else 'Non Pneumoperitoneum'
        
        plt.figure(figsize=(7,7), dpi=114.1)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        plt.title(f"Predicted: {clspred}\nActual: {actual_label}\nLikelihood: {outputs['cls_pred_rounded']}   Threshold: {threshold:.4f}", fontsize=17)
        
        file_name = f"{png_path[0].split('.')[0]}_scorecam.png"
        plt.savefig(os.path.join(scorecam_dir, file_name), bbox_inches='tight', pad_inches=0.15)
        plt.close()
    
    print("Score-CAM visualizations saved successfully!")


##### MISC ############################################################################
####################################### GradCAM ###########################################
def create_gradcam(model, DEVICE):
    # Wrap the model if not already wrapped.
    model.to(DEVICE)
    model.eval()

    # If model is wrapped in DataParallel, get the underlying model.
    if hasattr(model, "module"):
        base_model = model.module
    else:
        base_model = model

    # Our RADDINO_Model stores the backbone under "rad_dino".
    # Wrap the final norm1 layer of the last encoder layer.
    if hasattr(base_model, "rad_dino"):
        target_layer = [base_model.rad_dino.encoder.layer[-1].norm1]
        # target_layers = [model.rad_dino.encoder.layer[-1].norm1]
    else:
        raise AttributeError("Model does not have attribute 'rad_dino'")

    print("#########Target Layer of RADDINO for GradCAM#########")
    print(target_layer)

    # Create and return the GradCAM instance.
    return GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)

def generate_gradcam_visualizations_test(model, test_loader, DEVICE, threshold, save_dir):
    """
    Generate and save Grad-CAM visualizations for testing.
    """
    grad_cam = create_gradcam(model, DEVICE)
    
    # Ensure the grad-CAM output directory exists.
    gradcam_dir = os.path.join(save_dir, "gradcam")
    if not os.path.exists(gradcam_dir):
        os.makedirs(gradcam_dir, exist_ok=True)
    
    for i, data in enumerate(test_loader):
        inputs = data['image']
        png_path = data['png_name']

        # get ground truth label from the batch & convert ground-truth value to human-readable string.
        actual_label_val = data['label'][0].item() if torch.is_tensor(data['label'][0]) else data['label'][0]
        actual_label = 'Pneumoperitoneum' if int(actual_label_val) == 1 else 'Non Pneumoperitoneum'

        outputs = process_gradcam_batch(model, inputs, grad_cam, DEVICE, threshold, rounding_precision=4)
        
        # Determine predicted class label.
        clspred = 'Pneumoperitoneum' if int(outputs['cls_pred_bin'].item()) == 1 else 'Non Pneumoperitoneum'
        
        plt.figure(figsize=(7,7), dpi=114.1)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        plt.title(f"Predicted: {clspred}\nActual: {actual_label}\nLikelihood: {outputs['cls_pred_rounded']}   Threshold: {threshold:.4f}", fontsize=17)
        
        file_name = f"{png_path[0].split('.')[0]}_gradcam.png"
        plt.savefig(os.path.join(gradcam_dir, file_name), bbox_inches='tight', pad_inches=0.15)
        plt.close()
    print("Grad-CAM visualizations saved successfully!")

############################# EigenCAM #######################################
def create_eigencam(model, DEVICE):
    # Wrap the model if not already wrapped
    model.to(DEVICE)
    model.eval()

    # If model is wrapped in DataParallel, get the underlying model
    if hasattr(model, "module"):
        base_model = model.module
    else:
        base_model = model

    # Target the final norm1 layer of the last encoder layer
    if hasattr(base_model, "rad_dino"):
        target_layer = [base_model.rad_dino.encoder.layer[-1].norm1]
    else:
        raise AttributeError("Model does not have attribute 'rad_dino'")

    print("#########Target Layer of RADDINO for EigenCAM#########")
    print(target_layer)

    # Create and return the EigenCAM instance
    eigen_cam = EigenCAM(
        model=model, 
        target_layers=target_layer, 
        reshape_transform=reshape_transform
    )
    
    return eigen_cam

def generate_eigencam_visualizations_test(model, test_loader, DEVICE, threshold, save_dir):
    """
    Generate and save EigenCAM visualizations for testing.
    """
    eigen_cam = create_eigencam(model, DEVICE)
    
    # Ensure the output directory exists
    eigencam_dir = os.path.join(save_dir, "eigencam")
    if not os.path.exists(eigencam_dir):
        os.makedirs(eigencam_dir, exist_ok=True)

    for i, data in enumerate(test_loader):
        inputs = data['image']
        png_path = data['png_name']

        # get ground truth label from the batch & convert ground-truth value to human-readable string.
        actual_label_val = data['label'][0].item() if torch.is_tensor(data['label'][0]) else data['label'][0]
        actual_label = 'Pneumoperitoneum' if int(actual_label_val) == 1 else 'Non Pneumoperitoneum'

        outputs = process_gradcam_batch(model, inputs, eigen_cam, DEVICE, threshold, rounding_precision=4)
        
        # Determine predicted class label.
        clspred = 'Pneumoperitoneum' if int(outputs['cls_pred_bin'].item()) == 1 else 'Non Pneumoperitoneum'
        
        plt.figure(figsize=(7,7), dpi=114.1)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        plt.title(f"Predicted: {clspred}\nActual: {actual_label}\nLikelihood: {outputs['cls_pred_rounded']}   Threshold: {threshold}", fontsize=17)
        
        file_name = f"{png_path[0].split('.')[0]}_eigencam.png"
        plt.savefig(os.path.join(eigencam_dir, file_name), bbox_inches='tight', pad_inches=0.15)
        plt.close()
    print("EigenCAM visualizations saved successfully!")

############################################################################
######################## Attention Rollout #################################
############################################################################
# def rollout(attentions, discard_ratio, head_fusion):
#     """
#     Computes the Attention Rollout mask. (Identical to previous version)

#     Args:
#         attentions (list or tuple): List/tuple of attention tensors
#                                    (e.g., [batch, heads, seq, seq]) from each layer.
#                                    Should be on CPU or the correct device.
#         discard_ratio (float): Percentage of low-attention connections to discard.
#         head_fusion (str): Method to fuse attention heads ('min', 'max', or 'mean').

#     Returns:
#         np.ndarray: The normalized 2D attention rollout mask.
#     """
#     if not attentions:
#         raise ValueError("Attention list/tuple cannot be empty.")
#     if not (0.0 <= discard_ratio < 1.0):
#         raise ValueError("discard_ratio must be between 0.0 and 1.0 (exclusive of 1.0).")
#     if head_fusion not in ['min', 'max', 'mean']:
#          raise ValueError("head_fusion must be 'min', 'max', or 'mean'.")

#     # Determine device from first attention tensor
#     device = attentions[0].device
#     seq_len = attentions[0].size(-1)
#     result = torch.eye(seq_len, device=device)

#     with torch.no_grad():
#         for i, attention in enumerate(attentions):
#             attention = attention.to(device) # Ensure tensor is on the calculation device

#             # print(f"Layer {i} - Attention shape: {attention.shape}") # Debug

#             # --- Shape check and Head Fusion ---
#             if attention.ndim != 4:
#                  print(f"Warning: Skipping layer {i}. Expected 4D attention tensor, got {attention.ndim}D (Shape: {attention.shape})")
#                  continue
#             if attention.shape[0] != 1:
#                  print(f"Warning: Assuming batch size 1 for rollout, but got batch size {attention.shape[0]} in layer {i}. Using first item.")
#                  attention = attention[0].unsqueeze(0) # Select first item, keep 4D

#             # Squeeze batch dim if B=1
#             if attention.shape[0] == 1:
#                  attention = attention.squeeze(0) # [H, N, N]
#             else: # Should not happen if we selected first item above
#                  print(f"Error: Unexpected batch dimension {attention.shape[0]} after check.")
#                  continue

#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=0) # [N, N]
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=0)[0] # [N, N]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=0)[0] # [N, N]

#             # --- Discard Ratio ---
#             flat = attention_heads_fused.view(-1)
#             num_to_discard = int(flat.size(0) * discard_ratio)
#             if num_to_discard > 0 and flat.numel() > 0:
#                 try:
#                     _, indices = flat.topk(num_to_discard, largest=False)
#                     discard_mask = torch.ones_like(flat)
#                     discard_mask[indices] = 0
#                     attention_heads_fused = attention_heads_fused * discard_mask.view(seq_len, seq_len)
#                 except RuntimeError as e:
#                     print(f"  Error during topk/discard in layer {i}: {e}. Skipping discard.")

#             # --- Add identity and normalize ---
#             I = torch.eye(seq_len, device=attention_heads_fused.device)
#             a = (attention_heads_fused + I) / 2.0
#             a = a / (a.sum(dim=-1, keepdim=True) + 1e-10) # Add epsilon

#             # --- Accumulate ---
#             result = torch.matmul(a, result)

#     # --- Extract final mask ---
#     mask = result[0, 1:] # Class token 0 -> Patch tokens 1:N

#     # --- Reshape into a 2D grid ---
#     num_patches = mask.size(-1)
#     if num_patches <= 0:
#          print("ERROR: Calculated mask has zero or negative size. Cannot reshape.")
#          return np.zeros((1, 1))

#     width = int(math.sqrt(num_patches))
#     if width * width != num_patches:
#         print(f"Warning: Number of patches ({num_patches}) is not a perfect square. Reshaping as {width}x{width}.")
#         if width * width > num_patches: width = int(math.floor(math.sqrt(num_patches)))
#         mask = mask[:width*width]

#     if mask.numel() == 0:
#          print("ERROR: Mask became empty after processing. Cannot reshape.")
#          return np.zeros((1, 1))

#     try:
#         mask = mask.reshape(width, width)
#     except RuntimeError as e:
#          print(f"ERROR during final reshape: {e} (Mask size: {mask.shape}, Target: ({width}, {width}))")
#          return np.zeros((width, width) if width > 0 else (1,1))

#     # --- Normalize and convert to NumPy ---
#     mask = mask.cpu().numpy()
#     max_val = np.max(mask)
#     if max_val > 1e-8: mask = mask / max_val
#     else: mask = np.zeros_like(mask)
#     mask = np.clip(mask, 0, 1)
#     return mask


# # --- New Class: No Hooks ---
# class VITAttentionRolloutFromOutput:
#     """
#     Computes Vanilla Attention Rollout OR visualizes ONLY the last layer's attention map
#     using the 'attentions' attribute from the model's final output object.

#     Args:
#         model (torch.nn.Module): The Vision Transformer model (or wrapper).
#         head_fusion (str): Method for fusing attention heads ('min', 'max', 'mean').
#         discard_ratio (float): Used only if compute_full_rollout=True.
#         compute_full_rollout (bool): If True, computes standard rollout.
#                                      If False, computes only the last layer's fused map.
#     """
#     def __init__(self, model, head_fusion="max", discard_ratio=0.9, compute_full_rollout=False): # Added flag
#         self.model = model
#         self.model.eval()
#         self.head_fusion = head_fusion
#         self.discard_ratio = discard_ratio
#         self.compute_full_rollout = compute_full_rollout # Store the flag
#         if self.compute_full_rollout:
#              print("Initialized VITAttentionRolloutFromOutput (No Hooks - Full Rollout Mode)")
#         else:
#              print("Initialized VITAttentionRolloutFromOutput (No Hooks - Last Layer Only Mode)")


#     def __call__(self, input_tensor):
#         """
#         Generates the visualization mask.

#         Args:
#             input_tensor (torch.Tensor): Input tensor for the model (e.g., [1, 3, 224, 224]).

#         Returns:
#             np.ndarray: The normalized 2D mask (either full rollout or last layer).
#         """
#         input_tensor = input_tensor.to(next(self.model.parameters()).device)
#         attentions = None

#         with torch.no_grad():
#             try:
#                 model_output = self.model(input_tensor, output_attentions=True)
#                 if isinstance(model_output, dict) and 'attentions' in model_output:
#                     attentions = model_output['attentions']
#                     if not (attentions and isinstance(attentions, (list, tuple)) and len(attentions) > 0 and torch.is_tensor(attentions[0])):
#                         print("Warning: model_output['attentions'] is empty or invalid.")
#                         attentions = None
#                 else:
#                     print("Warning: 'attentions' key not found or model output is not a dict.")
#                     attentions = None
#             except Exception as e:
#                 print(f"Error during model forward pass: {e}")
#                 raise

#         if not attentions:
#             print("ERROR: Could not compute visualization because attentions were not found.")
#             # Return a dummy mask
#             num_patches = input_tensor.shape[2] // 14 * input_tensor.shape[3] // 14 # Guess shape
#             width = int(math.sqrt(num_patches))
#             return np.zeros((width, width) if width > 0 else (1,1))

#         # --- Choose computation based on the flag ---
#         if self.compute_full_rollout:
#             # --- Compute Full Rollout ---
#             print("Computing full Attention Rollout...")
#             try:
#                 attentions_cpu = [attn.cpu() for attn in attentions if torch.is_tensor(attn)]
#                 if not attentions_cpu: raise ValueError("No valid tensors in attentions list.")
#                 # Assuming rollout function is defined elsewhere
#                 mask = rollout(attentions_cpu, self.discard_ratio, self.head_fusion)
#                 return mask
#             except Exception as e:
#                  print(f"Error during full rollout calculation: {e}")
#                  num_patches = input_tensor.shape[2] // 14 * input_tensor.shape[3] // 14
#                  width = int(math.sqrt(num_patches))
#                  return np.zeros((width, width) if width > 0 else (1,1))
#         else:
#             # --- Compute Last Layer Attention Map Only ---
#             print("Computing last layer attention map only...")
#             try:
#                 # Get the last attention tensor (keep on original device for now)
#                 last_attn = attentions[-1].to(next(self.model.parameters()).device) # Ensure device consistency

#                 if last_attn.ndim != 4:
#                      raise ValueError(f"Last attention tensor is not 4D (shape: {last_attn.shape})")
#                 if last_attn.shape[0] != 1:
#                      print(f"Warning: Batch size > 1 in last attention map ({last_attn.shape[0]}). Using first item.")
#                      last_attn = last_attn[0].unsqueeze(0)

#                 # Squeeze batch dim
#                 last_attn = last_attn.squeeze(0) # [H, N, N]
#                 seq_len = last_attn.size(-1) # N

#                 # Fuse heads
#                 if self.head_fusion == "mean":
#                     fused_last_attn = last_attn.mean(axis=0) # [N, N]
#                 elif self.head_fusion == "max":
#                     fused_last_attn = last_attn.max(axis=0)[0] # [N, N]
#                 elif self.head_fusion == "min":
#                     fused_last_attn = last_attn.min(axis=0)[0] # [N, N]
#                 else:
#                      raise ValueError(f"Unknown head_fusion type: {self.head_fusion}")

#                 # Extract attention from CLS token (index 0) to patch tokens (1:)
#                 mask = fused_last_attn[0, 1:] # Shape: [N-1]

#                 # Reshape into 2D grid
#                 num_patches = mask.size(-1)
#                 if num_patches <= 0: raise ValueError("Calculated mask has zero size.")
#                 width = int(math.sqrt(num_patches))
#                 if width * width != num_patches:
#                     print(f"Warning: Last layer patches ({num_patches}) not square. Reshaping as {width}x{width}.")
#                     if width * width > num_patches: width = int(math.floor(math.sqrt(num_patches)))
#                     mask = mask[:width*width]
#                 if mask.numel() == 0: raise ValueError("Mask empty after processing.")

#                 mask = mask.reshape(width, width)

#                 # Normalize and convert to NumPy
#                 mask = mask.cpu().numpy()
#                 max_val = np.max(mask)
#                 if max_val > 1e-8: mask = mask / max_val
#                 else: mask = np.zeros_like(mask)
#                 mask = np.clip(mask, 0, 1)
#                 return mask

#             except Exception as e:
#                  print(f"Error during last layer attention processing: {e}")
#                  num_patches = input_tensor.shape[2] // 14 * input_tensor.shape[3] // 14
#                  width = int(math.sqrt(num_patches))
#                  return np.zeros((width, width) if width > 0 else (1,1))


# def process_vanilla_rollout_batch(model, inputs, vanilla_rollout_instance, DEVICE, threshold, rounding_precision=5):
#     """
#     Processes one batch for Vanilla Attention Rollout visualization.
#     (Handles dictionary output from model for predictions).

#     Args:
#         model: The Vision Transformer model (used for getting predictions).
#         inputs (torch.Tensor): Input tensor batch (usually batch size 1).
#         vanilla_rollout_instance (VITAttentionRolloutFromOutput): Instance of the rollout class.
#         DEVICE (str): Device ('cuda' or 'cpu').
#         threshold (float): Threshold for binary classification prediction display.
#         rounding_precision (int): Precision for displaying likelihood.

#     Returns:
#         dict: Dictionary containing processed results for visualization.
#               {'cls_pred_bin', 'cls_pred_rounded', 'inputs_np', 'visualization_v'}
#     """
#     inputs = inputs.to(DEVICE)
#     cls_pred_logits = None # Initialize

#     # Get predictions for display purposes (optional but often useful)
#     with torch.no_grad():
#         # This call uses output_attentions=False by default, but model now always returns dict
#         model_output = model(inputs)

#         # --- Extract classification logits from dictionary ---
#         if isinstance(model_output, dict):
#             if "classification" in model_output and torch.is_tensor(model_output["classification"]):
#                 cls_pred_logits = model_output["classification"]
#             else:
#                 print(f"Warning: 'classification' key missing or not a tensor in model output during prediction step.")
#         elif torch.is_tensor(model_output):
#              # Fallback if model somehow returned only a tensor
#              cls_pred_logits = model_output
#         # else: # Handle other unexpected types if necessary
#         #     print(f"Warning: Unexpected model output type during prediction step: {type(model_output)}")

#         # --- Calculate predictions IF logits were found ---
#         if cls_pred_logits is not None:
#              cls_pred_prob = torch.sigmoid(cls_pred_logits) # Apply sigmoid
#              cls_pred_bin = (cls_pred_prob > threshold).float().item() # Get binary prediction
#              cls_pred_rounded = round(cls_pred_prob.item(), rounding_precision) # Rounded probability
#         else:
#              # Handle case where logits couldn't be extracted
#              print("Warning: Could not extract prediction tensor. Using default values.")
#              cls_pred_bin = 0.0 # Default value
#              cls_pred_rounded = 0.0 # Default value


#     # Prepare the input image for visualization (denormalize if needed)
#     inputs_squeezed = inputs.squeeze(0).cpu() # Remove batch dim, move to CPU
#     # !! Adjust mean/std to match your preprocessing !!
#     mean = torch.tensor([0.5, 0.5, 0.5], device=inputs_squeezed.device).view(3, 1, 1)
#     std = torch.tensor([0.5, 0.5, 0.5], device=inputs_squeezed.device).view(3, 1, 1)
#     inputs_denorm = inputs_squeezed * std + mean
#     inputs_np = np.transpose(inputs_denorm.cpu().numpy(), (1, 2, 0)) # HWC format
#     inputs_np = np.clip(inputs_np, 0, 1) # Ensure values are in [0, 1] range

#     # --- Compute Vanilla Attention Rollout mask ---
#     # The instance call might return None or zeros if it fails internally
#     grayscale_cam = vanilla_rollout_instance(inputs) # Call the vanilla rollout instance

#     # --- Resize CAM and Overlay ---
#     height, width = inputs_np.shape[:2]

#     # Handle potential failure in grayscale_cam generation
#     if grayscale_cam is None or not isinstance(grayscale_cam, np.ndarray) or grayscale_cam.size == 0:
#          print("Warning: grayscale_cam is invalid or empty. Using zeros.")
#          grayscale_cam_resized = np.zeros((height, width), dtype=np.float32)
#     else:
#          # Ensure cam is 2D before resize
#          if grayscale_cam.ndim > 2:
#               print(f"Warning: grayscale_cam has unexpected dims {grayscale_cam.ndim}. Squeezing.")
#               grayscale_cam = np.squeeze(grayscale_cam)
#          if grayscale_cam.ndim != 2:
#               print(f"Error: Cannot resize grayscale_cam with {grayscale_cam.ndim} dimensions.")
#               grayscale_cam_resized = np.zeros((height, width), dtype=np.float32)
#          else:
#               grayscale_cam_resized = cv2.resize(grayscale_cam, (width, height), interpolation=cv2.INTER_LINEAR)


#     # Ensure mask is float32 [0, 1]
#     grayscale_cam_resized = np.float32(grayscale_cam_resized)
#     min_val, max_val = np.min(grayscale_cam_resized), np.max(grayscale_cam_resized)
#     if max_val > min_val + 1e-8: # Add epsilon for stability
#          grayscale_cam_resized = (grayscale_cam_resized - min_val) / (max_val - min_val)
#     else:
#          grayscale_cam_resized = np.zeros_like(grayscale_cam_resized) # Set to zero if range is tiny
#     grayscale_cam_resized = np.clip(grayscale_cam_resized, 0, 1)


#     # Overlay heatmap
#     if inputs_np.ndim == 2:
#         inputs_np_rgb = cv2.cvtColor(np.uint8(inputs_np*255), cv2.COLOR_GRAY2RGB) / 255.0
#     elif inputs_np.shape[2] == 1:
#         inputs_np_rgb = cv2.cvtColor(np.uint8(inputs_np*255), cv2.COLOR_GRAY2RGB) / 255.0
#     else:
#         inputs_np_rgb = inputs_np # Assume it's already RGB

#     visualization = show_cam_on_image(inputs_np_rgb, grayscale_cam_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)

#     return {
#         'cls_pred_bin': cls_pred_bin,
#         'cls_pred_rounded': cls_pred_rounded,
#         'inputs_np': inputs_np_rgb,
#         'visualization_v': visualization # 'v' for vanilla
#     }

# # --- Update the generation function ---
# def generate_vanilla_rollout_visualizations_test(model, test_loader, DEVICE, threshold, save_dir,
#                                                  head_fusion='max', # 'min' or 'max'
#                                                  discard_ratio=0.9):
#     """
#     Generates and saves Vanilla Attention Rollout visualizations for the test set
#     using the NO-HOOK method (VITAttentionRolloutFromOutput).
#     """
#     # --- Instantiate the NEW Vanilla Attention Rollout class ---
#     try:
#         # Use the class that gets attentions from final model output
#         vanilla_rollout_instance = VITAttentionRolloutFromOutput(model,
#                                                                  head_fusion=head_fusion,
#                                                                  compute_full_rollout=False,
#                                                                  discard_ratio=discard_ratio)
#     except NameError:
#         print("ERROR: VITAttentionRolloutFromOutput class not found.")
#         return
#     except Exception as e:
#         print(f"ERROR: Failed to instantiate VITAttentionRolloutFromOutput: {e}")
#         return

#     # Prepare the output directory.
#     vanilla_rollout_dir = os.path.join(save_dir, f"vanilla_rollout_NOHOOK_{head_fusion}_{discard_ratio:.2f}")
#     os.makedirs(vanilla_rollout_dir, exist_ok=True)
#     print(f"Saving Vanilla Rollout (No Hook Method) visualizations to: {vanilla_rollout_dir}")

#     model.eval() # Ensure model is in eval mode
#     total_processed = 0
#     for i, data in enumerate(test_loader):
#         # --- This part remains largely the same ---
#         inputs = data['image']
#         png_paths = data.get('png_name', [f'image_{i*test_loader.batch_size + j}' for j in range(inputs.size(0))])
#         actual_labels_val = data.get('label', [None]*inputs.size(0))

#         if inputs.size(0) != 1:
#              print("Warning: Batch size > 1 detected. Processing only the first item.")
#              # Add loop here if needed

#         input_single = inputs[0].unsqueeze(0)
#         png_path = png_paths[0]
#         actual_label_val = actual_labels_val[0]

#         try:
#             # --- Call the existing process_vanilla_rollout_batch ---
#             # It needs the rollout instance and doesn't care how it was generated
#             # Ensure process_vanilla_rollout_batch uses the updated de-normalization logic
#             outputs = process_vanilla_rollout_batch(model, input_single, vanilla_rollout_instance, DEVICE, threshold)

#             # Check if mask generation failed inside the instance call (returned zeros)
#             if np.sum(outputs['visualization_v']) == 0 : # Crude check if image is all black
#                  print(f"Warning: Visualization for {png_path} seems empty. Rollout might have failed.")
#                  # Optionally skip saving or save a placeholder
#                  # continue # Skip saving this image

#             total_processed += 1

#             # --- Plotting and Saving (remains the same) ---
#             pred_label_str = 'Pneumoperitoneum' if int(outputs['cls_pred_bin']) == 1 else 'Non Pneumoperitoneum'
#             if actual_label_val is not None:
#                  actual_label_str = 'Pneumoperitoneum' if int(actual_label_val) == 1 else 'Non Pneumoperitoneum'
#                  title = f"Pred: {pred_label_str} ({outputs['cls_pred_rounded']:.4f})\nActual: {actual_label_str}\nRollout(NoHook): {head_fusion}, d={discard_ratio:.2f}"
#             else:
#                  title = f"Pred: {pred_label_str} ({outputs['cls_pred_rounded']:.4f})\nRollout(NoHook): {head_fusion}, d={discard_ratio:.2f}"

#             plt.figure(figsize=(7, 7))
#             plt.imshow(outputs['visualization_v']) # Use the key from process_vanilla_rollout_batch
#             plt.axis('off')
#             plt.title(title, fontsize=10)

#             base_name = os.path.splitext(os.path.basename(png_path))[0]
#             file_name = f"{base_name}_vanilla_rollout_NOHOOK_{head_fusion}_{discard_ratio:.2f}.png"
#             save_path = os.path.join(vanilla_rollout_dir, file_name)

#             plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
#             plt.close()

#         except Exception as e:
#             print(f"ERROR processing image {png_path}: {e}")

#         # Optional: Limit number of images processed for testing
#         # if i >= 9: break

#     print(f"Vanilla Attention Rollout (No Hook Method) visualizations saved successfully! ({total_processed} images processed)")
