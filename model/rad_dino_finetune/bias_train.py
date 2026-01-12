import datetime
import os
import json
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import monai
from torch.utils.data._utils.collate import default_collate
from monai.metrics import ConfusionMatrixMetric
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score

# Optionally use HuggingFace’s PEFT library if available
try:
    from peft import get_peft_model, LoraConfig, TaskType
    peft_available = True
except ImportError:
    peft_available = False

import rdino_config
import rdino_utils
from rdino_gradcam import generate_vit_reciprocam_visualizations_test
from rdino_losses import ClassificationLoss
import rdino_model
from scheduler import CosineAnnealingWarmUpRestarts
from rfbs import RepresentationFocusedBatchSampler

import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations import Lambda as A_Lambda # Not strictly needed for this change
from monai.data import Dataset # Assuming this is torch.utils.data.Dataset or similar
import pandas as pd # Import pandas for type hint


class RADDINO_LateralDatasetPNG(Dataset):
    def __init__(self, df: pd.DataFrame, args, training=True, modeling=True):
        # *** CHANGE: Check for required columns ***
        required_cols = ['png_path', 'biased_label', 'Orientation']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        self.df = df.reset_index(drop=True)
        self.args = args
        self.training = training
        self.min_side = self.args.size
        self.modeling = modeling

        # Build Albumentations transforms.
        if self.training:
            # Define transforms lists based on args
            if self.args.edge_augmentation:
                base_transforms = [
                    A.RandomResizedCrop(height=self.min_side, width=self.min_side, scale=(0.8, 0.9), ratio=(1.0, 1.0), p=1),
                    A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),
                ]
            else:
                 base_transforms = [
                    A.Resize(self.min_side, self.min_side, p=1),
                    A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),
                ]

            # Add other augmentations conditionally
            other_augments = []
            if not args.original:
                 other_augments.extend([
                    # A.HorizontalFlip(p=args.horizontalflip_percentage),
                    A.Sharpen(alpha=(0.1,0.4),lightness=(0.5,1.0),p=args.sharp_percentage),
                 ])
            if not args.flip:
                other_augments.extend([
                    A.HorizontalFlip(p=args.horizontalflip_percentage),
                ])

            other_augments.extend([
                A.RandomBrightnessContrast(brightness_limit=self.args.rbc_brightness, contrast_limit=self.args.rbc_contrast, p=self.args.rbc_percentage),
                A.RandomGamma(gamma_limit=(self.args.gamma_min, self.args.gamma_max), p=self.args.gamma_percentage) if self.args.gamma_truefalse else None,
                A.GaussNoise(var_limit=(self.args.gaussian_min, self.args.gaussian_max), p=self.args.gaussian_percentage) if self.args.gaussian_truefalse else None
            ])

            additional_transforms = []
            if self.args.elastic_truefalse:
                additional_transforms.append(A.ElasticTransform(alpha=self.args.elastic_alpha, sigma=self.args.elastic_sigma, alpha_affine=self.args.elastic_alpha_affine, p=self.args.elastic_percentage))
            additional_transforms.append(A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5))

            # Combine all transforms for training
            transforms_list = base_transforms + other_augments
            if additional_transforms:
                transforms_list.append(A.OneOf(additional_transforms, p=0.5))

            # Add final steps
            transforms_list.extend([
                A.Lambda(image=self.normalize),
                ToTensorV2(),
            ])

        else: # Validation/Testing transforms
            transforms_list = [
                A.Resize(self.min_side, self.min_side, p=1),
                A.Lambda(image=self.normalize),
                ToTensorV2(),
            ]

        # Remove any None values from the transform list.
        self.transforms = A.Compose([t for t in transforms_list if t is not None])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1) Load label, PNG path, and Orientation.
        row = self.df.iloc[idx]  # <--- CORRECTED LINE
        label = row['biased_label']
        png_path = row['png_path']
        orientation = row['Orientation']

        # 2) Load PNG image using cv2.
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or failed to load: {png_path}")

        # Deterministic flip based on Orientation BEFORE masking
        if self.args.flip:
            if orientation == 'L':
                image = cv2.flip(image, 1)

        # Apply top mask
        if self.modeling:
            H = image.shape[0]
            top_crop = int(0.25 * H)
            bottom_crop = int(0.10 * H)
            
            # Zero out top and bottom
            image[:top_crop, :] = 0
            image[-bottom_crop:, :] = 0
            # height, width = image.shape[:2]
            # mask_height = int(height * 0.25)
            # image[:mask_height, :] = 0  # Set top portion to black (0)

        # 3) Apply Albumentations transforms to the (potentially flipped and masked) image.
        data_transformed = self.transforms(image=image)
        final_img_tensor = data_transformed['image']

        sample = {
            'image': final_img_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'png_name': os.path.basename(png_path)
        }
        return sample

    # ---------------------------
    # Utility Methods
    # ---------------------------
    def normalize(self, image, option=False, **kwargs):
        """Normalize image to [0,1] and optionally scale to [-1,1]."""
        # Convert the image to float32 if it's not already
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Avoid division by zero if image is flat (e.g., all black)
        img_max = image.max()
        img_min = image.min()
        if img_max > img_min: # Check if image is not flat
            image -= img_min
            image /= (img_max - img_min) # Normalize to [0, 1]
        elif img_max == 0: # Handle all black image
             pass # Already 0
        else: # Handle flat non-black image (all pixels same value > 0)
             image[:] = 1.0 # Set all to 1

        if option:
            image = (image - 0.5) / 0.5 # Scale to [-1, 1]

        return image



######################################################################

def log_total_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total Gradient Norm: {total_norm:.4f}")

def train_model(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    running_loss = 0
    
    for i, data in enumerate(data_loader):
        inputs, labels, _ = data['image'].to(device), data['label'].unsqueeze(1).float().to(device), data['png_name']
        
        with torch.set_grad_enabled(True):
            classification_prediction = model(inputs)
            
            # Calculate Loss
            loss, loss_detail = criterion(cls_pred=classification_prediction, cls_gt=labels)
            loss_value = loss.item()

            # Check if loss is finite and non-negative.
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                raise ValueError("Error")
            if loss_value < 0:
                print("Loss is negative ({}), stopping training.".format(loss_value))
                raise ValueError("Error")
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Log gradient norm only on the last batch every 5 epochs
            if epoch % 5 == 0 and i == len(data_loader) - 1:
                print(f"Epoch {epoch} - ", end="")
                log_total_gradient_norm(model)

            # optimization step
            optimizer.step()
            
            # Accumulate running loss.
            running_loss += loss_value * inputs.size(0)
            
    epoch_loss = running_loss / len(data_loader.dataset)
    print('Train: \n Loss: {:.4f} \n'.format(epoch_loss))
    sample_loss = {'epoch_loss': epoch_loss}
    
    return sample_loss
            
def validate_model(model, criterion, data_loader, device):
    model.eval()   
    
    all_labels = []
    all_preds = []    
    running_loss = 0
    confuse_metric = ConfusionMatrixMetric()
    
    for i, data in enumerate(data_loader):
        inputs, labels, _ = data['image'].to(device), data['label'].unsqueeze(1).float().to(device), data['png_name']
        
        with torch.no_grad():
            classification_prediction = model(inputs)
            
            # Calculate Loss
            loss, loss_detail = criterion(cls_pred=classification_prediction, cls_gt=labels)
            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                raise ValueError("Error")
            
            running_loss += loss_value * inputs.size(0)
            classification_prediction = torch.sigmoid(classification_prediction)
    
        all_labels.append(labels.detach().cpu().numpy()) # .labels.cpu().numpy() --> Avoid CPU to GPU transfers or vice-versa
        all_preds.append(classification_prediction.cpu().numpy())
        confuse_metric(y_pred=classification_prediction.round(), y=labels)
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    auc_value = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds.round())
    acc = accuracy_score(all_labels, all_preds.round())
    sen = recall_score(all_labels, all_preds.round())
    spe = precision_score(all_labels, all_preds.round(), zero_division=1)
    confuse_metric.reset()
            
    epoch_loss = running_loss / len(data_loader.dataset)
    print('Validation: \n Loss: {:.4f} \n'.format(epoch_loss))
    sample_loss = {'epoch_loss': epoch_loss}
    sample_metrics = {'AUC': auc_value, 'F1': f1, 'Accuracy': acc, 'Sensitivity': sen, 'Specificity': spe}
    print(' AUC: {:.4f} F1: {:.4f} Accuracy: {:.4f} Sensitivity: {:.4f} Specificity: {:.4f} \n'.format(auc_value, f1, acc, sen, spe))
    
    return sample_loss, sample_metrics

def main():
    args = rdino_config.get_args_train()
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    print("torch.cuda.is_available() == ", torch.cuda.is_available())
    print("gpu_count == ", gpu_count)
    print("torch.cuda.current_device() == ", torch.cuda.current_device())
    print("torch.cuda.device(0) == ", torch.cuda.device(0))
    print("gpu_names == ", gpu_names)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cuda')
    rdino_utils.my_seed_everywhere(args.seed)
    
    # Initialize model with RAD-DINO encoder
    # Set include_segmentation=True to enable the segmentation decoder
    # Initially with frozen encoder
    include_segmentation = args.include_segmentation  # Set to True when you have strong labels

    if args.unfreeze_all:
        freeze = False
    else:
        freeze = True
    
    # Create Model
    model = rdino_model.RADDINO_Model(
        n_classes=1,
        use_lora=args.use_lora,
        r=args.lora_r,
        alpha=args.lora_alpha,
        apply_mlp_lora=args.apply_mlp_lora,
        freeze_encoder=freeze,
        include_segmentation=args.include_segmentation,
        img_dim=(args.size, args.size)
    )
    
    # Multi-GPU support
    # num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    # if num_gpus > 1:
    #     model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # Compute total and trainable parameters:
    # model.parameters() returns an iterator (or list) of all the parameter tensors in the model, whereas sum(p.numel() for p in model.parameters()) computes the total number of scalar values (elements) across all those parameter tensors. In other words, the first gives you the actual parameters (tensors), and the second gives you a count of how many parameters there are in total.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params
    print(f"Trainable parameters: {trainable_params} / {total_params} ({percentage:.2f}%)")
    
    
    # Compute only the LoRA parameters (if available):
    if not args.use_peft:
        if args.use_lora:
            total_lora_params = (sum(p.numel() for module in model.w_a for p in module.parameters()) +
                                 sum(p.numel() for module in model.w_b for p in module.parameters()))
            
            if args.apply_mlp_lora:
                total_lora_params += (
                    sum(p.numel() for (_, w_a_fc1, _) in model.lora_mlp_modules for p in w_a_fc1.parameters()) +
                    sum(p.numel() for (_, _, w_b_fc1) in model.lora_mlp_modules for p in w_b_fc1.parameters())
                )
                
            lora_percentage = 100 * total_lora_params / total_params
            print(f"Total LoRA params: {total_lora_params} --> {total_lora_params} / {total_params} ({lora_percentage:.2f}%)")
            print(f"The rest are classification_head params: {trainable_params - total_lora_params}")
        else:
            print("LoRA is disabled; skipping LoRA parameter count.")
    
    # Check mode: training or inference
    if args.inference:
        print("Inference mode: Loading pretrained weight from", args.pretrained_weight)
        checkpoint = torch.load(args.pretrained_weight, map_location=DEVICE)
        if args.use_lora and not args.use_peft:
            model.load_lora_weights(checkpoint, DEVICE)
        elif args.use_peft:
            model.load_model(checkpoint)
        else:
            model.load_model_checkpoint(checkpoint, DEVICE)
    else:
        if args.lr_type == 'reduce':
            optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.000001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-4, # 1e-4 ;; 1e-2 to 5e-2
            amsgrad=False
            )
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.patience) # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        elif args.lr_type == 'cosinewarm':
            optimizer = torch.optim.AdamW(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.000001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=5e-4, # 1e-4 ;; 1e-2 to 5e-2
                amsgrad=False
            )
            # For example, using T_0=20, T_mult=2, T_up=0 (no warmup) and a gamma factor (to scale eta_max at each restart)
            if not args.cosinewarmrefined:
                scheduler = CosineAnnealingWarmUpRestarts(
                    optimizer,
                    T_0=25,
                    T_mult=1,
                    eta_max=args.lr_startstep,  # initial lr
                    T_up=7, 
                    gamma=0.3  # This will reduce eta_max by 0.5 after each cycle
                )
            else:
                scheduler = CosineAnnealingWarmUpRestarts(
                    optimizer,
                    T_0=args.t0,
                    T_mult=args.tmult,
                    eta_max=args.lr_startstep,
                    T_up=args.t_up,
                    gamma=args.gam
                )
        elif args.lr_type == 'step':
            optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_startstep,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-4, # 1e-4 ;; 1e-2 to 5e-2
            amsgrad=False
            )
            scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5) # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        else:
            optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_startstep,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-4, # 1e-4 ;; 1e-2 to 5e-2
            amsgrad=False
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Create unique run name
    inference = "T" if args.inference else "F"
    apply_mlp_lora = "T" if args.apply_mlp_lora else "F"
    unfreeze_all = "T" if args.unfreeze_all else "F"
    edgeAug = "T" if args.edge_augmentation else "F"
    custombatch = "T" if args.custom_batch_atypical else "F"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    run_name = f"v{args.version}_{current_time}_SBATCH{args.job_id}_{args.job_array}_inference{inference}_batch{args.batch}_epoch{args.epoch}_useLoRA{args.use_lora}_loraR{args.lora_r}_loraAlpha{args.lora_alpha}_loraMLP{apply_mlp_lora}_unfreezeAll{unfreeze_all}_edgeAug{edgeAug}_customBatch{custombatch}_{device_name}"
    
    # Create directories for saving results and weights
    save_dir = f"/workspace/jun/nec_lat/rad_dino/bias_model_results/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(save_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Run name:", run_name)
    print("Configuration saved to:", config_file)
    
    # Read CSV and adjust image paths
    df = pd.read_csv(args.path)
    
    # # Split into train, validation, and test sets
    # stratify_col = df['Binary_Label']
    # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=stratify_col)
    # val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['Binary_Label'])
    # print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    #--------------
    if 'biased_label' not in df.columns:
        raise ValueError("The required column 'biased_label' was not found in the DataFrame for stratification. Cannot proceed.")
    
    print(f"Original dataset size: {len(df)}")
    print(f"Value counts for 'biased_label' in original df:\n{df['biased_label'].value_counts(normalize=True)}")

    # Define the column to stratify on
    stratify_col_series = df['biased_label']

    # First split: 70% train, 30% temporary (for validation and test)
    # Stratify by 'biased_label'
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,  # 30% for temp_df
        random_state=args.seed,
        stratify=stratify_col_series
    )

    # Stratify temp_df again for validation and test based on its 'biased_label' distribution
    # The stratify_col_series for the second split should come from temp_df
    stratify_temp_series = temp_df['biased_label']

    # Second split: From the 30% temp_df, split into validation and test
    # For example, if you want validation to be 10% and test to be 20% of the original:
    # test_size for the second split would be 0.20 / 0.30 = 2/3
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(2/3),  # Makes test_df 2/3 of temp_df (20% of original), val_df 1/3 of temp_df (10% of original)
        random_state=args.seed,
        stratify=stratify_temp_series
    )
    
    # Reset index for cleaner access later, though not strictly necessary for Dataset class if using .iloc
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train samples: {len(train_df)}")
    print(f"Value counts for 'biased_label' in train_df:\n{train_df['biased_label'].value_counts(normalize=True)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Value counts for 'biased_label' in val_df:\n{val_df['biased_label'].value_counts(normalize=True)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Value counts for 'biased_label' in test_df:\n{test_df['biased_label'].value_counts(normalize=True)}")
    print("\nSuccessfully created train_df, val_df, and test_df using stratified split on 'biased_label'.\n")
    
    
    # Create datasets and dataloaders
    print("\nCreating Datasets...")
    train_dataset = RADDINO_LateralDatasetPNG(train_df, args, training=True)
    val_dataset = RADDINO_LateralDatasetPNG(val_df, args, training=False)

    # 1. Calculate pos_weight value (outside the class definition, based on training data)
    class_counts = train_df['biased_label'].value_counts().sort_index()
    count0 = class_counts.get(0, 0)
    count1 = class_counts.get(1, 0)
    if count1 > 0 and count0 > 0: # Avoid division by zero and handle cases with only one class
         pos_weight_value = count0 / count1 + 0.1
    else:
         pos_weight_value = 1.0 # Default if calculation is not possible
    # 2. Create the tensor (make sure it's on the correct device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Or your specific device
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    print(f"Using pos_weight for BCE Loss: {pos_weight_tensor.item():.4f}")

    drop_lastbatch = args.drop_last

    if args.custom_batch_atypical:
        if args.apply_patient_awareness_to_custom_batch:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=RepresentationFocusedBatchSampler(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=drop_lastbatch, debug=False),
                num_workers=4,
                worker_init_fn=rdino_utils.seed_worker,
                pin_memory=True
            )
        elif args.apply_label_and_patient_awareness:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                num_workers=4,
                worker_init_fn=rdino_utils.seed_worker,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=AtypicalInclusiveBatchSampler(train_dataset, batch_size=args.batch, shuffle=True, drop_last=drop_lastbatch, debug=False),
                num_workers=4,
                worker_init_fn=rdino_utils.seed_worker,
                pin_memory=True
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            collate_fn=default_collate,
            shuffle=True,
            num_workers=4,
            worker_init_fn=rdino_utils.seed_worker,
            pin_memory=True
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=default_collate, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=rdino_utils.seed_worker,
        pin_memory=True
    )

    test_dataset = RADDINO_LateralDatasetPNG(test_df, args, training=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        collate_fn=default_collate, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=rdino_utils.seed_worker,
        pin_memory=True
    )

    if train_dataset: # Check if train_dataset exists
        # Get a sample from the train_dataset (e.g., the first image)
        sample_data = train_dataset[0]
        image_tensor = sample_data['image'] # This is the tensor after all transforms
        label = sample_data['label'].item()
        png_name = sample_data['png_name']

        # Convert tensor to a NumPy array suitable for saving/viewing
        # 1. Move to CPU (if it was on GPU, though dataset usually yields CPU tensors)
        image_tensor_cpu = image_tensor.cpu()
        # 2. Convert to NumPy array
        image_numpy = image_tensor_cpu.numpy()

        # 3. Handle normalization: Your normalize function scales to [0, 1]
        #    If it were [-1, 1], you'd do: image_numpy = (image_numpy * 0.5) + 0.5
        #    For [0, 1], we just need to scale to [0, 255]
        image_numpy = (image_numpy * 255).astype(np.uint8)

        # 4. Adjust channel dimension:
        #    Input is grayscale, ToTensorV2 makes it [1, H, W].
        #    For cv2.imwrite or plt.imshow with cmap='gray', we need [H, W].
        if image_numpy.shape[0] == 1: # Grayscale [1, H, W]
            image_numpy_to_save = image_numpy.squeeze(0) # Result is [H, W]
        elif image_numpy.shape[0] == 3: # RGB [3, H, W]
            image_numpy_to_save = np.transpose(image_numpy, (1, 2, 0)) # Result is [H, W, C]
        else:
            image_numpy_to_save = image_numpy # Should not happen for typical image data

        # 5. Save the image using OpenCV
        sample_save_dir = os.path.join(save_dir, "processed_samples")
        os.makedirs(sample_save_dir, exist_ok=True)
        
        # Use a unique name, perhaps including the original png name and label
        save_filename = f"sample_idx0_label{int(label)}_{png_name}"
        save_path_cv2 = os.path.join(sample_save_dir, save_filename)
        
        try:
            cv2.imwrite(save_path_cv2, image_numpy_to_save)
            print(f"[Debug] Saved processed sample to: {save_path_cv2}")
        except Exception as e:
            print(f"[Debug] Error saving image with OpenCV: {e}")

    else:
        print("[Debug] train_dataset not available to save a sample.")
    # --- End Visualization ---
    
    # Only run the training loop if not in inference mode
    if not args.inference:
        weights_dir = os.path.join(save_dir, "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)
            print(f"Created weights directory: {weights_dir}")

        # Instantiate loss functions ONCE before the loop
        # Apply pos_weight ONLY to the training loss
        print("Instantiating loss functions...")
        train_criterion = ClassificationLoss(
            classification_weight=1.0,
            pos_weight_tensor=pos_weight_tensor # Pass the calculated weight here
        ).to(DEVICE)
        # Validation/Test loss should NOT be weighted
        validate_criterion = ClassificationLoss(
            classification_weight=1.0,
            pos_weight_tensor=None # Explicitly None or omit
        ).to(DEVICE)
        # --- END ADDED SECTION ---
        
        # Initialize tracking dictionaries
        losses = {k: [] for k in ['train_epoch_loss', 'test_epoch_loss']}
        metrics = {k: [] for k in ['AUC', 'F1', 'Accuracy', 'Sensitivity', 'Specificity']}
        lrs = []
        best_loss = float('inf')
        best_auc = -float('inf')

        load_path_best_loss = ""
        load_path_best_auc = ""

        wait = 0
        
        # Training loop
        for epoch in range(args.epoch):
            if wait >= args.patience:
                print("Patience reached, triggering Early Stopping")
                break
            
            print(f"--------------------------------------------------\nEpoch {epoch+1}/{args.epoch}\n")
            
            # train_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
            # test_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
            # train_sample_loss = train_model(model, train_criterion, train_loader, optimizer, DEVICE, epoch)
            # test_sample_loss, test_sample_metrics = validate_model(model, test_criterion, val_loader, DEVICE)

            train_sample_loss = train_model(model, train_criterion, train_loader, optimizer, DEVICE, epoch)
            test_sample_loss, test_sample_metrics = validate_model(model, validate_criterion, val_loader, DEVICE) # Use validate_criterion
    
            for key in losses.keys():
                if 'train' in key:
                    losses[key].append(train_sample_loss[key.split('train_')[1]])
                else:
                    losses[key].append(test_sample_loss[key.split('test_')[1]])
    
            for key in metrics.keys():
                metrics[key].append(test_sample_metrics[key])
            
            if test_sample_loss['epoch_loss'] < best_loss:
                best_loss = test_sample_loss['epoch_loss']
                if args.use_lora and not args.use_peft:
                    checkpoint_path = os.path.join(weights_dir, f"{run_name}_lora_best")
                    model.save_lora_weights(checkpoint_path)
                    load_path_best_loss = checkpoint_path
                    print("Saved LoRA weights checkpoint to", checkpoint_path)
                elif args.use_peft:
                    checkpoint_path = os.path.join(weights_dir, f"{run_name}_PEFT_best")
                    model.save_model(checkpoint_path)
                    load_path_best_loss = checkpoint_path
                    print("Saved PEFT weights checkpoint to", checkpoint_path)
                else:
                    checkpoint_path = os.path.join(weights_dir, f"{run_name}_FullTune_best")
                    model.save_model_checkpoint(checkpoint_path)
                    load_path_best_loss = checkpoint_path
                    print("Saved full-finetuning weights checkpoint to", checkpoint_path)
                wait = 0
            else:
                wait += 1

            if args.lr_type == 'reduce':
                scheduler.step(metrics=test_sample_loss['epoch_loss'])
            elif args.lr_type == 'cosinewarm':
                scheduler.step()
            elif args.lr_type == 'step':
                scheduler.step()
            else:
                scheduler.step()
            
            lrs.append(optimizer.param_groups[0]["lr"])
            
            print(f"Epoch {epoch+1} - Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if args.lr_type == 'cosinewarm':
                print(f"Scheduler state: T_cur={scheduler.T_cur}, T_i={scheduler.T_i}")
            
            print("GPU memory:{}GB | GPU reserved memory:{}GB".format(
                "{:<6.2f}".format(torch.cuda.max_memory_allocated(DEVICE) / 2**30),
                "{:<6.2f}".format(torch.cuda.max_memory_reserved(DEVICE) / 2**30)
            ))
            torch.cuda.reset_peak_memory_stats()
    
        print("Training complete!")
        
        # This correctly gets the number of epochs that actually ran before stopping
        num_epochs_completed = len(losses['train_epoch_loss'])
        print(f"Plotting results for {num_epochs_completed} completed epochs.")
    
        # Optional but recommended: Add a check for consistent list lengths
        if not (len(losses['test_epoch_loss']) == num_epochs_completed and
                len(metrics['AUC']) == num_epochs_completed and
                len(lrs) == num_epochs_completed):
            print("WARNING: Mismatch in lengths of collected metrics/losses lists! Check training loop appends.")
            # Decide how to handle: plot up to shortest list, or raise error?
            # Plotting up to num_epochs_completed based on train_loss is reasonable, but check appends if warning appears.
    
        # Create the correct x-axis range based on completed epochs
        epoch_range = range(1, num_epochs_completed + 1)
    
        # --- Save training curves using the correct epoch_range ---
    
        # Learning Rate Plot
        plt.figure(figsize=(10, 6))
        # Slice lrs using num_epochs_completed to ensure matching length
        plt.plot(epoch_range, lrs[:num_epochs_completed], color='g', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "LR.png"))
        plt.close()
    
        # Loss and Metrics Plots
        plt.figure(figsize=(12, 18))
    
        plt.subplot(311)
        # Use epoch_range for x; lists are already the correct length (num_epochs_completed)
        plt.plot(epoch_range, losses['train_epoch_loss'], color='darkred', label='Train Loss')
        plt.plot(epoch_range, losses['test_epoch_loss'], color='darkblue', label='Validation Loss')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title("Training and Validation Loss", fontsize=16)
        plt.legend(loc='upper right')
    
        plt.subplot(312)
        plt.plot(epoch_range, metrics['AUC'], color='green', label='AUC')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("AUC", fontsize=11)
        plt.title("AUC Score", fontsize=16)
        plt.legend()
    
        plt.subplot(313)
        plt.plot(epoch_range, metrics['F1'], color='hotpink', label='F1 Score')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("F1 Score", fontsize=11)
        plt.title("F1 Score", fontsize=16)
        plt.legend()
    
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_metrics.png"))
        plt.close()
        print(f"Saved training plots to {save_dir}")
        
        # After training, load the best model
        if args.use_lora and not args.use_peft:
            checkpoint = torch.load(load_path_best_loss,map_location=DEVICE)
            model.load_lora_weights(checkpoint, DEVICE)
        elif args.use_peft:
            checkpoint = torch.load(load_path_best_loss,map_location=DEVICE)
            model.load_model(checkpoint)
        else:
            checkpoint = torch.load(load_path_best_loss,map_location=DEVICE)
            model.load_model_checkpoint(checkpoint, DEVICE)


    if not args.inference:
        checkpoints = [
            ("best_loss", load_path_best_loss),
            # ("best_auc", load_path_best_auc)
        ]

        validate_criterion = ClassificationLoss(
            classification_weight=1.0,
            pos_weight_tensor=None # Explicitly None or omit
        ).to(DEVICE)

        # Loop over each checkpoint
        for weight_label, checkpoint_path in checkpoints:
            if not checkpoint_path:
                print(f"No checkpoint found for {weight_label}; skipping inference for this weight.")
                continue
        
            # Create a new subdirectory for this inference run
            inference_save_dir = os.path.join(save_dir, f"{weight_label}_inference")
            os.makedirs(inference_save_dir, exist_ok=True)
            print(f"Running inference with {weight_label} weight. Results will be saved in: {inference_save_dir}")
        
            # Load the checkpoint into the model
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if args.use_lora and not args.use_peft:
                model.load_lora_weights(checkpoint, DEVICE)
            elif args.use_peft:
                model.load_model(checkpoint)
            else:
                model.load_model_checkpoint(checkpoint, DEVICE)

            y_val, p_val, _, _ = rdino_utils.test_inference(
                model,
                validate_criterion,
                val_loader,
                DEVICE,
                threshold=0   # skip binning so we only collect y_true & y_prob
            )
            fpr, tpr, th = roc_curve(y_val, p_val)
            youden_idx = np.argmax(tpr - fpr)
            thr_val = th[youden_idx]
            print(f"Youden τ (from validation) = {thr_val:.4f}")
            
            y_true, y_prob, average_loss, results = rdino_utils.test_inference(
               model, ClassificationLoss(classification_weight=1.0).to(DEVICE),
                test_loader, DEVICE, threshold=thr_val
            )

            # Compute ROC and pick Youden threshold
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc_value = auc(fpr, tpr)

            
            # Process and save evaluation results.
            y_true_flat = [item[0] for item in y_true]
            y_prob_flat = [item[0] for item in y_prob]
            y_prob_flat_rounded = [round(num, 5) for num in y_prob_flat]
            print(f'y_true: \n{y_true_flat}\n')
            print(f'y_prob: \n{y_prob_flat_rounded}\n')
            
            with open(os.path.join(inference_save_dir, 'results_y_true_prob.txt'), 'w', encoding='utf-8') as f:
                f.write(f'y_true: \n{y_true_flat}\n')
                f.write(f'y_prob: \n{y_prob_flat_rounded}\n')
            
            y_true_np = np.array(y_true_flat)
            y_prob_np = np.array(y_prob_flat_rounded)
            np.savez(os.path.join(inference_save_dir, "results_y_true_prob.npz"), y_true=y_true_np, y_prob=y_prob_np)
            
            ci_lower, ci_upper = rdino_utils.calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))

            # ─── Efficiency measurements ───
            latency_ms = rdino_utils.measure_latency(model, DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            with torch.no_grad():
                _ = model(torch.randn((1, 3, args.size, args.size)).to(DEVICE))
            peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
            
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f, 95%% CI: %0.2f-%0.2f)" % (roc_auc_value, ci_lower, ci_upper))
            plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel("False Positive Rate", fontsize=14)
            plt.ylabel("True Positive Rate", fontsize=14)
            plt.legend(loc="lower right", fontsize=12)
            plt.savefig(os.path.join(inference_save_dir, "roc_curve.png"))
            plt.close()
            
            
        
            
            # OPTIMIZING SENSITIVITY ↓
            # Choose your desired minimum sensitivity (e.g., 0.9 or 90%)
            desired_sensitivity = 0.9
            sensitivities = {}
            for t in th:
                pred = [1 if p >= t else 0 for p in y_prob]
                sens = recall_score(y_true, pred)
                sensitivities[t] = sens
            valid_thresholds = [t for t, s in sensitivities.items() if s >= desired_sensitivity]
            sensitivity_optimized_threshold = max(valid_thresholds) if valid_thresholds else thr_val
        
            y_pred = [1 if prob >= thr_val else 0 for prob in y_prob]
            y_pred_sensitivity = [1 if prob >= sensitivity_optimized_threshold else 0 for prob in y_prob]
        
            # Youden threshold metrics
            A_pred = (y_prob_np >= thr_val).astype(int)
            accuracy_A = accuracy_score(y_true_np, A_pred)
            sensitivity_A = recall_score(y_true_np, A_pred)
            conf_matrix_A = confusion_matrix(y_true_np, A_pred)
            specificity_A = conf_matrix_A[0, 0] / (conf_matrix_A[0, 0] + conf_matrix_A[0, 1])
            ci_accuracy_A = rdino_utils.calculate_ci(accuracy_A, len(y_true_np))
            ci_sensitivity_A = rdino_utils.calculate_ci(sensitivity_A, np.sum(y_true_np == 1))
            ci_specificity_A = rdino_utils.calculate_ci(specificity_A, np.sum(y_true_np == 0))
        
            # Add sensitivity-optimized threshold metrics
            B_pred = (y_prob_np >= sensitivity_optimized_threshold).astype(int)
            accuracy_B = accuracy_score(y_true_np, B_pred)
            sensitivity_B = recall_score(y_true_np, B_pred)
            conf_matrix_B = confusion_matrix(y_true_np, B_pred)
            specificity_B = conf_matrix_B[0, 0] / (conf_matrix_B[0, 0] + conf_matrix_B[0, 1])
            ci_accuracy_B = rdino_utils.calculate_ci(accuracy_B, len(y_true_np))
            ci_sensitivity_B = rdino_utils.calculate_ci(sensitivity_B, np.sum(y_true_np == 1))
            ci_specificity_B = rdino_utils.calculate_ci(specificity_B, np.sum(y_true_np == 0))
            
            print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
            print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
            print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
            print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc_value, ci_lower, ci_upper))
            
            target_names = ["True Non-PP", "True PP"]
            report_youden = classification_report(y_true, y_pred, target_names=target_names)
            report_sensitivity = classification_report(y_true, y_pred_sensitivity, target_names=target_names)
            
            with open(os.path.join(inference_save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
                # Write GPU details
                f.write("GPU Information:\n")
                f.write(f"Number of GPUs: {gpu_count}\n")
                f.write("GPU Names: " + ", ".join(gpu_names) + "\n\n")

                f.write(f"Total Params: {total_params}\n")
                if args.use_lora:
                    f.write(f"Total LoRA params: {total_lora_params} --> {total_lora_params} / {total_params} ({lora_percentage:.2f}%)\n\n")
                    f.write(f"The rest are classification_head params: {trainable_params - total_lora_params}")

                f.write("\nEfficiency & Parameter Metrics:\n")
                f.write(f"Total Parameters:          {total_params/1e6:.2f}M\n")
                f.write(f"Trainable Parameters (LoRA only): {trainable_params/1e6:.3f}M\n")
                f.write(f"AUC per Million Total Params:     {roc_auc_value/(total_params/1e6):.4f}\n")
                f.write(f"AUC per Million Trainable Params: {roc_auc_value/(trainable_params/1e6):.4f}\n")
                f.write(f"Average Inference Latency:         {latency_ms:.2f} ms/image\n")
                f.write(f"Peak GPU Memory Usage:             {peak_mem_mb:.2f} MB\n")
                
                f.write(f'Average Classification Loss: {average_loss}\n\n')
        
                f.write(f"\nModel Performance Metrics:\n")
                f.write(f'YOUDEN THRESHOLD METRICS (threshold = {thr_val:.4f}):\n')
                f.write(f'Classification Report:\n{report_youden}\n')
                f.write(f"Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})\n")
                f.write(f"Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})\n")
                f.write(f"Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n\n")
                
                f.write(f'SENSITIVITY-OPTIMIZED METRICS (threshold = {sensitivity_optimized_threshold:.4f}):\n')
                f.write(f'Classification Report:\n{report_sensitivity}\n')
                f.write(f"Accuracy: {accuracy_B:.4f}, 95% CI: ({ci_accuracy_B[0]:.4f}, {ci_accuracy_B[1]:.4f})\n")
                f.write(f"Sensitivity: {sensitivity_B:.4f}, 95% CI: ({ci_sensitivity_B[0]:.4f}, {ci_sensitivity_B[1]:.4f})\n")
                f.write(f"Specificity: {specificity_B:.4f}, 95% CI: ({ci_specificity_B[0]:.4f}, {ci_specificity_B[1]:.4f})\n\n")
                
                f.write(f"ROC curve (area = {roc_auc_value:.4f}, 95% CI: {ci_lower:.4f}-{ci_upper:.4f})\n")
        
        
                 # Write FP & FN images.
                f.write("\nFalse Positives:\n")
                for fp in results.get("false_positives", []):
                    f.write(fp + "\n")
                f.write("\nFalse Negatives:\n")
                for fn in results.get("false_negatives", []):
                    f.write(fn + "\n")
                
                # Save any per-image metrics stored in results.
                for png_name, m in results.items():
                    # Skip our FP/FN entries if needed.
                    if png_name in ["false_positives", "false_negatives"]:
                        continue
                    f.write(f'\n png_name: {png_name}\n')
                    for metric, value in m.items():
                        f.write(f'  {metric}: {value}\n')
        
        
            cm_youden = confusion_matrix(y_true, y_pred)
            rdino_utils.plot_confusion_matrix(cm_youden, target_names, ["Pred Non-PP", "Pred PP"], threshold=thr_val, save_path=os.path.join(inference_save_dir, "confusion_matrix.png"))
        
            # Sensitivity-optimized threshold confusion matrix
            cm_sensitivity = confusion_matrix(y_true, y_pred_sensitivity)
            rdino_utils.plot_confusion_matrix(cm_sensitivity, target_names, ["Pred Non-PP", "Pred PP"], threshold=sensitivity_optimized_threshold, save_path=os.path.join(inference_save_dir, "confusion_matrix_sensitivity.png"))
        
            print(f'Youden threshold : {thr_val}')
            print(f"Sensitivity-optimized threshold: {sensitivity_optimized_threshold}")
            print(f'Youden Threshold Classification Report:\n{report_youden}\n')
            print(f'Sensitivity Focused Threshold Classification Report:\n{report_sensitivity}\n')
            print("ROC curve (area = %0.2f)" % auc(fpr, tpr),'\n')
            if args.inference:
                print(f'weight : \n{args.pretrained_weight}\n')
            else:
                print(f'weight : \n{run_name}\n')

            # Generate CAM visualizations.
            print("\nCreating Full Dataset for CAM generation...")
            full_dataset = RADDINO_LateralDatasetPNG(df, args, training=False, modeling=False)
    
            full_loader = torch.utils.data.DataLoader(
                full_dataset,
                batch_size=1,  # Process one image at a time for CAMs
                collate_fn=default_collate, # Assuming this is suitable
                shuffle=False, # No need to shuffle for this purpose
                num_workers=4,  # Or your preferred number of workers
                worker_init_fn=rdino_utils.seed_worker, # Optional
                pin_memory=True
            )
            print(f"Full dataset contains {len(full_dataset)} images for CAM generation.")
            
            generate_vit_reciprocam_visualizations_test(model, full_loader, DEVICE, thr_val, save_dir)


if __name__ == "__main__":
    main()


    