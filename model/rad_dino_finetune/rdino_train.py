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
from rdino_dataset import RADDINO_LateralDatasetPNG
from scheduler import CosineAnnealingWarmUpRestarts
from rfbs import RepresentationFocusedBatchSampler


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
    if args.use_peft:
        model = rdino_model.RADDINOPeft_Model(
            n_classes=1, # train_dataset.num_classes
            use_peft=True,
            r=args.lora_r,
            apply_mlp_lora=args.apply_mlp_lora,
            include_segmentation=args.include_segmentation,
            img_dim=(args.size, args.size)
        )
        apply_mlp_lora=args.apply_mlp_lora
    else:
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
                    T_mult=2,
                    eta_max=args.lr_startstep,  # initial lr
                    T_up=5, 
                    gamma=0.5  # This will reduce eta_max by 0.5 after each cycle
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
    save_dir = f"/workspace/jun/nec_lat/rad_dino/results/{run_name}"
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
    # Check if 'data_split_kfold' column exists in the DataFrame's columns
    if 'data_split_kfold' not in df.columns:
        raise ValueError("The required column 'data_split_kfold' was not found in the DataFrame. Cannot proceed.")
    else:
        print("Found 'data_split_kfold' column. Proceeding with splitting...") # Optional: Confirmation message
    
        # Using .copy() is recommended to avoid potential SettingWithCopyWarning later
        train_df = df[df['data_split_kfold'] == 'train'].copy()
        val_df = df[df['data_split_kfold'] == 'valid'].copy()
        test_df = df[df['data_split_kfold'] == 'test'].copy()

        print(f"Train samples (loaded based on 'data_split_kfold' column): {len(train_df)}")
        print(f"Validation samples (loaded based on 'data_split_kfold' column): {len(val_df)}")
        print(f"Test samples (loaded based on 'data_split_kfold' column): {len(test_df)}")
    
        # Optional: Verify that all rows have been assigned
        total_split_rows = len(train_df) + len(val_df) + len(test_df)
        if total_split_rows != len(df):
            print(f"\nWarning: {len(df) - total_split_rows} rows in the original DataFrame were not assigned to train, valid, or test.")
            print("This might happen if the 'data_split_kfold' column contains values other than 'train', 'valid', 'test', or NaN/missing values.")
            # Debug
            # print("Unique values found in 'data_split_kfold':", df['data_split_kfold'].unique())
        else:
            print("\nAll rows successfully assigned based on 'data_split_kfold' column.")
    print("\nSuccessfully created train_df, val_df, and test_df.\n")
    
    
    # Create datasets and dataloaders
    print("\nCreating Datasets...")
    train_dataset = RADDINO_LateralDatasetPNG(train_df, args, training=True)
    val_dataset = RADDINO_LateralDatasetPNG(val_df, args, training=False)

    # 1. Calculate pos_weight value (outside the class definition, based on training data)
    class_counts = train_df['Binary_Label'].value_counts().sort_index()
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

    # if args.custom_batch_atypical:
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=RepresentationFocusedBatchSampler(
            dataset=train_dataset,
            batch_size=args.batch,
            target_positive_ratio=0.5, # Adjust as needed
            shuffle=True,
            drop_last=drop_lastbatch,
            debug=False # Set True to see detailed batch info
        ),
        num_workers=4,
        worker_init_fn=rdino_utils.seed_worker,
        pin_memory=True
    )
    # else:
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch,
    #         collate_fn=default_collate,
    #         shuffle=True,
    #         num_workers=4,
    #         worker_init_fn=rdino_utils.seed_worker,
    #         pin_memory=True
    #     )
    
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
    else:
        validate_criterion = ClassificationLoss(
                classification_weight=1.0,
                pos_weight_tensor=None # Explicitly None or omit
            ).to(DEVICE)
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
        
        with open(os.path.join(save_dir, 'results_y_true_prob.txt'), 'w', encoding='utf-8') as f:
            f.write(f'y_true: \n{y_true_flat}\n')
            f.write(f'y_prob: \n{y_prob_flat_rounded}\n')
        
        y_true_np = np.array(y_true_flat)
        y_prob_np = np.array(y_prob_flat_rounded)
        np.savez(os.path.join(save_dir, "results_y_true_prob.npz"), y_true=y_true_np, y_prob=y_prob_np)
        
        
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
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
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

        print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
        print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
        print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
        print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc_value, ci_lower, ci_upper))
    
        # Add sensitivity-optimized threshold metrics
        B_pred = (y_prob_np >= sensitivity_optimized_threshold).astype(int)
        accuracy_B = accuracy_score(y_true_np, B_pred)
        sensitivity_B = recall_score(y_true_np, B_pred)
        conf_matrix_B = confusion_matrix(y_true_np, B_pred)
        specificity_B = conf_matrix_B[0, 0] / (conf_matrix_B[0, 0] + conf_matrix_B[0, 1])
        ci_accuracy_B = rdino_utils.calculate_ci(accuracy_B, len(y_true_np))
        ci_sensitivity_B = rdino_utils.calculate_ci(sensitivity_B, np.sum(y_true_np == 1))
        ci_specificity_B = rdino_utils.calculate_ci(specificity_B, np.sum(y_true_np == 0))
        
        target_names = ["True Non-PP", "True PP"]
        report_youden = classification_report(y_true, y_pred, target_names=target_names)
        report_sensitivity = classification_report(y_true, y_pred_sensitivity, target_names=target_names)
        
        with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
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
        rdino_utils.plot_confusion_matrix(cm_youden, target_names, ["Pred Non-PP", "Pred PP"], threshold=thr_val, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    
        # Sensitivity-optimized threshold confusion matrix
        cm_sensitivity = confusion_matrix(y_true, y_pred_sensitivity)
        rdino_utils.plot_confusion_matrix(cm_sensitivity, target_names, ["Pred Non-PP", "Pred PP"], threshold=sensitivity_optimized_threshold, save_path=os.path.join(save_dir, "confusion_matrix_sensitivity.png"))
    
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
        # generate_scorecam_visualizations_test(model, test_loader, DEVICE, thr_val, save_dir)
        generate_vit_reciprocam_visualizations_test(model, test_loader, DEVICE, thr_val, save_dir)


if __name__ == "__main__":
    main()


    