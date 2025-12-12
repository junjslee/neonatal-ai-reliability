###################$$DEPRECIATED$$##########################
import datetime
import os
import json
import hashlib
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
import segmentation_models_pytorch as smp
# Optionally use HuggingFace’s PEFT library if available
try:
    from peft import get_peft_model, LoraConfig, TaskType
    peft_available = True
except ImportError:
    peft_available = False
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler  

import rdino_config
import rdino_utils
from rdino_gradcam import generate_gradcam_visualizations_test
from rdino_losses import ClassificationLoss
import rdino_model
from rdino_dataset import RADDINO_LateralDatasetPNG


'''DistributedDataParallel Version (Original DataParallel Version is commented out below)'''
def train_model(model, criterion, data_loader, optimizer, device):
    model.train()
    running_loss = 0
    
    for i, data in enumerate(data_loader):
        inputs, labels, _ = data['image'].to(device), data['label'].unsqueeze(1).float().to(device), data['png_name']
        
        with torch.set_grad_enabled(True):
            classification_prediction = model(inputs)
            
            # Calculate Loss
            loss, loss_detail = criterion(cls_pred=classification_prediction, cls_gt=labels)
            loss_value = loss.item()
            
            # Forward pass
            # if model.include_segmentation and "mask" in batch:
            #     masks = batch["mask"].to(device)
            #     outputs = model(images)
            #     cls_loss = criterion["classification"](outputs["classification"], labels)
            #     seg_loss = criterion["segmentation"](outputs["segmentation"], masks)
            #     loss = cls_loss + seg_loss
            # else:
            #     outputs = model(images)
            #     if isinstance(outputs, dict):
            #         outputs = outputs["classification"]
            #     loss = criterion["classification"](outputs, labels)
            
            # Check if loss is finite and non-negative.
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                raise ValueError("Error")
            if loss_value < 0:
                print("Loss is negative ({}), stopping training.".format(loss_value))
                raise ValueError("Error")
            
            # Backpropagation and optimization step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate running loss.
            running_loss += loss_value * inputs.size(0)
            
    epoch_loss = running_loss / len(data_loader.dataset)
    if dist.get_rank() == 0:   # Only rank 0 prints
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
    if dist.get_rank() == 0:
        print('Validation: \n Loss: {:.4f} \n'.format(epoch_loss))
        print(' AUC: {:.4f} F1: {:.4f} Accuracy: {:.4f} Sensitivity: {:.4f} Specificity: {:.4f} \n'.format(auc_value, f1, acc, sen, spe))
    sample_loss = {'epoch_loss': epoch_loss}
    sample_metrics = {'AUC': auc_value, 'F1': f1, 'Accuracy': acc, 'Sensitivity': sen, 'Specificity': spe}
    
    return sample_loss, sample_metrics

def main():   
    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device('cuda', local_rank)
    
    args = rdino_config.get_args_train()
    rdino_utils.my_seed_everywhere(args.seed)
    
    # Initialize model with RAD-DINO encoder
    include_segmentation = args.include_segmentation  # Set to True when you have strong labels
    
    # Create Model
    if args.use_peft:
        model = rdino_model.RADDINOPeft_Model(
            n_classes=1, # train_dataset.num_classes
            use_peft=True,
            r=args.lora_r,
            include_segmentation=args.include_segmentation,
            img_dim=(args.size, args.size)
        )
    else:
        model = rdino_model.RADDINO_Model(
            n_classes=1,
            use_lora=args.use_lora,
            r=args.lora_r,
            freeze_encoder=True,
            include_segmentation=args.include_segmentation,
            img_dim=(args.size, args.size)
        )
    
    model = model.to(DEVICE)

    # Compute total and trainable parameters:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params
    print(f"Trainable parameters: {trainable_params} / {total_params} ({percentage:.2f}%)")
    # Optionally, compute only the LoRA parameters (if available):
    lora_params = (
        sum(p.numel() for module in model.w_a for p in module.parameters()) +
        sum(p.numel() for module in model.w_b for p in module.parameters())
    )
    print(f"Total LoRA parameters: {lora_params}")
        
        
    # NEW: Wrap model in DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Check mode: training or inference
    if args.inference:
        if dist.get_rank() == 0:
            print("Inference mode: Loading pretrained weight from", args.pretrained_weight)
        checkpoint = torch.load(args.pretrained_weight, map_location=DEVICE)
        model.module.load_lora_weights(checkpoint)
    else:
        # Set up optimizer and scheduler
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr_startstep,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=5e-4,
                amsgrad=False
            )
        elif args.optim == 'adamw':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr_startstep,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=5e-4,
                amsgrad=False
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optim}")
        if args.lr_type == 'reduce':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args.lr_patience) # factor = 0.5, verbose = True
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Rank
    rank = dist.get_rank()
    
    # Create unique run name
    inference = "T" if args.inference else "F"
    param_str = (
        f"version{args.version}_inference{inference}_rad-dino_batch{args.batch}_size{args.size}_gpu{args.gpu}"
        f"_epoch{args.epoch}_seg{include_segmentation}"
    )
    run_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{current_time}_ddp_v{args.version}_inference{inference}_rad-dino_batch{args.batch}_epoch{args.epoch}_{run_hash}"
    
    # Create directories for saving results and weights
    if rank == 0:
        save_dir = f"/workspace/jun/nec_lat/foundation_model/results/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        config_file = os.path.join(save_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2)
        print("Run name:", run_name)
        print("Configuration saved to:", config_file)
        weights_dir = os.path.join(save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        print(f"Created weights directory: {weights_dir}")
    else:
        save_dir = None  # Only rank 0 writes to disk
        weights_dir = None # Only rank 0 writes to disk
    
    # Read CSV and adjust image paths
    df = pd.read_csv(args.path)
    df['png_path'] = df['png_path'].apply(lambda x: x.replace('/home/brody9512', ''))
    stratify_col = df['Binary_Label']
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=stratify_col)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['Binary_Label'])
    if rank == 0:
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    train_dataset = RADDINO_LateralDatasetPNG(train_df, args, training=True)
    val_dataset = RADDINO_LateralDatasetPNG(val_df, args, training=False)
    
    # Create DistributedSampler for train and validation sets.
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch, 
        collate_fn=default_collate, 
        shuffle=False,  # Do not use shuffle when using sampler.
        num_workers=4, 
        sampler=train_sampler,
        worker_init_fn=rdino_utils.seed_worker,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=default_collate, 
        shuffle=False, 
        num_workers=4, 
        sampler=val_sampler,
        worker_init_fn=rdino_utils.seed_worker,
        pin_memory=True
    )
    
    # Only run the training loop if not in inference mode
    if not args.inference:
        # weights_dir = os.path.join(save_dir, "weights")
        # if not os.path.exists(weights_dir):
        #     os.makedirs(weights_dir, exist_ok=True)
        #     print(f"Created weights directory: {weights_dir}")
        
        # Initialize tracking dictionaries
        losses = {k: [] for k in ['train_epoch_loss', 'test_epoch_loss']}
        metrics = {k: [] for k in ['AUC', 'F1', 'Accuracy', 'Sensitivity', 'Specificity']}
        lrs = []
        best_loss = float('inf')

        load_path = ""

        # Training loop
        for epoch in range(args.epoch):
            # Set epoch for DistributedSampler for proper shuffling.
            train_sampler.set_epoch(epoch)
            print(f"Rank {rank} - Epoch {epoch+1}/{args.epoch}\n--------------------------------------------------")
            train_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
            test_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
    
            train_sample_loss = train_model(model, train_criterion, train_loader, optimizer, DEVICE)
            test_sample_loss, test_sample_metrics = validate_model(model, test_criterion, val_loader, DEVICE)
    
            if dist.get_rank() == 0:
                # Original aggregation loop for metrics
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
                        checkpoint_path = os.path.join(save_dir, f"{run_name}_lora_best_epoch{epoch+1}.pt")
                        model.module.save_lora_weights(checkpoint_path) # model.save_lora_weights
                        load_path = checkpoint_path
                    elif args.use_peft:
                        checkpoint_path = os.path.join(save_dir, f"{run_name}_PEFT_best_epoch{epoch+1}")
                        model.module.save_model(checkpoint_path) # model.save_model
                        load_path = checkpoint_path
                    print("Saved LoRA weights checkpoint to", checkpoint_path)
                
                scheduler.step(metrics=test_sample_loss['epoch_loss'])
                lrs.append(optimizer.param_groups[0]["lr"])
                lr_current = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1} - LR: {lr_current}")
                print("GPU memory:{}GB | GPU reserved memory:{}GB".format(
                    "{:<6.2f}".format(torch.cuda.max_memory_allocated(DEVICE) / 2**30),
                    "{:<6.2f}".format(torch.cuda.max_memory_reserved(DEVICE) / 2**30)
                ))
                torch.cuda.reset_peak_memory_stats()
    
        if dist.get_rank() == 0:
            print("Training complete!")
        
            # Save training curves
            plt.figure(figsize=(10, 6))
            plt.plot([i+1 for i in range(len(lrs))], lrs, color='g', label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.savefig(os.path.join(save_dir, "LR.png"))
            plt.close()
        
            plt.figure(figsize=(12, 18))
            plt.subplot(311)
            plt.plot(range(1, args.epoch + 1), losses['train_epoch_loss'], color='darkred', label='Train Loss')
            plt.plot(range(1, args.epoch + 1), losses['test_epoch_loss'], color='darkblue', label='Validation Loss')
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("Loss", fontsize=11)
            plt.title("Training and Validation Loss", fontsize=16)
            plt.legend(loc='upper right')
        
            plt.subplot(312)
            plt.plot(range(1, args.epoch + 1), metrics['AUC'], color='green', label='AUC')
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("AUC", fontsize=11)
            plt.title("AUC Score", fontsize=16)
            plt.legend()
        
            plt.subplot(313)
            plt.plot(range(1, args.epoch + 1), metrics['F1'], color='hotpink', label='F1 Score')
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("F1 Score", fontsize=11)
            plt.title("F1 Score", fontsize=16)
            plt.legend()
        
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "training_metrics.png"))
            plt.close()
        
        # After training, load the best model (all processes should load the checkpoint, not just rank==0 ones)
        if args.use_lora and not args.use_peft:
            checkpoint = torch.load(load_path,map_location=DEVICE)
            model.module.load_lora_weights(checkpoint)
        elif args.use_peft:
            checkpoint = torch.load(load_path,map_location=DEVICE)
            model.module.load_model(checkpoint)
    
    test_dataset = RADDINO_LateralDatasetPNG(test_df, args, training=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        collate_fn=default_collate, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=rdino_utils.seed_worker, 
        pin_memory=True)
    y_true, y_prob, average_loss, results = rdino_utils.test_inference(model, ClassificationLoss(classification_weight=1.0).to(DEVICE), test_loader, DEVICE, threshold=args.model_threshold)

    if rank == 0:
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
        
        fpr, tpr, th = roc_curve(y_true, y_prob)
        roc_auc_value = auc(fpr, tpr)
        youden = np.argmax(tpr - fpr)
        ci_lower, ci_upper = rdino_utils.calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))
        
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
        
        thr_val = th[youden]
        y_pred_1 = [1 if prob >= thr_val else 0 for prob in y_prob]
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
        
        target_names = ["True Non-PP", "True PP"]
        report = classification_report(y_true, y_pred_1, target_names=target_names)
        with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Average Classification Loss: {average_loss}\n')
            f.write(f'Classification Report:\n{report}\n')
            f.write(f'Youden index:\n{thr_val}\n')
            f.write(f"\nModel Performance Metrics:\n")
            f.write(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})\n")
            f.write(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})\n")
            f.write(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
            f.write(f"ROC curve (area = {roc_auc_value:.2f}, 95% CI: {ci_lower:.4f}-{ci_upper:.4f})\n")
            
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
        
        cm = confusion_matrix(y_true, y_pred_1)
        rdino_utils.plot_confusion_matrix(cm, target_names, ["Pred Non-PP", "Pred PP"], threshold=args.model_threshold, save_path=os.path.join(save_dir, "confusion_matrix.png"))
        print(f'Classification Report:\n{report}')
        print("ROC curve (area = %0.2f)" % auc(fpr, tpr),'\n')
        if args.inference:
            print(f'weight : \n{args.pretrained_weight}\n')
        else:
            print(f'weight : \n{run_name}\n')
        print(f'Thresold Value : {thr_val}')

    
        # Generate GradCAM visualizations.
        generate_gradcam_visualizations_test(model, test_loader, DEVICE, args.model_threshold, save_dir)

if __name__ == "__main__":
    main()

