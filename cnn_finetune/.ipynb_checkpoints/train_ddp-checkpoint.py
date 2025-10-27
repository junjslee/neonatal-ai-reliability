import datetime
import os
import json
import hashlib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from monai.metrics import ConfusionMatrixMetric
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score

import config
import utils
import optim
from gradcam import ModelWrapper, generate_gradcam_visualizations_test
from losses import ClassificationLoss
from model import LateralClassificationModel
from dataset.lateral_dataset import LateralDataset
from dataset.lateral_dataset_png import LateralDatasetPNG

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


def train(model, criterion, data_loader, optimizer, device):
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
                return
            if loss_value < 0:
                print("Loss is negative ({}), stopping training.".format(loss_value))
                return
            
            # Backpropagation and optimization step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate running loss.
            running_loss += loss_value * inputs.size(0)
            
    epoch_loss = running_loss / len(data_loader.dataset)
    if dist.get_rank() == 0:
        print('Train: \n Loss: {:.4f} \n'.format(epoch_loss))
    sample_loss = {'epoch_loss': epoch_loss}
    
    return sample_loss
            
def test(model, criterion, data_loader, device):
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
                return
            
            running_loss += loss_value * inputs.size(0)
            classification_prediction = torch.sigmoid(classification_prediction)
    
        all_labels.append(labels.detach().cpu().numpy()) # all_labels.append(labels.cpu().numpy())
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
    # --- Initialize Distributed Process Group ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # --- Original GPU Info and Setup ---
    args = config.get_args_train()
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    if dist.get_rank() == 0:
        print("torch.cuda.is_available() == ", torch.cuda.is_available())
        print("gpu_count == ", gpu_count)
        print("torch.cuda.current_device() == ", torch.cuda.current_device())
        print("torch.cuda.device(0) == ", torch.cuda.device(0))
        print("gpu_names == ", gpu_names)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Overwrite DEVICE with local_rank GPU.
    DEVICE = torch.device('cuda', local_rank)
    utils.my_seed_everywhere(args.seed)

    # Build model.
    aux_params = dict(
        pooling='avg',
        dropout=0.5,
        activation=None,
        classes=1,
    )
    model = LateralClassificationModel(layers=args.layers, aux_params=aux_params)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if args.inference:
        if dist.get_rank() == 0:
            print("Inference mode: Loading pretrained weight from", args.pretrained_weight)
        checkpoint = torch.load(args.pretrained_weight, map_location=DEVICE)
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        optimizer = optim.create_optimizer(args.optim, model, args.lr_startstep)
        if args.lr_type == 'reduce':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args.lr_patience)
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # Create unique run name.
    rank = dist.get_rank()
    inference = "T" if args.inference else "F"
    param_str = (
        f"version{args.version}_inference{inference}_layer{args.layers}_batch{args.batch}_size{args.size}_gpu{args.gpu}"
        f"_layers{args.layers}_epoch{args.epoch}"
    )
    run_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    run_name = f"v{args.version}_{current_time}_SBATCH{args.job_id}_{args.job_array}_inference{inference}_layer{args.layers}_batch{args.batch}_epoch{args.epoch}_{device_name}_{run_hash}"

    if rank == 0:
        save_dir = f"/workspace/jun/nec_lat/cnn_classification/results/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        config_file = os.path.join(save_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2)
        print("Run name:", run_name)
        print("Configuration saved to:", config_file)
        weights_dir = os.path.join(save_dir, "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)
            print(f"Created weights directory: {weights_dir}")
    else:
        save_dir = None  # Only rank 0 writes to disk.
        weights_dir = None
    
    # Read CSV and adjust image paths.
    df = pd.read_csv(args.path)
    df['png_path'] = df['png_path'].apply(lambda x: x.replace('/home/brody9512', ''))
    
    # Split into train, validation, and test sets.
    stratify_col = df['Binary_Label']
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=stratify_col)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['Binary_Label'])
    if rank == 0:
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Create datasets and dataloaders.
    if args.input_format == 'png':
        train_dataset = LateralDatasetPNG(train_df, args, training=True)
        val_dataset = LateralDatasetPNG(val_df, args, training=False)
    else:
        train_dataset = LateralDataset(train_df, args, training=True)
        val_dataset = LateralDataset(val_df, args, training=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
    
    # --- Create DataLoaders ---
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch, 
        collate_fn=default_collate, 
        shuffle=False,  # Shuffling is handled by the DistributedSampler.
        num_workers=4, 
        worker_init_fn=utils.seed_worker,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=default_collate, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=utils.seed_worker,
        pin_memory=True,
        sampler=val_sampler
    )
    
    # Only run the training loop if not in inference mode.
    if not args.inference:
        checkpoint_path = os.path.join(weights_dir, f"{run_name}_best") if rank == 0 else None
        losses = {k: [] for k in ['train_epoch_loss', 'test_epoch_loss']}
        metrics = {k: [] for k in ['AUC', 'F1', 'Accuracy', 'Sensitivity', 'Specificity']}
        lrs = []
        prev_weights = None
        best_loss = float('inf')
    
        for epoch in range(args.epoch):
            # Set epoch for DistributedSampler for proper shuffling.
            train_sampler.set_epoch(epoch)
            if rank == 0:
                print(f"Epoch {epoch+1}/{args.epoch}\n--------------------------------------------------")
            weights = utils.get_weights_for_epoch(epoch, [0, 100, 120, 135, 160, 170, 175], [[5, 5]] * 7)
            if prev_weights is None or not np.array_equal(prev_weights, weights):
                if rank == 0:
                    print(f"Weights for Epoch {epoch + 1}: {weights}")
                prev_weights = weights
            
            train_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
            test_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
    
            train_sample_loss = train(model, train_criterion, train_loader, optimizer, DEVICE)
            test_sample_loss, test_sample_metrics = test(model, test_criterion, val_loader, DEVICE)
    
            if rank == 0:
                for key in losses.keys():
                    if 'train' in key:
                        losses[key].append(train_sample_loss[key.split('train_')[1]])
                    else:
                        losses[key].append(test_sample_loss[key.split('test_')[1]])
        
                for key in metrics.keys():
                    metrics[key].append(test_sample_metrics[key])
                
                if test_sample_loss['epoch_loss'] < best_loss:
                    best_loss = test_sample_loss['epoch_loss']
                    utils.save_checkpoint(model.module, optimizer, checkpoint_path)
                    print('Model saved! \n')
                
                scheduler.step(metrics=test_sample_loss['epoch_loss'])
                lrs.append(optimizer.param_groups[0]["lr"])
    
        if rank == 0:
            print("Training complete!")
    
            # Save training curves.
            plt.plot([i+1 for i in range(len(lrs))], lrs, color='g', label='Learning_Rate')
            plt.savefig(os.path.join(save_dir, "LR.png"))
        
            plt.figure(figsize=(12, 27))
            plt.subplot(211)
            plt.plot(range(args.epoch), losses['train_epoch_loss'], color='darkred', label='Train Total Loss')
            plt.plot(range(args.epoch), losses['test_epoch_loss'], color='darkblue', label='Val Total Loss')
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("Loss", fontsize=11)
            plt.title("Total Losses", fontsize=16)
            plt.legend(loc='upper right')
            plt.subplot(212)
            plt.plot(range(args.epoch), metrics['F1'], color='hotpink', label='F1 Score_(CLS)')
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("Score", fontsize=11)
            plt.title("F1 (Classification)", fontsize=16)
            plt.legend()
            plt.savefig(os.path.join(save_dir, "train_val_loss.png"))
            plt.close()
    
            # After training, load the best model.
            checkpoint = torch.load(os.path.join(weights_dir, f"{run_name}_best.pth"), map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Inference/Evaluation common to both training and inference mode.
    if args.input_format == 'png':
        test_dataset = LateralDatasetPNG(test_df, args, training=False)
    else:
        test_dataset = LateralDataset(test_df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=default_collate, shuffle=False, num_workers=0, worker_init_fn=utils.seed_worker, pin_memory=True)
    y_true, y_prob, average_loss, results = utils.test_inference(model, ClassificationLoss(classification_weight=1.0).to(DEVICE), test_loader, DEVICE)

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
        ci_lower, ci_upper = utils.calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))
        
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
        
        y_pred = [1 if prob >= thr_val else 0 for prob in y_prob]
        
        # Youden threshold metrics
        A_pred = (y_prob_np >= thr_val).astype(int)
        accuracy_A = accuracy_score(y_true_np, A_pred)
        sensitivity_A = recall_score(y_true_np, A_pred)
        conf_matrix_A = confusion_matrix(y_true_np, A_pred)
        specificity_A = conf_matrix_A[0, 0] / (conf_matrix_A[0, 0] + conf_matrix_A[0, 1])
        ci_accuracy_A = utils.calculate_ci(accuracy_A, len(y_true_np))
        ci_sensitivity_A = utils.calculate_ci(sensitivity_A, np.sum(y_true_np == 1))
        ci_specificity_A = utils.calculate_ci(specificity_A, np.sum(y_true_np == 0))
        
        print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
        print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
        print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
        print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc_value, ci_lower, ci_upper))
        
        target_names = ["True Non-PP", "True PP"]
        report_youden = classification_report(y_true, y_pred, target_names=target_names)
    
        
        with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
            # Write GPU details
            f.write("GPU Information:\n")
            f.write(f"Number of GPUs: {gpu_count}\n")
            f.write("GPU Names: " + ", ".join(gpu_names) + "\n\n")
            
            f.write(f'Average Classification Loss: {average_loss}\n\n')
    
            f.write(f"\nModel Performance Metrics:\n")
            f.write(f'YOUDEN THRESHOLD METRICS (threshold = {thr_val:.4f}):\n')
            f.write(f'Classification Report:\n{report_youden}\n')
            f.write(f"Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})\n")
            f.write(f"Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})\n")
            f.write(f"Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n\n")
            
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
    
    
        cm_youden = confusion_matrix(y_true, y_pred)
        utils.plot_confusion_matrix(cm_youden, target_names, ["Pred Non-PP", "Pred PP"], threshold=thr_val, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    
        
        print(f'Youden threshold : {thr_val}')
        print(f'Youden Threshold Classification Report:\n{report_youden}\n')
        print("ROC curve (area = %0.2f)" % auc(fpr, tpr),'\n')
        if args.inference:
            print(f'weight : \n{args.pretrained_weight}\n')
        else:
            print(f'weight : \n{run_name}\n')
    
        
        # Generate GradCAM visualizations.
        modelwrapper = ModelWrapper(model)
        generate_gradcam_visualizations_test(modelwrapper, args.layers, test_loader, DEVICE, thr_val, save_dir)

if __name__ == "__main__":
    main()
