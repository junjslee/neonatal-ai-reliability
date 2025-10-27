import datetime
import os
import json
# import hashlib
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
from custom_batch_sampler import AtypicalInclusiveBatchSampler, PatientAwareAtypicalInclusiveBatchSampler, PatientAwareAtypicalLabelBalancedSampler


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
    print('Validation: \n Loss: {:.4f} \n'.format(epoch_loss))
    sample_loss = {'epoch_loss': epoch_loss}
    sample_metrics = {'AUC': auc_value, 'F1': f1, 'Accuracy': acc, 'Sensitivity': sen, 'Specificity': spe}
    print(' AUC: {:.4f} F1: {:.4f} Accuracy: {:.4f} Sensitivity: {:.4f} Specificity: {:.4f} \n'.format(auc_value, f1, acc, sen, spe))
    
    return sample_loss, sample_metrics

def count_parameters(model):
    # If your model is wrapped in DataParallel, unwrap it.
    model_to_check = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_params = sum(p.numel() for p in model_to_check.parameters())
    trainable_params = sum(p.numel() for p in model_to_check.parameters() if p.requires_grad)
    return total_params, trainable_params
    
def main():
    args = config.get_args_train()

    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    print("torch.cuda.is_available() == ", torch.cuda.is_available())
    print("gpu_count == ", gpu_count)
    print("torch.cuda.current_device() == ", torch.cuda.current_device())
    print("torch.cuda.device(0) == ", torch.cuda.device(0))
    print("gpu_names == ", gpu_names)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device('cuda')
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

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Check mode: training or inference.
    if args.inference:
        print("Inference mode: Loading pretrained weight from", args.pretrained_weight)
        checkpoint = torch.load(args.pretrained_weight, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Set up optimizer and scheduler.
        optimizer = optim.create_optimizer(args.optim, model, args.lr_startstep)
        if args.lr_type == 'reduce':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args.lr_patience)
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # Create unique run name.
    inference = "T" if args.inference else "F"
    # param_str = (
    #     f"version{args.version}_inference{inference}_layer{args.layers}_batch{args.batch}_size{args.size}_gpu{args.gpu}"
    #     f"_layers{args.layers}_epoch{args.epoch}"
    # )
    # run_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    run_name = f"v{args.version}_{current_time}_SBATCH{args.job_id}_{args.job_array}_inference{inference}_layer{args.layers}_batch{args.batch}_epoch{args.epoch}_{device_name}"

    # Create directories for saving results and weights.
    save_dir = f"/workspace/jun/nec_lat/cnn_classification/results/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    config_file = os.path.join(save_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Run name:", run_name)
    print("Configuration saved to:", config_file)
    
    # Read CSV and adjust image paths.
    df = pd.read_csv(args.path)
    df['png_path'] = df['png_path'].apply(lambda x: x.replace('/home/brody9512', ''))
    
    # # Split into train, validation, and test sets.
    # stratify_col = df['Binary_Label']
    # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=stratify_col)
    # val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['Binary_Label'])
    # print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    #------------------------
    # Check if 'data_split' column exists in the DataFrame's columns
    if 'data_split_kfold' not in df.columns:
        raise ValueError("The required column 'data_split' was not found in the DataFrame. Cannot proceed.")
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
    
    # Create datasets and dataloaders.
    if args.input_format == 'png':
        train_dataset = LateralDatasetPNG(train_df, args, training=True)
        val_dataset = LateralDatasetPNG(val_df, args, training=False)
    else:
        raise ValueError("we are only using PNG for training, no DCM file allowed")
        #<--Depreciated-->#
        train_dataset = LateralDataset(train_df, args, training=True)
        val_dataset = LateralDataset(val_df, args, training=False)

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

    if args.custom_batch_atypical:
        if args.apply_patient_awareness_to_custom_batch:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=PatientAwareAtypicalInclusiveBatchSampler(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=drop_lastbatch, debug=False),
                num_workers=4,
                worker_init_fn=utils.seed_worker,
                pin_memory=True
            )
        elif args.apply_label_and_patient_awareness:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=PatientAwareAtypicalLabelBalancedSampler(
                    dataset=train_dataset,
                    batch_size=args.batch,
                    target_positive_ratio=0.5, # Adjust as needed
                    shuffle=True,
                    drop_last=drop_lastbatch,
                    debug=False # Set True to see detailed batch info
                ),
                num_workers=4,
                worker_init_fn=utils.seed_worker,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=AtypicalInclusiveBatchSampler(train_dataset, batch_size=args.batch, shuffle=True, drop_last=drop_lastbatch, debug=False),
                num_workers=4,
                worker_init_fn=utils.seed_worker,
                pin_memory=True
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            collate_fn=default_collate,
            shuffle=True,
            num_workers=4,
            worker_init_fn=utils.seed_worker,
            pin_memory=True
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=default_collate, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=utils.seed_worker,
        pin_memory=True
    )
    
    # Only run the training loop if not in inference mode.
    if not args.inference:
        weights_dir = f"/workspace/jun/nec_lat/cnn_classification/results/{run_name}/weights"
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)
            print(f"Created weights directory: {weights_dir}")
    
        checkpoint_path = os.path.join(weights_dir, f"{run_name}_best")
    
        # --- ADD THIS SECTION ---
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
        
        # Initialize tracking dictionaries.
        losses = {k: [] for k in ['train_epoch_loss', 'test_epoch_loss']}
        metrics = {k: [] for k in ['AUC', 'F1', 'Accuracy', 'Sensitivity', 'Specificity']}
        lrs = []
        prev_weights = None
        best_loss = float('inf')
    
        # Training loop.
        for epoch in range(args.epoch):
            print(f"Epoch {epoch+1}/{args.epoch}\n--------------------------------------------------")
            # Optionally update weights/hyperparameters per epoch.
            weights = utils.get_weights_for_epoch(epoch, [0, 100, 120, 135, 160, 170, 175], [[5, 5]] * 7)
            if prev_weights is None or not np.array_equal(prev_weights, weights):
                print(f"Weights for Epoch {epoch + 1}: {weights}")
                prev_weights = weights
            
            # train_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
            # test_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
            # train_sample_loss = train(model, train_criterion, train_loader, optimizer, DEVICE)
            # test_sample_loss, test_sample_metrics = test(model, test_criterion, val_loader, DEVICE)
            
            train_sample_loss = train(model, train_criterion, train_loader, optimizer, DEVICE)
            test_sample_loss, test_sample_metrics = test(model, validate_criterion, val_loader, DEVICE)
    
            for key in losses.keys():
                if 'train' in key:
                    losses[key].append(train_sample_loss[key.split('train_')[1]])
                else:
                    losses[key].append(test_sample_loss[key.split('test_')[1]])
    
            for key in metrics.keys():
                metrics[key].append(test_sample_metrics[key])
            
            if test_sample_loss['epoch_loss'] < best_loss:
                best_loss = test_sample_loss['epoch_loss']
                utils.save_checkpoint(model, optimizer, checkpoint_path)
                print('Model saved! \n')
            
            scheduler.step(metrics=test_sample_loss['epoch_loss'])
            lrs.append(optimizer.param_groups[0]["lr"])
    
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
    
        # 1) After training, load the best model.
        checkpoint = torch.load(os.path.join(weights_dir, f"{run_name}_best.pth"), map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    validate_criterion = ClassificationLoss(
        classification_weight=1.0,
        pos_weight_tensor=None # Explicitly None or omit
    ).to(DEVICE)
    # 2) **Compute τ on the validation set** (only once!)
    y_val, p_val, _, _ = utils.test_inference(
        model,
        validate_criterion,
        val_loader,      # <-- Use your validation loader
        DEVICE,
        threshold=0   # skip binning so we only collect y_true & y_prob
    )
    fpr, tpr, th = roc_curve(y_val, p_val)
    youden_idx = np.argmax(tpr - fpr)
    thr_val = th[youden_idx]
    print(f"Youden τ (from validation) = {thr_val:.4f}")

    # Inference/Evaluation common to both training and inference mode.
    if args.input_format == 'png':
        test_dataset = LateralDatasetPNG(test_df, args, training=False)
    else:
        test_dataset = LateralDataset(test_df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=default_collate, shuffle=False, num_workers=0, worker_init_fn=utils.seed_worker, pin_memory=True)

    # 3) First pass: get all true‐labels & probabilities (no error‐logging)
    y_true, y_prob, average_loss, results = utils.test_inference(model, ClassificationLoss(classification_weight=1.0).to(DEVICE), test_loader, DEVICE, threshold=thr_val)
    
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
    
    
    ci_lower, ci_upper = utils.calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))

    
    # Determine the expected number of input channels based on model configuration
    expected_channels = 3 if 'mit' in args.layers else 1
    # Use the image size specified in args
    dummy_input_shape = (1, expected_channels, args.size, args.size)
    print(f"Using dummy input shape for latency measurement: {dummy_input_shape}")
    # Pass this correct shape to the measure_latency function
    latency_ms = utils.measure_latency(model, DEVICE, input_shape=dummy_input_shape)
    # Also use it for peak memory measurement
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        _ = model(torch.randn(dummy_input_shape).to(DEVICE))
    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)

    
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.3f, 95%% CI: %0.2f-%0.2f)" % (roc_auc_value, ci_lower, ci_upper))
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
    ci_accuracy_A = utils.calculate_ci(accuracy_A, len(y_true_np))
    ci_sensitivity_A = utils.calculate_ci(sensitivity_A, np.sum(y_true_np == 1))
    ci_specificity_A = utils.calculate_ci(specificity_A, np.sum(y_true_np == 0))

    # Add sensitivity-optimized threshold metrics
    B_pred = (y_prob_np >= sensitivity_optimized_threshold).astype(int)
    accuracy_B = accuracy_score(y_true_np, B_pred)
    sensitivity_B = recall_score(y_true_np, B_pred)
    conf_matrix_B = confusion_matrix(y_true_np, B_pred)
    specificity_B = conf_matrix_B[0, 0] / (conf_matrix_B[0, 0] + conf_matrix_B[0, 1])
    ci_accuracy_B = utils.calculate_ci(accuracy_B, len(y_true_np))
    ci_sensitivity_B = utils.calculate_ci(sensitivity_B, np.sum(y_true_np == 1))
    ci_specificity_B = utils.calculate_ci(specificity_B, np.sum(y_true_np == 0))
    
    print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
    print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
    print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
    print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc_value, ci_lower, ci_upper))
    
    target_names = ["True Non-PP", "True PP"]
    report_youden = classification_report(y_true, y_pred, target_names=target_names)
    report_sensitivity = classification_report(y_true, y_pred_sensitivity, target_names=target_names)

    
    with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
        # Write GPU details
        f.write("GPU Information:\n")
        f.write(f"Number of GPUs: {gpu_count}\n")
        f.write("GPU Names: " + ", ".join(gpu_names) + "\n\n")

        f.write("\nEfficiency & Parameter Metrics:\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n\n")
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
        
        f.write(f"ROC curve (area = {roc_auc_value:.3f}, 95% CI: {ci_lower:.4f}-{ci_upper:.4f})\n")


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

    # Sensitivity-optimized threshold confusion matrix
    cm_sensitivity = confusion_matrix(y_true, y_pred_sensitivity)
    utils.plot_confusion_matrix(cm_sensitivity, target_names, ["Pred Non-PP", "Pred PP"], threshold=sensitivity_optimized_threshold, save_path=os.path.join(save_dir, "confusion_matrix_sensitivity.png"))

    print(f'Youden threshold : {thr_val}')
    print(f"Sensitivity-optimized threshold: {sensitivity_optimized_threshold}")
    print(f'Youden Threshold Classification Report:\n{report_youden}\n')
    print(f'Sensitivity Focused Threshold Classification Report:\n{report_sensitivity}\n')
    print("ROC curve (area = %0.3f)" % auc(fpr, tpr),'\n')
    if args.inference:
        print(f'weight : \n{args.pretrained_weight}\n')
    else:
        print(f'weight : \n{run_name}\n')

    
    # Generate GradCAM visualizations.
    modelwrapper = ModelWrapper(model)
    generate_gradcam_visualizations_test(modelwrapper, args.layers, test_loader, DEVICE, thr_val, save_dir)

if __name__ == "__main__":
    main()
