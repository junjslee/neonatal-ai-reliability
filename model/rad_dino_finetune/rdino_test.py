import datetime
import time
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
from rdino_external_dataset import RADDINO_ExternalLateralDataset
            

def main():
    args = rdino_config.get_args_test()
    
    print("torch.cuda.is_available() == ", torch.cuda.is_available())
    print("torch.cuda.device_count() == ", torch.cuda.device_count())
    print("torch.cuda.current_device() == ", torch.cuda.current_device())
    print("torch.cuda.device(0) == ", torch.cuda.device(0))
    print("torch.cuda.get_device_name(0) == ", torch.cuda.get_device_name(0))
    
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
    if args.use_lora:
        model.load_lora_weights(args.pretrained_weight, DEVICE)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model.load_model_checkpoint(args.pretrained_weight, DEVICE)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = model.to(DEVICE)
    # Create unique run name
    external = "T" if args.external else "F"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    weight_basename = os.path.basename(args.pretrained_weight)
    run_name = f"v{args.version}_{current_time}_SBATCH{args.job_id}_{args.job_array}_external{args.external}_{device_name}_weight__{weight_basename}"
    
    # Create directories for saving results and weights
    save_dir = f"/workspace/jun/nec_lat/rad_dino/external_results/{run_name}"
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
    df['png_path'] = df['png_path'].apply(lambda x: x.replace('/home/brody9512', ''))
    
    test_dataset = RADDINO_ExternalLateralDataset(df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=default_collate, shuffle=False, num_workers=4, worker_init_fn=rdino_utils.seed_worker, pin_memory=True)
    
    y_true, y_prob, results = rdino_utils.test_inference_test(model, ClassificationLoss(classification_weight=1.0).to(DEVICE), test_loader, DEVICE, args.model_threshold)
    
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

    # ─── Efficiency measurements ───
    total_params = sum(p.numel() for p in model.parameters())
    auc_per_million = roc_auc_value / (total_params / 1e6)
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
    
    thr_val = args.model_threshold # this for train is: th[youden]

    y_pred = [1 if prob >= thr_val else 0 for prob in y_prob]
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
    report = classification_report(y_true, y_pred, target_names=target_names)
    with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
        f.write("\nEfficiency Metrics:\n")
        f.write(f"Total Parameters: {total_params/1e6:.2f}M\n")
        f.write(f"Trainable Parameters (LoRA only): {trainable_params/1e6:.3f}M\n")
        f.write(f"AUC per Million Total Params:     {roc_auc_value/(total_params/1e6):.4f}\n")
        f.write(f"AUC per Million Trainable Params: {roc_auc_value/(trainable_params/1e6):.4f}\n")
        f.write(f"Average Inference Latency:         {latency_ms:.2f} ms/image\n")
        f.write(f"Peak GPU Memory Usage:             {peak_mem_mb:.2f} MB\n")
        
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Using Threshold:\n{thr_val}\n') # Youden Index
        f.write(f"\nModel Performance Metrics:\n")
        f.write(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})\n")
        f.write(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})\n")
        f.write(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
        f.write(f"ROC curve (area = {roc_auc_value:.3f}, 95% CI: {ci_lower:.4f}-{ci_upper:.4f})\n")

        # Write FP & FN images.
        f.write("\nFalse Positives:\n")
        for fp in results.get("false_positives", []):
            f.write(fp + "\n")
        f.write("\nFalse Negatives:\n")
        for fn in results.get("false_negatives", []):
            f.write(fn + "\n")
                
        for png_name, m in results.items():
            if png_name in ["false_positives", "false_negatives"]:
                    continue
            f.write(f'\n png_name: {png_name}\n')
            for metric, value in m.items():
                f.write(f'  {metric}: {value}\n')
    
    cm = confusion_matrix(y_true, y_pred)
    rdino_utils.plot_confusion_matrix(cm, target_names, ["Pred Non-PP", "Pred PP"], threshold=args.model_threshold, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    print(f'Classification Report:\n{report}')
    print("ROC curve (area = %0.3f)" % auc(fpr, tpr),'\n')
    if args.external:
        print(f'weight : \n{args.pretrained_weight}\n')
    else:
        raise RuntimeError("Not external mode: This operation is only permitted when running in external mode.")
    print(f'Thresold Value : {thr_val}')
    
    # Generate CAM visualizations.
    # //generate_scorecam_visualizations_test(model, test_loader, DEVICE, thr_val, save_dir)
    generate_vit_reciprocam_visualizations_test(model, test_loader, DEVICE, thr_val, save_dir)
    if args.make_reader_study_cam:
        generate_vit_reciprocam_visualizations_test(model, test_loader, DEVICE, thr_val, save_dir,reader_study=True, text=False)
        generate_vit_reciprocam_visualizations_test(model, test_loader, DEVICE, thr_val, save_dir,reader_study=True, text=True)

if __name__ == "__main__":
    main()