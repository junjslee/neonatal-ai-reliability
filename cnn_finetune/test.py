### RUN ###
import datetime
import os
import json
# import hashlib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, classification_report, confusion_matrix

import config
import utils
from gradcam import ModelWrapper, generate_gradcam_visualizations_test
from losses import ClassificationLoss
from model import LateralClassificationModel
from dataset.external_lateral_dataset import ExternalLateralDataset


def main():
    args = config.get_args_test()

    print("torch.cuda.is_available() == ", torch.cuda.is_available())
    print("torch.cuda.device_count() == ", torch.cuda.device_count())
    print("torch.cuda.current_device() == ", torch.cuda.current_device())
    print("torch.cuda.device(0) == ", torch.cuda.device(0))
    print("torch.cuda.get_device_name(0) == ", torch.cuda.get_device_name(0))
    
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

    cnn_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Multi-GPU support
    # num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    if args.external:
        print("External mode: Loading pretrained weight from", args.pretrained_weight)
        checkpoint = torch.load(args.pretrained_weight, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise RuntimeError("Not external mode: This operation is only permitted when running in external mode.")
    
    # Prepare output directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    # pretrained_weight path is too long, so get basename only
    pretrained_basename = os.path.basename(args.pretrained_weight).replace(".pth","")
    run_name = f"v{args.version}_{current_time}_SBATCH{args.job_id}_{args.job_array}_external_weight{pretrained_basename}_{device_name}"
    save_dir = f"/workspace/jun/nec_lat/cnn_classification/external_results/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    config_file = os.path.join(save_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Run name:", run_name)
    print("Configuration saved to:", config_file)
    
    # Read CSV and adjust image paths
    df = pd.read_csv(args.path)
    df['png_path'] = df['png_path'].apply(lambda x: x.replace('/home/brody9512', ''))
    
    # For external testing, we use the whole CSV.
    test_dataset = ExternalLateralDataset(df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=default_collate, shuffle=False, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)

    y_true, y_prob, results = utils.test_inference_test(model, ClassificationLoss(classification_weight=1.0).to(DEVICE), test_loader, DEVICE, args.model_threshold)

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

    # ─── Efficiency measurements ───
    total_params = sum(p.numel() for p in model.parameters())
    auc_per_million = roc_auc_value / (total_params / 1e6)
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
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f, 95%% CI: %0.2f-%0.2f)" % (roc_auc_value, ci_lower, ci_upper))
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()
    
    # thr_val = args.model_threshold # this for train is: th[youden]
    
    y_pred = [1 if prob >= args.model_threshold else 0 for prob in y_prob]
    A_pred = (y_prob_np >= args.model_threshold).astype(int)
    
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
    report = classification_report(y_true, y_pred, target_names=target_names)
    with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as f:
        f.write("\nEfficiency Metrics:\n")
        f.write(f"Total Parameters: {total_params/1e6:.2f}M\n")
        f.write(f"Trainable Parameters (CNN): {cnn_trainable_params/1e6:.3f}M\n")
        f.write(f"AUC per Million Total Params:     {roc_auc_value/(total_params/1e6):.4f}\n")
        f.write(f"AUC per Million Trainable Params: {roc_auc_value/(cnn_trainable_params/1e6):.4f}\n")
        f.write(f"Average Inference Latency:         {latency_ms:.2f} ms/image\n")
        f.write(f"Peak GPU Memory Usage:             {peak_mem_mb:.2f} MB\n")
        
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Using Threshold:\n{args.model_threshold}\n')
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
        
        # Save any per-image metrics stored in results.
        for png_name, m in results.items():
            # Skip our FP/FN entries if needed.
            if png_name in ["false_positives", "false_negatives"]:
                continue
            f.write(f'\n png_name: {png_name}\n')
            for metric, value in m.items():
                f.write(f'  {metric}: {value}\n')
    
    cm = confusion_matrix(y_true, y_pred)
    utils.plot_confusion_matrix(cm, target_names, ["Pred Non-PP", "Pred PP"], threshold=args.model_threshold, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    print(f'Classification Report:\n{report}')
    print("ROC curve (area = %0.3f)" % auc(fpr, tpr),'\n')
    if args.external:
        print(f'weight : \n{args.pretrained_weight}\n')
    else:
        raise RuntimeError("Not external mode: This operation is only permitted when running in external mode.")
    print(f'Thresold Value : {args.model_threshold}')

    
    # Generate GradCAM visualizations.
    modelwrapper = ModelWrapper(model)
    generate_gradcam_visualizations_test(modelwrapper, args.layers, test_loader, DEVICE, args.model_threshold, save_dir)
    

if __name__ == "__main__":
    main()
