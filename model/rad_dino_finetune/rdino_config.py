import argparse
import os

def get_args_train():
    parser = argparse.ArgumentParser(description="Lateral Image Classification Training Arguments")
    # Basic I/O and Paths
    parser.add_argument('--path', type=str, default='workspace/yeonsu/0.Projects/Pneumoperitoneum/data/internal_data/Internal6.csv', help='Path to CSV with lateral image data')
    parser.add_argument("--gpu", type=str, default='1', help="GPU id(s) to use for non-DDP runs (ignored when using DDP).")
    parser.add_argument('--version', type=int, default=0)

    # Custom Args
    parser.add_argument('--inference', action='store_true', help='Run inference instead of training')
    parser.add_argument('--pretrained_weight', type=str, default='workspace/jun/nec_lat/rad_dino/results/20250330_140909_v2_inferenceF_batch6_epoch300_useLoRATrue_loraR12_loraAlpha24_usePEFTFalse_fc8b0d3b/weights/20250330_140909_v2_inferenceF_batch6_epoch300_useLoRATrue_loraR12_loraAlpha24_usePEFTFalse_fc8b0d3b_lora_best_epoch20', help='Path to pretrained model weights for inference')
    parser.add_argument('--help_info', type=str, default='N/A', help='Basic information to understand the version and the model instance')
    parser.add_argument('--include_segmentation', action='store_true', help='Include segmentation or no')
    parser.add_argument('--unfreeze_all', action='store_true', help='Unfreeze and fully fine tune RAD-DINO') 
    parser.add_argument("--use_peft", action="store_true", help="Use HuggingFace PEFT for LoRA (requires peft library)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation")
    parser.add_argument("--lora_r", type=int, default=12, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=24, help="LoRA rank parameter")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank. This is passed automatically by distributed launcher.") # necessary
    parser.add_argument('--apply_mlp_lora', action='store_true', help='Apply LoRA to MLP layers in addition to query and value')
    parser.add_argument("--edge_augmentation", action='store_true', help="Edge Augmentation")
    parser.add_argument("--custom_batch_atypical", action='store_true', help="custom_batch_atypical")
    parser.add_argument("--apply_patient_awareness_to_custom_batch", action='store_true', help="patient_aware")
    parser.add_argument("--apply_label_and_patient_awareness", action='store_true', help="label_aware")
    # parser.add_argument('--discard_ratio', type=int, default=0.0) # experiment with discard_ratio values (e.g., 0.0, 0.5, 0.9, 0.95, 0.99)
    # parser.add_argument('--head_fusion', type=str, default='min', choices=['min','max', 'mean'])
    parser.add_argument("--original", action='store_true')
    parser.add_argument("--drop_last", action='store_true')
    parser.add_argument("--flip", action='store_true')

    # Model & Training
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=6, help='Train batch size.')
    parser.add_argument('--size', type=int, default=518, help='Target image size.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr_type', type=str, default='consinewarm', choices=['step', 'reduce', 'cosinewarm'])
    parser.add_argument('--lr_startstep', type=float, default=0.0001) # 0.00005, 0.000005
    parser.add_argument('--patience', type=int, default=15) # 1 epoch, no change in loss --> patience + 1

    # Scheduler hyperparams
    parser.add_argument('--cosinewarmrefined', action='store_true')
    parser.add_argument('--t0', type=int, default=40)
    parser.add_argument('--t_up', type=int, default=5)
    parser.add_argument('--tmult', type=int, default=1)
    parser.add_argument('--gam', type=float, default=0.5)
    

    # Data augmentation parameters
    # parser.add_argument('--clahe_cliplimit', type=float, default=2.0)
    # parser.add_argument('--clahe_limit', type=int, default=8)    
    # parser.add_argument('--clip_min', type=float, default=0.5)
    # parser.add_argument('--clip_max', type=float, default=98.5)
    parser.add_argument('--rotate_angle', type=float, default=30)
    parser.add_argument('--rotate_percentage', type=float, default=0.8)
    parser.add_argument('--rbc_brightness', type=float, default=0.05)
    parser.add_argument('--rbc_contrast', type=float, default=0.2)
    parser.add_argument('--rbc_percentage', type=float, default=0.5)
    parser.add_argument('--elastic_truefalse', action='store_true')
    parser.add_argument('--elastic_alpha', type=float, default=1.5)
    parser.add_argument('--elastic_sigma', type=float, default=30)
    parser.add_argument('--elastic_percentage', type=float, default=0.25)
    parser.add_argument('--elastic_alpha_affine', type=float, default=0.45)
    parser.add_argument('--gaussian_truefalse', action='store_true')
    parser.add_argument('--gaussian_min', type=float, default=0) 
    parser.add_argument('--gaussian_max', type=float, default=10)
    parser.add_argument('--gaussian_percentage', type=float, default=0.3)
    parser.add_argument('--gamma_truefalse', action='store_true', help='Apply gamma augmentation')
    parser.add_argument('--gamma_min', type=float, default=80.0)
    parser.add_argument('--gamma_max', type=float, default=120.0)
    parser.add_argument('--gamma_percentage', type=float, default=0.5)
    parser.add_argument('--sharp_percentage', type=float, default=0.3)
    parser.add_argument('--horizontalflip_percentage', type=float, default=0.5)

    # SBATCH Job-related arguments (automatically pulled from Slurm environment if available)
    parser.add_argument('--job_id', type=str, default=os.environ.get("SLURM_JOB_ID", "from_srun"),
                        help="Job id, taken from SLURM_JOB_ID if available.")
    parser.add_argument('--job_array', type=str, default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
                        help="Job array index, taken from SLURM_ARRAY_TASK_ID if available.")
    
    return parser.parse_args()


def get_args_test():
    parser = argparse.ArgumentParser(description="Lateral Image Classification Testing Arguments")
    parser.add_argument('--path', type=str, default='workspace/yeonsu/0.Projects/Pneumoperitoneum/data/external_data/External5.csv', help='Path to CSV for testing')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--size', type=int, default=518)
    parser.add_argument('--pretrained_weight', type=str, default='/workspace/jun/nec_lat/cnn_classification/results/0321_175206_version0/weights/0321_175206_version0_best.pth', help='Path to pretrained model weights for inference')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--include_segmentation', action='store_true', help='Include segmentation or no')
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument("--use_peft", action="store_true", help="Use HuggingFace PEFT for LoRA (requires peft library)")
    parser.add_argument('--apply_mlp_lora', action='store_true', help='Apply LoRA to MLP layers in addition to query and value')
    parser.add_argument('--unfreeze_all', action='store_true', help='Unfreeze and fully fine tune RAD-DINO')
    parser.add_argument("--make_reader_study_cam", action='store_true')
    parser.add_argument("--flipped_orientation_during_training", action='store_true')

    # SBATCH Job-related arguments (automatically pulled from Slurm environment if available)
    parser.add_argument('--job_id', type=str, default=os.environ.get("SLURM_JOB_ID", "from_srun"),
                        help="Job id, taken from SLURM_JOB_ID if available.")
    parser.add_argument('--job_array', type=str, default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
                        help="Job array index, taken from SLURM_ARRAY_TASK_ID if available.")

    parser.add_argument('--model_threshold', type=float, default=0.5)

    return parser.parse_args()
