import argparse
import os

def get_args_train():
    parser = argparse.ArgumentParser(description="Lateral Image Classification Training Arguments")
    # Basic I/O and Paths
    parser.add_argument('--path', type=str, default='workspace/yeonsu/0.Projects/Pneumoperitoneum/data/internal_data/Internal6.csv', help='Path to CSV with lateral image data')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--version', type=int, default=0)

    # Custom Args
    parser.add_argument('--inference', action='store_true', help='Run inference instead of training')
    parser.add_argument('--pretrained_weight', type=str, default='/workspace/jun/nec_lat/cnn_classification/results/0321_175206_version0/weights/0321_175206_version0_best.pth', help='Path to pretrained model weights for inference')
    parser.add_argument('--input_format', type=str, default='png', choices=['png','dcm'], help='Is input image in PNG or DCM?')
    parser.add_argument('--help_info', type=str, default='N/A', help='Basic information to understand the version and the model instance')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank. This is passed automatically by distributed launcher.") # necessary
    parser.add_argument("--edge_augmentation", action='store_true', help="Edge Augmentation")
    parser.add_argument("--custom_batch_atypical", action='store_true')
    parser.add_argument("--drop_last", action='store_true')
    parser.add_argument("--apply_patient_awareness_to_custom_batch", action='store_true', help="patient_aware")
    parser.add_argument("--apply_label_and_patient_awareness", action='store_true', help="label_aware")
    parser.add_argument("--original", action='store_true')

    
    
    # Model & Training
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--layers', type=str, default='densenet169', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext101_32x4d','resnext101_32x8d','inceptionresnetv2','mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7','vgg16','vgg19'])
    parser.add_argument('--batch', type=int, default=6, help='Train batch size.')
    parser.add_argument('--size', type=int, default=518, help='Target image size.')
    parser.add_argument('--lr_type', type=str, default='reduce', choices=['step', 'reduce'] )
    parser.add_argument('--lr_startstep', type=float, default=0.00005)
    parser.add_argument('--lr_patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    
    # Data augmentation parameters
    # parser.add_argument('--clahe_cliplimit', type=float, default=2.0)
    # parser.add_argument('--clahe_limit', type=int, default=8)    
    # parser.add_argument('--clip_min', type=float, default=0.5)
    # parser.add_argument('--clip_max', type=float, default=98.5) # 99.5
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
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--layers', type=str, default='densenet169', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext101_32x4d','resnext101_32x8d','inceptionresnetv2','mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7','vgg16','vgg19'])
    parser.add_argument('--size', type=int, default=518)
    parser.add_argument('--pretrained_weight', type=str, default='/workspace/jun/nec_lat/cnn_classification/results/0321_175206_version0/weights/0321_175206_version0_best.pth', help='Path to pretrained model weights for inference')
    parser.add_argument('--seed', type=int, default=42)

    # SBATCH Job-related arguments (automatically pulled from Slurm environment if available)
    parser.add_argument('--job_id', type=str, default=os.environ.get("SLURM_JOB_ID", "from_srun"),
                        help="Job id, taken from SLURM_JOB_ID if available.")
    parser.add_argument('--job_array', type=str, default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
                        help="Job array index, taken from SLURM_ARRAY_TASK_ID if available.")

    parser.add_argument('--model_threshold', type=float, default=0.5)
    
    return parser.parse_args()
