import argparse
import os
import json
import time
import shutil
import medmnist
import logging
import torch
import torch.utils.data
from fbd_utils import load_config
from medmnist import INFO

# Define SIIM-specific INFO entry since SIIM is not a MedMNIST dataset
SIIM_INFO = {
    'siim': {
        'task': 'segmentation',
        'description': 'SIIM-ACR Pneumothorax Segmentation Challenge',
        'n_channels': 1,
        'label': {'0': 'background', '1': 'pneumothorax'},
        'license': 'SIIM'
    }
}
from fbd_dataset_siim import load_siim_data, partition_siim_data
from fbd_client_siim import simulate_client_task
import numpy as np
import random
from fbd_server_siim import (
    initialize_server_simulation, 
    load_simulation_plans, 
    prepare_test_dataset,
    collect_and_evaluate_round,
    get_client_plans_for_round
)
from fbd_models_siim import get_siim_model

# Suppress noisy logging messages
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)

def load_fold_config(fold_idx):
    """Load fold configuration from saved JSON files."""
    fold_config_path = f"config/siim/siim_fbd_fold_{fold_idx}.json"
    
    if not os.path.exists(fold_config_path):
        raise FileNotFoundError(f"Fold configuration file not found: {fold_config_path}")
    
    with open(fold_config_path, 'r') as f:
        fold_config = json.load(f)
    
    return fold_config

def create_siim_fold_partitions(fold_config, args):
    """Create SIIM client partitions from fold configuration."""
    from fbd_dataset_siim import SIIMSegmentationDataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
        CenterSpatialCropd, RandFlipd, RandRotate90d, RandShiftIntensityd,
        RandScaleIntensityd, ToTensord, EnsureTyped, Resized, DivisiblePadd
    )
    
    # Define transforms (same as in load_siim_data)
    min_intensity = -1024
    max_intensity = 1976
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], b_min=0.0, b_max=1.0, 
                           a_min=min_intensity, a_max=max_intensity, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size),
        DivisiblePadd(keys=["image", "label"], k=16),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
        RandScaleIntensityd(keys=["image"], prob=0.5, factors=0.1),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], b_min=0.0, b_max=1.0, 
                           a_min=min_intensity, a_max=max_intensity, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size),
        DivisiblePadd(keys=["image", "label"], k=16),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Create client partitions from fold configuration
    client_partitions = []
    train_data = fold_config['train']
    
    print(f"Creating SIIM client partitions for fold {args.fold}:")
    total_samples = 0
    
    # Get the data root directory from args
    # The fold paths are relative to the parent directory of SIIM_Fed_Learning_Phase1Data
    data_root = getattr(args, 'data_root', 'siim-101')
    
    # If data_root ends with SIIM_Fed_Learning_Phase1Data, use its parent directory
    if data_root.endswith('SIIM_Fed_Learning_Phase1Data'):
        data_root = os.path.dirname(data_root)
    
    print(f"Using data root: {data_root}")
    
    for client_id in range(args.num_clients):
        client_key = f"client_{client_id}"
        if client_key in train_data:
            client_samples = train_data[client_key]
            
            # Resolve relative paths using data_root
            resolved_samples = []
            for sample in client_samples:
                resolved_sample = {}
                for key, path in sample.items():
                    if isinstance(path, str) and not os.path.isabs(path):
                        # Convert relative path to absolute path using data_root
                        resolved_path = os.path.join(data_root, path)
                        
                        # Handle the case where there's an extra folder with the same filename
                        if not os.path.exists(resolved_path) and path.endswith('.nii.gz'):
                            # Extract filename from path
                            filename = os.path.basename(path)
                            # Try the nested structure: path/filename
                            nested_path = os.path.join(data_root, path, filename)
                            if os.path.exists(nested_path):
                                resolved_path = nested_path
                        
                        resolved_sample[key] = resolved_path
                    else:
                        resolved_sample[key] = path
                resolved_samples.append(resolved_sample)
            
            print(f"  Client {client_id}: {len(resolved_samples)} samples")
            total_samples += len(resolved_samples)
            
            # Create dataset for this client
            client_dataset = SIIMSegmentationDataset(resolved_samples, transforms=train_transforms)
            client_partitions.append(client_dataset)
        else:
            print(f"  Client {client_id}: No data found")
            # Create empty dataset
            client_dataset = SIIMSegmentationDataset([], transforms=train_transforms)
            client_partitions.append(client_dataset)
    
    print(f"Total training samples: {total_samples}")
    
    # Create test dataset
    test_data = []
    for client_id in range(args.num_clients):
        client_key = f"client_{client_id}"
        if client_key in fold_config['test']:
            test_data.extend(fold_config['test'][client_key])
    
    # Resolve test data paths as well
    resolved_test_data = []
    for sample in test_data:
        resolved_sample = {}
        for key, path in sample.items():
            if isinstance(path, str) and not os.path.isabs(path):
                # Convert relative path to absolute path using data_root
                resolved_path = os.path.join(data_root, path)
                
                # Handle the case where there's an extra folder with the same filename
                if not os.path.exists(resolved_path) and path.endswith('.nii.gz'):
                    # Extract filename from path
                    filename = os.path.basename(path)
                    # Try the nested structure: path/filename
                    nested_path = os.path.join(data_root, path, filename)
                    if os.path.exists(nested_path):
                        resolved_path = nested_path
                
                resolved_sample[key] = resolved_path
            else:
                resolved_sample[key] = path
        resolved_test_data.append(resolved_sample)
    
    test_dataset = SIIMSegmentationDataset(resolved_test_data, transforms=test_transforms)
    print(f"Test dataset: {len(test_dataset)} samples")
    
    return client_partitions, test_dataset

def client_dataset_distribution(total_samples, num_clients, variation_ratio=0.3, seed=None):
    """
    Distribute dataset samples among clients with random variation around equal distribution.
    
    Args:
        total_samples (int): Total number of samples in the dataset
        num_clients (int): Number of clients to distribute to
        variation_ratio (float): Maximum variation from equal split (0.3 = ±30%)
        seed (int): Random seed for reproducibility
        
    Returns:
        list: Number of samples for each client
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Start with equal distribution
    base_samples_per_client = total_samples // num_clients
    # Generate random variations for each client
    client_samples = []
    remaining_samples = total_samples
    
    for i in range(num_clients - 1):  # Handle n-1 clients first
        # Calculate variation range
        max_variation = int(base_samples_per_client * variation_ratio)
        min_samples = max(1, base_samples_per_client - max_variation)
        max_samples = min(remaining_samples - (num_clients - i - 1), 
                         base_samples_per_client + max_variation)
        
        # Randomly assign samples within the variation range
        if max_samples > min_samples:
            samples = random.randint(min_samples, max_samples)
        else:
            samples = min_samples
        
        client_samples.append(samples)
        remaining_samples -= samples
    
    # Last client gets all remaining samples
    client_samples.append(remaining_samples)
    
    return client_samples

def create_client_partitions(train_dataset, num_clients, iid=False, variation_ratio=0.3, seed=None):
    """
    Create client-specific data partitions with varied sizes.
    
    Args:
        train_dataset: The training dataset to partition
        num_clients (int): Number of clients
        iid (bool): Whether to use IID data distribution
        variation_ratio (float): Dataset size variation around equal split
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of dataset partitions for each client
    """
    total_samples = len(train_dataset)
    
    # Get sample distribution for each client
    client_sample_counts = client_dataset_distribution(
        total_samples, num_clients, variation_ratio, seed
    )
    
    # Print distribution summary
    avg_samples = total_samples / num_clients
    print(f"Dataset Distribution Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Average per client: {avg_samples:.1f}")
    for i, count in enumerate(client_sample_counts):
        percentage = (count / avg_samples - 1) * 100
        sign = "+" if percentage >= 0 else ""
        print(f"  Client {i}: {count} samples ({sign}{percentage:.1f}%)")
    
    # Create partitions with the specified sizes
    if iid:
        # For IID: randomly shuffle and split according to sample counts
        indices = list(range(total_samples))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        
        partitions = []
        start_idx = 0
        for count in client_sample_counts:
            end_idx = start_idx + count
            client_indices = indices[start_idx:end_idx]
            partitions.append(torch.utils.data.Subset(train_dataset, client_indices))
            start_idx = end_idx
    else:
        # For non-IID: use existing partition_data but adjust sizes
        # First get standard partitions
        base_partitions = partition_data(train_dataset, num_clients, iid=False)
        
        # Then adjust partition sizes by resampling
        partitions = []
        for i, target_count in enumerate(client_sample_counts):
            base_partition = base_partitions[i]
            base_indices = base_partition.indices
            
            if len(base_indices) >= target_count:
                # Subsample if we have more than needed
                if seed is not None:
                    np.random.seed(seed + i)
                selected_indices = np.random.choice(base_indices, target_count, replace=False)
            else:
                # Oversample if we need more (with replacement)
                if seed is not None:
                    np.random.seed(seed + i)
                selected_indices = np.random.choice(base_indices, target_count, replace=True)
            
            partitions.append(torch.utils.data.Subset(train_dataset, selected_indices.tolist()))
    
    return partitions


def main():
    parser = argparse.ArgumentParser(description="Federated Barter-based Data Exchange Framework - Simulation")
    parser.add_argument("--experiment_name", type=str, default="siim", help="Name of the experiment.")
    parser.add_argument("--model_flag", type=str, default="unet", help="Model flag.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to the model and weights cache.")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution.")
    parser.add_argument("--fold", type=int, default=None, help="Fold number for cross-validation (0-3). If not specified, uses original dataset loading.")
    parser.add_argument("--model_size", type=str, choices=["small", "standard", "large", "xlarge", "xxlarge", "mega"], default="standard", help="Size of the model: 'small' (48), 'standard' (96), 'large' (192), 'xlarge' (384), 'xxlarge' (576), 'mega' (768 features).")
    parser.add_argument("--auto_detect_size", action="store_true", help="Automatically detect model size from saved weights (useful when switching between standard/small)")
    parser.add_argument("--eval_on_cpu", action="store_true", help="Force model evaluation on CPU to save GPU memory (useful for large models)")
    parser.add_argument("--init_method", type=str, choices=["pretrained", "shared_random", "random"], default="pretrained", 
                        help="Model initialization method: 'pretrained' uses medical weights, 'shared_random' uses same random seed for all clients, 'random' uses different random initialization")
    parser.add_argument("--reg", type=str, choices=["w", "y", "none"], default=None, 
                        help="Regularizer type: 'w' for weights distance, 'y' for consistency loss, 'none' for no regularizer")
    parser.add_argument("--reg_coef", type=float, default=None, 
                        help="Regularization coefficient (overrides config if specified)")
    parser.add_argument("--ensemble_size", type=int, default=None,
                        help="Ensemble size for evaluation (overrides config if specified)")
    parser.add_argument("--parallel", action="store_true", help="Run multiple client models in parallel (uses smaller models)")
    parser.add_argument("--auto_approval", action="store_true", help="Automatically approve the training plan")
    args = parser.parse_args()
    
    # Handle SIIM dataset configuration
    if args.experiment_name == "siim":
        # For SIIM, we set task and num_classes directly
        args.task = "segmentation"
        args.num_classes = 1  # Binary segmentation
        args.n_channels = 1  # Grayscale medical images
    else:
        # For MedMNIST datasets
        if args.experiment_name in INFO:
            info = INFO[args.experiment_name]
        elif args.experiment_name in SIIM_INFO:
            info = SIIM_INFO[args.experiment_name]
        else:
            raise ValueError(f"Dataset {args.experiment_name} is not supported.")
        
        args.task = info['task']
        args.num_classes = len(info['label'])
    
    # Load additional configuration from config.json (required)
    config = load_config(args.experiment_name, args.model_flag)
    args_dict = vars(args)
    for key, value in vars(config).items():
        if key not in ['num_classes', 'task']:
            args_dict[key] = value
    
    # Set n_channels based on config's as_rgb setting (after config is loaded)
    if args.experiment_name != "siim":
        args.n_channels = 3 if getattr(args, 'as_rgb', False) else info['n_channels']
    
    # Override ensemble_size if provided via command line
    if 'ensemble_size' in args_dict and args_dict['ensemble_size'] is not None:
        # Command line argument was provided, it takes precedence
        cmdline_ensemble_size = args_dict['ensemble_size']
        config_ensemble_size = getattr(config, 'ensemble_size', 'not set')
        if config_ensemble_size != 'not set' and cmdline_ensemble_size != config_ensemble_size:
            print(f"Overriding ensemble_size from config ({config_ensemble_size}) to command line value ({cmdline_ensemble_size})")
        args.ensemble_size = cmdline_ensemble_size
    
    # Determine regularizer settings for directory naming
    if args.reg is not None:
        if args.reg == 'none':
            reg_suffix = "_reg_none"
        else:
            reg_type_str = args.reg
            reg_coef_str = f"{args.reg_coef:.3f}" if args.reg_coef is not None else "0.100"
            reg_suffix = f"_reg_{reg_type_str}_coef_{reg_coef_str}"
    else:
        # Check config file for regularizer settings
        try:
            fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
            with open(fbd_settings_path, 'r') as f:
                fbd_settings = json.load(f)
            regularizer_params = fbd_settings.get('REGULARIZER_PARAMS', {})
            reg_type = regularizer_params.get('type')
            if reg_type == 'consistency loss':
                reg_type_str = 'y'
            elif reg_type == 'weights distance':
                reg_type_str = 'w'
            else:
                reg_type_str = 'none'
            
            if reg_type_str != 'none':
                reg_coef = regularizer_params.get('coefficient', 0.1)
                reg_suffix = f"_reg_{reg_type_str}_coef_{reg_coef:.3f}"
            else:
                reg_suffix = "_reg_none"
        except:
            reg_suffix = ""
    
    # Add fold suffix to output directory name
    fold_suffix = f"_fold_{args.fold}" if args.fold is not None else ""
    
    # Define temporary and final output directories (like original fbd_main.py)
    temp_output_dir = os.path.join(f"fbd_run", f"{args.experiment_name}_{args.model_flag}{reg_suffix}{fold_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
    final_output_dir = os.path.join(args.training_save_dir, f"{args.experiment_name}_{args.model_flag}{reg_suffix}{fold_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Clean up the temporary directory from any previous failed runs
    if os.path.exists(temp_output_dir):
        import shutil
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)
    
    args.output_dir = temp_output_dir
    
    # Set up logging directory (like original)
    log_dir = os.path.join(args.output_dir, "fbd_log")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize server simulation
    warehouse = initialize_server_simulation(args)
    shipping_plans, update_plans = load_simulation_plans(args)
    
    # Load and partition data
    print("Server: Loading and partitioning data...")
    if args.experiment_name == "siim":
        if args.fold is not None:
            # Use fold configuration
            if args.fold not in [0, 1, 2, 3]:
                raise ValueError(f"Fold must be 0, 1, 2, or 3, got {args.fold}")
            
            print(f"Loading SIIM data using fold configuration {args.fold}")
            fold_config = load_fold_config(args.fold)
            partitions, test_dataset = create_siim_fold_partitions(fold_config, args)
            args.test_dataset = test_dataset
        else:
            # Use original loading method
            print("Loading SIIM data using original method")
            train_dataset, test_dataset = load_siim_data(args)
            partitions = partition_siim_data(train_dataset, args.num_clients, args.iid)
            args.test_dataset = test_dataset
    else:
        from fbd_dataset import load_data, partition_data
        train_dataset, _ = load_data(args)
        partitions = create_client_partitions(
            train_dataset, 
            args.num_clients, 
            args.iid, 
            variation_ratio=0.3,  # ±30% variation from equal split
            seed=args.seed + 100  # Different seed for partition variation
        )
        args.test_dataset = prepare_test_dataset(args)
    
    # Generate and display training plan
    print("\n" + "="*80)
    print("TRAINING PLAN SUMMARY")
    print("="*80)
    print(f"\n1. DATASET AND MODEL:")
    print(f"   - Dataset: {args.experiment_name}")
    print(f"   - Model: {args.model_flag}")
    print(f"   - Task: {args.task}")
    print(f"   - Number of classes: {args.num_classes}")
    print(f"   - Input channels: {args.n_channels}")
    if args.experiment_name == "siim":
        print(f"   - ROI size: {args.roi_size}")
        print(f"   - Model size: {args.model_size}")
        
        # Add memory estimate (based on actual measurements - smaller models)
        memory_estimates = {
            'small': '48MB', 'standard': '192MB', 'large': '766MB', 
            'xlarge': '3.06GB', 'xxlarge': '6.89GB', 'mega': '12.24GB'
        }
        memory_str = memory_estimates.get(args.model_size, 'Unknown')
        print(f"   - Estimated memory per model: {memory_str}")
        
        # Add evaluation device info
        eval_device = "CPU" if getattr(args, 'eval_on_cpu', False) else "GPU"
        print(f"   - Evaluation device: {eval_device}")
        
        # Memory usage warning for large models on GPU
        if args.model_size in ['large', 'xlarge'] and not getattr(args, 'eval_on_cpu', False):
            total_eval_memory = memory_str
            if args.parallel:
                print(f"   ⚠️  Training + Evaluation on GPU may use significant memory")
                print(f"   💡 Consider --eval_on_cpu for GPU memory savings")
            else:
                print(f"   💡 Large model evaluation uses {total_eval_memory} GPU memory")
        
        if args.fold is not None:
            print(f"   - Cross-validation fold: {args.fold}")
    
    print(f"\n2. FEDERATED LEARNING SETUP:")
    print(f"   - Number of clients: {args.num_clients}")
    print(f"   - Number of rounds: {args.num_rounds}")
    print(f"   - Local epochs per round: {args.local_epochs}")
    print(f"   - Local batch size: {getattr(args, 'batch_size', 128)}")
    print(f"   - Local learning rate: {getattr(args, 'local_learning_rate', 0.001)}")
    print(f"   - Data distribution: {'IID' if args.iid else 'Non-IID'}")
    if args.fold is None:
        print(f"   - Sample variation: ±30% from equal distribution")
    else:
        print(f"   - Data partitioning: Pre-defined fold configuration")
    print(f"   - Parallel mode: {'ENABLED (' + args.model_size + ' models)' if args.parallel else 'DISABLED (single model)'}")
    
    # Add initialization method info
    init_method = getattr(args, 'init_method', 'pretrained')
    init_descriptions = {
        'pretrained': 'Medical pretrained weights (MONAI/lungmask)',
        'shared_random': f'Shared random initialization (seed: {args.seed})',
        'random': 'Different random weights per client'
    }
    print(f"   - Initialization: {init_descriptions.get(init_method, init_method)}")
    
    print(f"\n3. REGULARIZATION:")
    # Determine regularization settings for display
    if args.reg is not None:
        if args.reg == 'none':
            print(f"   - Type: None")
            print(f"   - Coefficient: N/A")
        else:
            reg_type = 'Weights Distance' if args.reg == 'w' else 'Consistency Loss'
            reg_coef = args.reg_coef if args.reg_coef is not None else 0.1
            print(f"   - Type: {reg_type}")
            print(f"   - Coefficient: {reg_coef}")
            print(f"   - Source: Command line arguments")
    else:
        # Check config file
        try:
            fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
            with open(fbd_settings_path, 'r') as f:
                fbd_settings = json.load(f)
            regularizer_params = fbd_settings.get('REGULARIZER_PARAMS', {})
            reg_type = regularizer_params.get('type', 'None')
            if reg_type and reg_type != 'None':
                reg_coef = regularizer_params.get('coefficient', 0.1)
                print(f"   - Type: {reg_type}")
                print(f"   - Coefficient: {reg_coef}")
                print(f"   - Source: Config file (fbd_settings.json)")
            else:
                print(f"   - Type: None")
                print(f"   - Coefficient: N/A")
        except:
            print(f"   - Type: None (config not found)")
            print(f"   - Coefficient: N/A")
    
    print(f"\n4. OUTPUT:")
    print(f"   - Output directory: {temp_output_dir}")
    print(f"   - Final directory: {final_output_dir}")
    
    print(f"\n5. FBD CONFIGURATION:")
    print(f"   - Model colors: {getattr(args, 'colors', ['red', 'yellow', 'blue'])}")
    print(f"   - Block assignment: {getattr(args, 'block_assignment', 'cyclic')}")
    print(f"   - Ensemble size: {getattr(args, 'ensemble_size', 1)}")
    print(f"   - Optimizer: Adam")
    print(f"   - Seed: {getattr(args, 'seed', 42)}")
    
    print("\n" + "="*80)
    
    # Ask for user approval
    if not args.auto_approval:
        approval = input("\nDo you want to proceed with this training plan? (yes/no): ").strip().lower()
        if approval not in ['yes', 'y']:
            print("Training cancelled by user.")
            return
    
    print(f"\nServer: Starting {args.num_rounds}-round simulation for {args.num_clients} clients.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed based on initialization method
    if args.init_method in ["shared_random", "pretrained"]:
        # Use same seed for all clients to ensure identical starting weights
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎯 Initialization method: {args.init_method}")
        print(f"🎲 Set shared random seed to {args.seed} for reproducible model initialization")
    else:
        print(f"🎯 Initialization method: {args.init_method} (different random weights per client)")
        print(f"🎲 Using different random initialization for each client")
    
    if args.parallel:
        # Create multiple models for parallel execution
        print(f"Creating {args.num_clients} models for parallel execution...")
        client_models = []
        
        if args.experiment_name == "siim":
            # Use the specified model size, but warn if it might not fit
            parallel_model_size = args.model_size
            
            # Estimate memory usage for parallel training (based on actual measurements - smaller models)
            memory_per_model = {
                'small': 0.048, 'standard': 0.192, 'large': 0.766, 
                'xlarge': 3.06, 'xxlarge': 6.89, 'mega': 12.24
            }
            
            model_memory = memory_per_model.get(parallel_model_size, 1.0)
            total_memory_gb = args.num_clients * model_memory
            
            if total_memory_gb > 20:  # Leave some headroom on 24GB GPU
                print(f"⚠️  Warning: {args.num_clients} {parallel_model_size} models may use ~{total_memory_gb:.1f}GB.")
                print(f"💡 Consider reducing --num_clients or using --eval_on_cpu")
            elif total_memory_gb > 12:
                print(f"📊 {args.num_clients} {parallel_model_size} models will use ~{total_memory_gb:.1f}GB GPU memory")
            
            # Suggest optimal configurations
            if parallel_model_size in ['xxlarge', 'mega']:
                optimal_clients = int(20 / model_memory)  # Leave 4GB headroom
                if optimal_clients < args.num_clients:
                    print(f"💡 For {parallel_model_size} models, consider --num_clients {optimal_clients} for optimal GPU usage")
            
            for i in range(args.num_clients):
                # Set different seed for each client if using random initialization
                if args.init_method == "random":
                    client_seed = args.seed + i + 1000  # Offset to avoid conflicts
                    torch.manual_seed(client_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(client_seed)
                        torch.cuda.manual_seed_all(client_seed)
                    print(f"  Client {i}: Using random seed {client_seed}")
                
                model = get_siim_model(
                    architecture=args.model_flag,
                    in_channels=args.n_channels,
                    out_channels=args.num_classes,
                    model_size=parallel_model_size
                )
                client_models.append(model)
        else:
            # This path is not expected for siim, but as a fallback:
            from fbd_models import get_pretrained_fbd_model
            for i in range(args.num_clients):
                model = get_pretrained_fbd_model(
                    architecture=args.model_flag,
                    norm=args.norm,
                    in_channels=args.n_channels,
                    num_classes=args.num_classes,
                    use_pretrained=False
                )
                client_models.append(model)
    else:
        # Create a single reusable model instance to save memory
        if args.experiment_name == "siim":
            reusable_model = get_siim_model(
                architecture=args.model_flag,
                in_channels=args.n_channels,
                out_channels=args.num_classes,
                model_size=args.model_size
            )
        else:
            # This path is not expected for siim, but as a fallback:
            from fbd_models import get_pretrained_fbd_model
            reusable_model = get_pretrained_fbd_model(
                architecture=args.model_flag,
                norm=args.norm,
                in_channels=args.n_channels,
                num_classes=args.num_classes,
                use_pretrained=False
            )
    
    # Run simulation rounds
    server_evaluation_history = []
    for r in range(args.num_rounds):
        print(f"\n=== Round {r} ===")
        
        # Collect client responses for this round
        client_responses = {}
        
        if args.parallel:
            # Run clients in parallel using concurrent.futures
            import concurrent.futures
            
            def run_client(client_id, model):
                client_shipping_list, client_update_plan = get_client_plans_for_round(
                    r, client_id, shipping_plans, update_plans
                )
                return simulate_client_task(
                    model, client_id, partitions[client_id], args, r, 
                    warehouse, client_shipping_list, client_update_plan, use_disk=False
                )
            
            # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound
            # Using ThreadPoolExecutor here since PyTorch models may have issues with multiprocessing
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_clients) as executor:
                # Submit all client tasks
                future_to_client = {
                    executor.submit(run_client, client_id, client_models[client_id]): client_id 
                    for client_id in range(args.num_clients)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_client):
                    client_id = future_to_client[future]
                    try:
                        response = future.result()
                        client_responses[client_id] = response
                    except Exception as e:
                        print(f"Client {client_id} generated an exception: {e}")
                        raise
        else:
            # Simulate all clients for this round sequentially
            for client_id in range(args.num_clients):
                client_shipping_list, client_update_plan = get_client_plans_for_round(
                    r, client_id, shipping_plans, update_plans
                )
                
                response = simulate_client_task(
                    reusable_model, client_id, partitions[client_id], args, r, 
                    warehouse, client_shipping_list, client_update_plan, use_disk=True
                )
                client_responses[client_id] = response
        
        # Server collects responses and evaluates
        round_eval_results = collect_and_evaluate_round(r, args, warehouse, client_responses)
        server_evaluation_history.append(round_eval_results)
        
        # Save results to eval_results directory (like original)
        eval_results_dir = os.path.join(args.output_dir, "eval_results")
        os.makedirs(eval_results_dir, exist_ok=True)
        
        # Save individual model results for this round
        for model_color, metrics in round_eval_results.items():
            if model_color != 'round':
                model_dir = os.path.join(eval_results_dir, args.experiment_name, args.model_flag, model_color)
                os.makedirs(model_dir, exist_ok=True)
                
                result_file = os.path.join(model_dir, f"round_{r}_eval_metrics.json")
                with open(result_file, 'w') as f:
                    json.dump({
                        "round": r,
                        "model_color": model_color,
                        "model_name": args.model_flag,
                        "dataset": args.experiment_name,
                        "fold": args.fold,
                        **metrics
                    }, f, indent=4)
        
        # Save complete server evaluation history
        history_save_path = os.path.join(eval_results_dir, "server_evaluation_history.json")
        with open(history_save_path, 'w') as f:
            json.dump(server_evaluation_history, f, indent=4)
            
        # Save warehouse state after each round
        warehouse_save_path = os.path.join(args.output_dir, f"fbd_warehouse_round_{r}.pth")
        warehouse.save_warehouse(warehouse_save_path)
        
        print(f"Server evaluation history and warehouse updated for round {r}")
    
    # Move the temporary run folder to its final destination (like original)
    # try:
    #     shutil.move(temp_output_dir, final_output_dir)
    #     print(f"\nSimulation complete! Results saved to: {final_output_dir}")
    # except Exception as e:
    #     print(f"Error moving results to final destination: {e}")
    #     print(f"Results remain in: {temp_output_dir}")

if __name__ == "__main__":
    main()