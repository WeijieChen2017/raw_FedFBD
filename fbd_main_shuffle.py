import multiprocessing
import argparse
import os
import json
from fbd_server import server_send_to_clients, server_collect_from_clients, end_experiment, evaluate_server_model, prepare_initial_model
from fbd_client import client_task
from fbd_utils import load_config, handle_dataset_cache, setup_logger, load_fbd_settings, FBDWarehouse, save_json
from fbd_dataset import load_data, partition_data
from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_plot import generate_plots
import time
import shutil
import medmnist
from medmnist import INFO
import subprocess
import logging
import torch
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*weights_only=False.*"
)

def initialize_shuffle_experiment(args, dataset_name):
    """Initialize experiment for shuffle mode, handling dataset name correctly"""
    # Import what we need
    from fbd_server import prepare_initial_model
    from fbd_utils import setup_logger, load_fbd_settings, FBDWarehouse, save_json
    from fbd_model_ckpt import get_pretrained_fbd_model
    import torch
    
    # Set up logger
    log_dir = os.path.join(args.output_dir, "fbd_log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "server.log")
    logger = setup_logger("fbd_server", log_file)
    
    logger.info("="*80)
    logger.info("Server: Initializing shuffle experiment")
    logger.info(f"Server: Dataset name for loading: {dataset_name}")
    logger.info(f"Server: Config experiment name: {args.experiment_name}")
    logger.info("="*80)
    
    # 1. Create/clean communication directory
    if getattr(args, 'remove_communication', False) and os.path.exists(args.comm_dir):
        logger.info(f"Server: Clearing communication directory at {args.comm_dir}")
        shutil.rmtree(args.comm_dir)
    
    if not os.path.exists(args.comm_dir):
        os.makedirs(args.comm_dir)
        logger.info(f"Server: Created communication directory at {args.comm_dir}")

    # 2. Handle dataset caching - use the original dataset name
    logger.info("Server: Checking dataset cache...")
    # Create cache directory if it doesn't exist
    if not args.cache_dir:
        args.cache_dir = "cache"
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)
        logger.info(f"Created cache directory: {args.cache_dir}")
    handle_dataset_cache(dataset_name, args.cache_dir)

    # 3. Prepare and cache the initial model
    logger.info("Server: Preparing initial model...")
    prepare_initial_model(args)
    
    # 4. Plans are already pre-generated for shuffle experiments
    logger.info("Server: Using pre-generated shuffle plans...")
    
    # 5. Initialize FBD Warehouse
    logger.info("Server: Initializing FBD Warehouse...")
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.json")
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)

    model_template = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=getattr(args, 'norm', 'bn'),
        in_channels=getattr(args, 'n_channels', 3),
        num_classes=args.num_classes,
        use_pretrained=True
    )
    initial_model_path = os.path.join(args.cache_dir, f"initial_{args.model_flag}.pth")
    model_template.load_state_dict(torch.load(initial_model_path))

    args.warehouse = FBDWarehouse(
        fbd_trace=fbd_trace,
        model_template=model_template,
        log_file_path=os.path.join(args.comm_dir, "warehouse.log")
    )
    
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")
    args.warehouse.save_warehouse(warehouse_path)
    logger.info(f"Server: FBD Warehouse initialized and saved to {warehouse_path}")
    
    # 6. Save training configuration
    config_path = os.path.join(args.comm_dir, "train_config.json")
    logger.info(f"Server: Saving training configuration to {config_path}")
    
    # Create a serializable copy of the configuration
    args_to_save = vars(args).copy()
    args_to_save.pop('warehouse', None)
    args_to_save.pop('test_dataset', None)
    save_json(args_to_save, config_path)
    
    # 7. Prepare test dataset
    logger.info("Server: Preparing test dataset...")
    from medmnist import INFO
    import medmnist
    import torchvision.transforms as transforms
    
    info = INFO[dataset_name]  # Use dataset_name, not experiment_name
    DataClass = getattr(medmnist, info['python_class'])
    as_rgb = getattr(args, 'as_rgb', False)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    args.test_dataset = DataClass(
        split='test', 
        transform=data_transform, 
        download=True, 
        as_rgb=as_rgb,
        root=args.cache_dir
    )
    logger.info("Server: Test dataset prepared.")

def main():
    parser = argparse.ArgumentParser(description="Federated Barter-based Data Exchange Framework - Shuffle")
    parser.add_argument("--experiment_name", type=str, default="bloodmnist", help="Name of the experiment.")
    parser.add_argument("--model_flag", type=str, default="resnet18", help="Model flag.")
    parser.add_argument("--comm_dir", type=str, default="fbd_comm", help="Directory for communication files.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to the model and weights cache.")
    parser.add_argument("--poll_interval", type=float, default=1.0, help="Polling interval in seconds for file checks.")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution.")
    # Add regularization arguments from fbd_main_tau.py
    parser.add_argument("--reg", type=str, choices=["w", "y", "none"], default=None, 
                        help="Regularizer type: 'w' for weights distance, 'y' for consistency loss, 'none' for no regularizer")
    parser.add_argument("--reg_coef", type=float, default=None, 
                        help="Regularization coefficient (overrides config if specified)")
    parser.add_argument("--ensemble_size", type=int, default=None,
                        help="Ensemble size for evaluation (overrides config if specified)")
    args = parser.parse_args()

    # Store the original experiment name for medmnist dataset loading
    dataset_name = args.experiment_name
    # Add _shuffle suffix for config loading
    config_experiment_name = f"{args.experiment_name}_shuffle"

    # 0. Load configuration from medmnist INFO instead of config.json
    if dataset_name not in INFO:
        raise ValueError(f"Dataset {dataset_name} is not supported by medmnist.")
    
    info = INFO[dataset_name]
    args.task = info['task']
    args.n_channels = 3 if getattr(args, 'as_rgb', False) else info['n_channels']
    args.num_classes = len(info['label'])
    
    # Load additional configuration from config.json for non-dataset parameters
    try:
        config = load_config(config_experiment_name, args.model_flag)
        args_dict = vars(args)
        # Only update non-dataset-specific parameters
        for key, value in vars(config).items():
            if key not in ['num_classes', 'task', 'n_channels']:
                args_dict[key] = value
    except Exception as e:
        print(f"Warning: Could not load config.json, using defaults: {e}")
        # Set defaults for required parameters
        args.num_clients = 6
        args.num_rounds = 30
        args.batch_size = 128
        args.local_learning_rate = 0.001
        args.local_epochs = 1
        args.size = 28
        args.num_ensemble = 24
        args.seed = 42
        args.remove_communication = False
        args.training_save_dir = "fbd_results"

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
            fbd_settings_path = f"config/{config_experiment_name}/fbd_settings.json"
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

    # Define temporary and final output directories. The experiment runs in a temporary location
    # and is moved to the final destination only upon successful completion.
    temp_output_dir = os.path.join(f"fbd_run", f"{config_experiment_name}_{args.model_flag}{reg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
    args.comm_dir = os.path.join(temp_output_dir, "fbd_comm")
    os.makedirs(args.comm_dir, exist_ok=True)
    final_output_dir = os.path.join(args.training_save_dir, f"{config_experiment_name}_{args.model_flag}{reg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Clean up the temporary directory from any previous failed runs
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)

    args.output_dir = temp_output_dir
    
    # Store dataset name separately for medmnist operations
    args.dataset_name = dataset_name
    # Update args.experiment_name to include _shuffle for config loading
    args.experiment_name = config_experiment_name
    
    # 1. Initialize Experiment - use custom function that handles dataset name correctly
    initialize_shuffle_experiment(args, dataset_name)
    
    # 1.A. Load and partition data
    print("Server: Loading and partitioning data...")
    # Temporarily override experiment_name for data loading
    temp_experiment_name = args.experiment_name
    args.experiment_name = dataset_name
    train_dataset, _ = load_data(args)
    args.experiment_name = temp_experiment_name
    partitions = partition_data(train_dataset, args.num_clients, args.iid)

    print(f"Server: Starting {args.num_rounds}-round simulation for {args.num_clients} clients.")

    # 1.5. Evaluate server models
    # print("Server: Evaluating initial models from warehouse...")
    # for i in range(6):  # Evaluate models M0 to M5
    #     model_color = f"M{i}"
    #     evaluate_server_model(args, model_color, args.model_flag, args.experiment_name, args.test_dataset, args.warehouse)

    # 2. Start client processes
    processes = []
    for i in range(args.num_clients):
        process = multiprocessing.Process(target=client_task, args=(i, partitions[i], args))
        processes.append(process)
        process.start()

    # Define path for server-side evaluation history log
    eval_results_dir = os.path.join(temp_output_dir, "eval_results")
    os.makedirs(eval_results_dir, exist_ok=True)
    history_save_path = os.path.join(eval_results_dir, "server_evaluation_history.json")

    # 3. Run Rounds
    server_evaluation_history = []
    for r in range(args.num_rounds):
        # Server sends tasks to clients
        server_send_to_clients(r, args)
        
        # Server collects responses from clients and gets eval results
        round_eval_results = server_collect_from_clients(r, args)
        server_evaluation_history.append(round_eval_results)

        # Save the complete server evaluation history after each round
        with open(history_save_path, 'w') as f:
            json.dump(server_evaluation_history, f, indent=4)
        print(f"Server evaluation history updated for round {r} at {history_save_path}")

    # 4. Evaluate server models
    # print("Server: Evaluating final models from warehouse...")
    # for i in range(6):  # Evaluate models M0 to M5
    #     model_color = f"M{i}"
    #     evaluate_server_model(args, model_color, args.model_flag, args.experiment_name, args.test_dataset, args.warehouse)

    # 5. End Experiment
    end_experiment(args)

    # Wait for all client processes to finish
    for process in processes:
        process.join()

    # 6. Generate and save plots
    # print("Generating plots for the experiment...")
    # generate_plots(args.output_dir)

    # print("Framework execution complete.")

    # 7. Move the temporary run folder to its final destination
    try:
        shutil.move(temp_output_dir, final_output_dir)
        print(f"Experiment results saved to: {final_output_dir}")
    except Exception as e:
        print(f"Error moving results to final destination: {e}")

if __name__ == "__main__":
    main()