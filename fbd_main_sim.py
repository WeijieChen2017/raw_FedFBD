import argparse
import os
import json
import time
import shutil
import medmnist
import logging
import torch
import torch.utils.data
from medmnist import INFO

from fbd_utils import load_config
from fbd_dataset import load_data, partition_data
from fbd_client_sim import simulate_client_task
import numpy as np
import random
from fbd_server_sim import (
    initialize_server_simulation, 
    load_simulation_plans, 
    prepare_test_dataset,
    collect_and_evaluate_round,
    get_client_plans_for_round
)

# Suppress noisy logging messages
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)

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
    parser.add_argument("--experiment_name", type=str, default="bloodmnist", help="Name of the experiment.")
    parser.add_argument("--model_flag", type=str, default="resnet18", help="Model flag.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to the model and weights cache.")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution.")
    parser.add_argument("--reg", type=str, choices=["w", "y", "none"], default=None, 
                        help="Regularizer type: 'w' for weights distance, 'y' for consistency loss, 'none' for no regularizer")
    parser.add_argument("--reg_coef", type=float, default=None, 
                        help="Regularization coefficient (overrides config if specified)")
    args = parser.parse_args()
    
    # Load configuration from medmnist INFO
    if args.experiment_name not in INFO:
        raise ValueError(f"Dataset {args.experiment_name} is not supported by medmnist.")
    
    info = INFO[args.experiment_name]
    args.task = info['task']
    args.num_classes = len(info['label'])
    
    # Load additional configuration from config.json (required)
    config = load_config(args.experiment_name, args.model_flag)
    args_dict = vars(args)
    for key, value in vars(config).items():
        if key not in ['num_classes', 'task']:
            args_dict[key] = value
    
    # Set n_channels based on config's as_rgb setting (after config is loaded)
    args.n_channels = 3 if getattr(args, 'as_rgb', False) else info['n_channels']
    
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
    
    # Define temporary and final output directories (like original fbd_main.py)
    temp_output_dir = os.path.join(f"fbd_run", f"{args.experiment_name}_{args.model_flag}{reg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
    final_output_dir = os.path.join(args.training_save_dir, f"{args.experiment_name}_{args.model_flag}{reg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
    
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
    args.test_dataset = prepare_test_dataset(args)
    
    # Load and partition data
    print("Server: Loading and partitioning data...")
    train_dataset, _ = load_data(args)
    partitions = create_client_partitions(
        train_dataset, 
        args.num_clients, 
        args.iid, 
        variation_ratio=0.3,  # ±30% variation from equal split
        seed=args.seed + 100  # Different seed for partition variation
    )
    
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
    
    print(f"\n2. FEDERATED LEARNING SETUP:")
    print(f"   - Number of clients: {args.num_clients}")
    print(f"   - Number of rounds: {args.num_rounds}")
    print(f"   - Local epochs per round: {args.local_epochs}")
    print(f"   - Local batch size: {getattr(args, 'batch_size', 128)}")
    print(f"   - Local learning rate: {getattr(args, 'local_learning_rate', 0.001)}")
    print(f"   - Data distribution: {'IID' if args.iid else 'Non-IID'}")
    print(f"   - Sample variation: ±30% from equal distribution")
    
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
    print(f"   - Optimizer: Adam")
    print(f"   - Seed: {getattr(args, 'seed', 42)}")
    
    print("\n" + "="*80)
    
    # Ask for user approval
    approval = input("\nDo you want to proceed with this training plan? (yes/no): ").strip().lower()
    if approval not in ['yes', 'y']:
        print("Training cancelled by user.")
        return
    
    print(f"\nServer: Starting {args.num_rounds}-round simulation for {args.num_clients} clients.")
    
    # Run simulation rounds
    server_evaluation_history = []
    for r in range(args.num_rounds):
        print(f"\n=== Round {r} ===")
        
        # Collect client responses for this round
        client_responses = {}
        
        # Simulate all clients for this round
        for client_id in range(args.num_clients):
            client_shipping_list, client_update_plan = get_client_plans_for_round(
                r, client_id, shipping_plans, update_plans
            )
            
            response = simulate_client_task(
                client_id, partitions[client_id], args, r, 
                warehouse, client_shipping_list, client_update_plan
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