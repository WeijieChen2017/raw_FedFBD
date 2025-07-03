import argparse
import os
import json
import time
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
    print(f"Debug: total_samples={total_samples}, num_clients={num_clients}, base_samples_per_client={base_samples_per_client}")
    
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
        
        print(f"Debug: Client {i}: min={min_samples}, max={max_samples}, assigned={samples}, remaining={remaining_samples-samples}")
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
    args = parser.parse_args()
    
    # Load configuration from medmnist INFO
    if args.experiment_name not in INFO:
        raise ValueError(f"Dataset {args.experiment_name} is not supported by medmnist.")
    
    info = INFO[args.experiment_name]
    args.task = info['task']
    args.n_channels = 3 if getattr(args, 'as_rgb', False) else info['n_channels']
    args.num_classes = len(info['label'])
    
    # Load additional configuration from config.json
    try:
        config = load_config(args.experiment_name, args.model_flag)
        args_dict = vars(args)
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
        args.norm = "bn"  # Batch normalization
        args.in_channels = 3  # RGB images
    
    # Set output directory for simulation
    args.output_dir = f"fbd_sim_{args.experiment_name}_{args.model_flag}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Verify partition sizes
    print("Actual partition sizes:")
    for i, partition in enumerate(partitions):
        print(f"  Client {i}: {len(partition)} samples")
    print()
    
    print(f"Server: Starting {args.num_rounds}-round simulation for {args.num_clients} clients.")
    
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
        
        # Save results
        history_save_path = os.path.join(args.output_dir, "server_evaluation_history.json")
        with open(history_save_path, 'w') as f:
            json.dump(server_evaluation_history, f, indent=4)
        print(f"Server evaluation history updated for round {r}")
    
    print(f"\nSimulation complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()