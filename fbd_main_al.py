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
from fbd_client_al import simulate_client_task
from fbd_server_al import secure_aggregate_counts
from collections import defaultdict
import numpy as np
import random
from fbd_server_al import (
    initialize_server_simulation, 
    load_simulation_plans, 
    prepare_test_dataset,
    collect_and_evaluate_round,
    get_client_plans_for_round
)

# Suppress noisy logging messages
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)

def initialize_active_learning(partitions, num_clients, initial_frac, seed=None):
    """
    Initialize active learning state with a small fraction of labeled data.
    
    Args:
        partitions: List of client data partitions
        num_clients: Number of clients
        initial_frac: Initial fraction of data to label
        seed: Random seed
    
    Returns:
        dict: Active learning state
    """
    if seed is not None:
        np.random.seed(seed)
    
    labeled_indices = {}
    unlabeled_indices = {}
    
    for client_id in range(num_clients):
        # Get all indices for this client
        all_indices = list(range(len(partitions[client_id])))
        
        # Randomly select initial labeled samples
        n_initial = max(1, int(len(all_indices) * initial_frac))
        np.random.shuffle(all_indices)
        
        labeled_indices[str(client_id)] = all_indices[:n_initial]
        unlabeled_indices[str(client_id)] = all_indices[n_initial:]
    
    # Calculate global N for secure aggregation
    global_N = sum(len(p) for p in partitions)
    
    return {
        'labeled_indices': labeled_indices,
        'unlabeled_indices': unlabeled_indices,
        'global_N': global_N,
        'al_history': [],
        'perform_selection': False,
        'budget_frac': 0.0
    }

def update_active_learning_state(al_state, selections):
    """
    Update active learning state with newly selected samples.
    
    Args:
        al_state: Current active learning state
        selections: Dict mapping client_id to selected indices
    """
    for client_id, selected_indices in selections.items():
        if selected_indices:
            # Move selected samples from unlabeled to labeled
            al_state['labeled_indices'][client_id].extend(selected_indices)
            al_state['unlabeled_indices'][client_id] = [
                idx for idx in al_state['unlabeled_indices'][client_id] 
                if idx not in selected_indices
            ]
    
    # Record this AL round in history
    al_state['al_history'].append({
        'round': len(al_state['al_history']) + 1,
        'selections': selections,
        'total_selected': sum(len(s) for s in selections.values())
    })

def load_hetero_config(experiment_name):
    """
    Load heterogeneous dataset configuration from hetero_data.json
    
    Args:
        experiment_name: Name of the experiment (e.g., 'organamnist')
    
    Returns:
        dict: Heterogeneous distribution configuration
    """
    hetero_config_path = f"config/{experiment_name}/hetero_data.json"
    
    try:
        with open(hetero_config_path, 'r') as f:
            hetero_config = json.load(f)
        print(f"Loaded heterogeneous configuration from: {hetero_config_path}")
        return hetero_config
    except FileNotFoundError:
        print(f"Warning: Heterogeneous config not found at {hetero_config_path}")
        print("Falling back to standard partitioning...")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing hetero config: {e}")
        return None

def create_hetero_partitions(train_dataset, hetero_config, seed=None):
    """
    Create client partitions based on heterogeneous distribution configuration.
    
    Args:
        train_dataset: The training dataset to partition
        hetero_config: Heterogeneous distribution configuration from hetero_data.json
        seed: Random seed for reproducibility
    
    Returns:
        list: List of dataset partitions for each client
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Get dataset labels
    labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, np.ndarray):
            label = int(label.item()) if label.size == 1 else int(label[0])
        elif hasattr(label, 'item'):
            label = int(label.item())
        else:
            label = int(label)
        labels.append(label)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    num_clients = hetero_config['num_clients']
    num_classes = hetero_config['num_classes']
    
    print(f"\nHeterogeneous Data Distribution:")
    print(f"  Dataset: {hetero_config['dataset']}")
    print(f"  Clients: {num_clients}")
    print(f"  Classes: {num_classes}")
    print(f"  Alpha: {hetero_config['alpha']}")
    
    # Create client partitions
    client_partitions = []
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples according to percentages
    for class_idx in range(num_classes):
        if class_idx not in class_indices:
            print(f"Warning: Class {class_idx} not found in dataset")
            continue
        
        class_samples = class_indices[class_idx]
        total_class_samples = len(class_samples)
        
        # Shuffle class samples for randomness
        random.shuffle(class_samples)
        
        # Calculate how many samples each client gets for this class
        distributed_samples = 0
        for client_idx in range(num_clients):
            client_key = f"client_{client_idx}"
            percentage = hetero_config['client_distributions'][client_key]['class_percentages'][class_idx]
            
            # Calculate number of samples for this client and class
            if client_idx == num_clients - 1:  # Last client gets remaining samples
                samples_for_client = total_class_samples - distributed_samples
            else:
                samples_for_client = int(total_class_samples * percentage / 100.0)
            
            # Assign samples to client
            start_idx = distributed_samples
            end_idx = start_idx + samples_for_client
            client_samples = class_samples[start_idx:end_idx]
            client_indices[client_idx].extend(client_samples)
            distributed_samples += samples_for_client
    
    # Create Subset objects for each client
    for client_idx in range(num_clients):
        client_subset = torch.utils.data.Subset(train_dataset, client_indices[client_idx])
        client_partitions.append(client_subset)
    
    # Print distribution summary
    print(f"\nClient Dataset Distribution:")
    for client_idx in range(num_clients):
        client_size = len(client_partitions[client_idx])
        client_key = f"client_{client_idx}"
        percentages = hetero_config['client_distributions'][client_key]['class_percentages']
        print(f"  Client {client_idx}: {client_size} samples")
        print(f"    Class percentages: {[f'{p:.1f}%' for p in percentages]}")
    
    return client_partitions

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
    parser.add_argument("--ensemble_size", type=int, default=None,
                        help="Ensemble size for evaluation (overrides config if specified)")
    parser.add_argument("--FedAvg", action="store_true", 
                        help="Enable FedAvg-style averaging of model blocks across colors at the end of each epoch")
    parser.add_argument("--save_affix", type=str, default="", 
                        help="String to append to the end of the output directory name")
    parser.add_argument("--auto", action="store_true", 
                        help="Auto-approve training plan without user confirmation")
    # Active learning arguments
    parser.add_argument("--al_initial_frac", type=float, default=0.01,
                        help="Initial fraction of data to label (default: 1%)")
    parser.add_argument("--al_budget_per_round", type=float, default=0.05,
                        help="Fraction of data to label per AL round (default: 5%)")
    parser.add_argument("--al_rounds", type=int, default=4,
                        help="Number of active learning rounds (default: 4 rounds: 1%->6%->11%->16%->21%)")
    parser.add_argument("--al_diversity", action="store_true",
                        help="Enable diversity filtering in active learning")
    parser.add_argument("--al_metric", type=str, choices=["entropy", "ensemble_ratio"], default="entropy",
                        help="Metric to use for active learning selection")
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
    
    # Define temporary and final output directories (like original fbd_main.py)
    fedavg_suffix = "_FA" if args.FedAvg else ""
    save_affix = f"_{args.save_affix}" if args.save_affix else ""
    temp_output_dir = os.path.join(f"fbd_run", f"{args.experiment_name}_{args.model_flag}{reg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}_al{fedavg_suffix}{save_affix}")
    final_output_dir = os.path.join(args.training_save_dir, f"{args.experiment_name}_{args.model_flag}{reg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}_al{fedavg_suffix}{save_affix}")
    
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
    
    # Try to load heterogeneous configuration
    hetero_config = load_hetero_config(f"{args.experiment_name}")
    
    if hetero_config is not None:
        # Use heterogeneous partitioning
        partitions = create_hetero_partitions(
            train_dataset, 
            hetero_config,
            seed=args.seed + 100
        )
    else:
        # Fall back to standard partitioning
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
    if args.save_affix:
        print(f"   - Save affix: '{args.save_affix}'")
    
    print(f"\n5. FBD CONFIGURATION:")
    print(f"   - FedAvg enabled: {'Yes' if args.FedAvg else 'No'}")
    print(f"   - Model colors: {getattr(args, 'colors', ['red', 'yellow', 'blue'])}")
    print(f"   - Block assignment: {getattr(args, 'block_assignment', 'cyclic')}")
    print(f"   - Ensemble size: {getattr(args, 'ensemble_size', 1)}")
    print(f"   - Optimizer: Adam")
    print(f"   - Seed: {getattr(args, 'seed', 42)}")
    
    print(f"\n6. ACTIVE LEARNING:")
    print(f"   - Initial labeled fraction: {args.al_initial_frac:.1%}")
    print(f"   - Budget per round: {args.al_budget_per_round:.1%}")
    print(f"   - Number of AL rounds: {args.al_rounds}")
    print(f"   - Diversity filtering: {'Enabled' if args.al_diversity else 'Disabled'}")
    print(f"   - Selection metric: {args.al_metric}")
    print(f"   - Progressive labeling schedule:")
    cumulative = args.al_initial_frac
    print(f"     Round 0: {cumulative:.1%} (initial)")
    for i in range(1, args.al_rounds + 1):
        cumulative += args.al_budget_per_round
        print(f"     Round {i}: {cumulative:.1%} (+{args.al_budget_per_round:.1%})")
    
    print("\n" + "="*80)
    
    # Ask for user approval
    if args.auto:
        print("Auto-approval enabled. Proceeding with training plan.")
    else:
        approval = input("\nDo you want to proceed with this training plan? (yes/no): ").strip().lower()
        if approval not in ['yes', 'y']:
            print("Training cancelled by user.")
            return
    
    # Initialize active learning state
    active_learning_state = initialize_active_learning(
        partitions, args.num_clients, args.al_initial_frac, seed=args.seed
    )
    
    print(f"\nServer: Starting {args.num_rounds}-round simulation for {args.num_clients} clients.")
    print(f"Active Learning: Starting with {args.al_initial_frac:.1%} labeled data per client")
    
    # Run simulation rounds
    server_evaluation_history = []
    for r in range(args.num_rounds):
        print(f"\n=== Round {r} ===")
        
        # Collect client responses for this round
        client_responses = {}
        
        # Determine if this is an AL selection round
        al_round_interval = max(1, args.num_rounds // (args.al_rounds + 1))
        is_al_round = (r > 0 and r % al_round_interval == 0 and 
                      len(active_learning_state['al_history']) < args.al_rounds)
        
        # Simulate all clients for this round
        for client_id in range(args.num_clients):
            client_shipping_list, client_update_plan = get_client_plans_for_round(
                r, client_id, shipping_plans, update_plans
            )
            
            # Get the current labeled subset for this client
            client_labeled_indices = active_learning_state['labeled_indices'][str(client_id)]
            client_partition = torch.utils.data.Subset(
                partitions[client_id].dataset, client_labeled_indices
            )
            
            # Update AL state for this round
            active_learning_state['perform_selection'] = is_al_round
            active_learning_state['budget_frac'] = args.al_budget_per_round
            
            response = simulate_client_task(
                client_id, client_partition, args, r, 
                warehouse, client_shipping_list, client_update_plan,
                active_learning_state
            )
            client_responses[client_id] = response
        
        # Server collects responses and evaluates
        round_eval_results = collect_and_evaluate_round(r, args, warehouse, client_responses)
        server_evaluation_history.append(round_eval_results)
        
        # Handle active learning updates if this was an AL round
        if is_al_round and any('active_learning_info' in resp for resp in client_responses.values() if resp):
            print(f"\n--- Active Learning Round {len(active_learning_state['al_history']) + 1} ---")
            
            # Collect AL selections from all clients
            al_selections = {}
            total_selected = 0
            for client_id, response in client_responses.items():
                if response and 'active_learning_info' in response:
                    al_info = response['active_learning_info']
                    if al_info and 'selected_indices' in al_info:
                        al_selections[str(client_id)] = al_info['selected_indices']
                        total_selected += len(al_info['selected_indices'])
                        print(f"Client {client_id}: Selected {len(al_info['selected_indices'])} samples")
            
            # Update the active learning state
            update_active_learning_state(active_learning_state, al_selections)
            
            # Log AL progress
            total_labeled = sum(len(indices) for indices in active_learning_state['labeled_indices'].values())
            total_samples = sum(len(p) for p in partitions)
            print(f"Total labeled samples: {total_labeled}/{total_samples} ({100*total_labeled/total_samples:.1f}%)")
            print(f"New samples this round: {total_selected}")
        
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
        
        # Apply FedAvg if enabled
        if args.FedAvg:
            print(f"\nApplying FedAvg averaging at the end of round {r}...")
            
            # Load FBD settings to get model parts
            fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
            with open(fbd_settings_path, 'r') as f:
                fbd_settings = json.load(f)
            
            model_parts = fbd_settings.get('MODEL_PARTS', [])
            colors = ["M0", "M1", "M2", "M3", "M4", "M5"]  # All 6 colors
            
            # For each model part, average weights and optimizer states across all colors
            for part in model_parts:
                # Collect block IDs for this part across all colors
                block_ids_for_part = []
                for color in colors:
                    # Find block ID for this (part, color) combination
                    for block_id, block_info in warehouse.fbd_trace.items():
                        if block_info['model_part'] == part and block_info['color'] == color:
                            block_ids_for_part.append(block_id)
                            break
                
                if len(block_ids_for_part) != 6:
                    print(f"  Warning: Expected 6 blocks for part '{part}', found {len(block_ids_for_part)}")
                    continue
                
                # Average weights for this part
                all_weights = []
                for block_id in block_ids_for_part:
                    try:
                        weights = warehouse.retrieve_weights(block_id)
                        all_weights.append(weights)
                    except Exception as e:
                        print(f"  Warning: Could not retrieve weights for block {block_id}: {e}")
                
                if all_weights:
                    # Average the weights
                    averaged_weights = {}
                    for key in all_weights[0].keys():
                        tensors = [w[key] for w in all_weights]
                        if all(t.dtype == tensors[0].dtype for t in tensors) and tensors[0].dtype.is_floating_point:
                            averaged_weights[key] = torch.stack(tensors).mean(dim=0)
                        else:
                            averaged_weights[key] = tensors[0].clone()
                    
                    # Update all blocks with averaged weights
                    for block_id in block_ids_for_part:
                        warehouse.store_weights(block_id, averaged_weights)
                
                # Average optimizer states for this part
                all_optimizer_states = []
                for block_id in block_ids_for_part:
                    try:
                        optimizer_state = warehouse.retrieve_optimizer_state(block_id)
                        if optimizer_state:
                            all_optimizer_states.append(optimizer_state)
                    except Exception as e:
                        print(f"  Warning: Could not retrieve optimizer state for block {block_id}: {e}")
                
                if all_optimizer_states:
                    # Average the optimizer states
                    averaged_optimizer_state = {}
                    
                    # First, check if all optimizer states have the same structure
                    if 'state' in all_optimizer_states[0]:
                        averaged_state_dict = {}
                        # Get all parameter names
                        all_param_names = set()
                        for opt_state in all_optimizer_states:
                            if 'state' in opt_state:
                                all_param_names.update(opt_state['state'].keys())
                        
                        # Average each parameter's optimizer state
                        for param_name in all_param_names:
                            # Collect states for this parameter from all optimizer states
                            param_states = []
                            for opt_state in all_optimizer_states:
                                if 'state' in opt_state and param_name in opt_state['state']:
                                    param_states.append(opt_state['state'][param_name])
                            
                            if param_states:
                                # Average the state values
                                averaged_param_state = {}
                                for state_key in param_states[0].keys():
                                    state_values = [ps[state_key] for ps in param_states if state_key in ps]
                                    if state_values:
                                        if isinstance(state_values[0], torch.Tensor):
                                            if state_values[0].dtype.is_floating_point:
                                                averaged_param_state[state_key] = torch.stack(state_values).mean(dim=0)
                                            else:
                                                averaged_param_state[state_key] = state_values[0].clone()
                                        else:
                                            # For non-tensor values (like step count), use the mean
                                            averaged_param_state[state_key] = sum(state_values) / len(state_values)
                                
                                averaged_state_dict[param_name] = averaged_param_state
                        
                        averaged_optimizer_state['state'] = averaged_state_dict
                        
                        # Copy other optimizer settings from the first state
                        for key in all_optimizer_states[0].keys():
                            if key != 'state':
                                averaged_optimizer_state[key] = all_optimizer_states[0][key]
                    
                    # Update all blocks with averaged optimizer state
                    for block_id in block_ids_for_part:
                        warehouse.store_optimizer_state(block_id, averaged_optimizer_state)
                
                print(f"  Averaged {part} across all colors")
            
            print(f"FedAvg averaging completed for round {r}")
    
    # Move the temporary run folder to its final destination (like original)
    # try:
    #     shutil.move(temp_output_dir, final_output_dir)
    #     print(f"\nSimulation complete! Results saved to: {final_output_dir}")
    # except Exception as e:
    #     print(f"Error moving results to final destination: {e}")
    #     print(f"Results remain in: {temp_output_dir}")

if __name__ == "__main__":
    main()