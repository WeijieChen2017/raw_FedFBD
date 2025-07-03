import argparse
import os
import json
import time
import medmnist
from medmnist import INFO

from fbd_utils import load_config
from fbd_dataset import load_data, partition_data
from fbd_client_sim import simulate_client_task
from fbd_server_sim import (
    initialize_server_simulation, 
    load_simulation_plans, 
    prepare_test_dataset,
    collect_and_evaluate_round,
    get_client_plans_for_round
)


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
    partitions = partition_data(train_dataset, args.num_clients, args.iid)
    
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