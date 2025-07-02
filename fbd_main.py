import multiprocessing
import argparse
import os
import json
from fbd_server import initialize_experiment, server_send_to_clients, server_collect_from_clients, end_experiment, evaluate_server_model
from fbd_client import client_task
from fbd_utils import load_config
from fbd_dataset import load_data, partition_data
from fbd_plot import generate_plots
import time
import shutil

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*weights_only=False.*"
)

def main():
    parser = argparse.ArgumentParser(description="Federated Barter-based Data Exchange Framework")
    parser.add_argument("--experiment_name", type=str, default="bloodmnist", help="Name of the experiment.")
    parser.add_argument("--model_flag", type=str, default="resnet18", help="Model flag.")
    parser.add_argument("--comm_dir", type=str, default="fbd_comm", help="Directory for communication files.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to the model and weights cache.")
    parser.add_argument("--poll_interval", type=float, default=0.5, help="Polling interval in seconds for file checks.")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution.")
    args = parser.parse_args()

    # 0. Load config and merge into args
    config = load_config(args.experiment_name, args.model_flag)
    args_dict = vars(args)
    args_dict.update(vars(config))

    # Define temporary and final output directories. The experiment runs in a temporary location
    # and is moved to the final destination only upon successful completion.
    temp_output_dir = os.path.join(f"fbd_run", f"{args.experiment_name}_{args.model_flag}_{time.strftime('%Y%m%d_%H%M%S')}")
    final_output_dir = os.path.join(args.training_save_dir, f"{args.experiment_name}_{args.model_flag}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Clean up the temporary directory from any previous failed runs
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)

    args.output_dir = temp_output_dir
    
    # 1. Initialize Experiment
    initialize_experiment(args)
    
    # 1.A. Load and partition data
    print("Server: Loading and partitioning data...")
    train_dataset, _ = load_data(args)
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
    print("Generating plots for the experiment...")
    generate_plots(args.output_dir)

    print("Framework execution complete.")

    # 7. Move the temporary run folder to its final destination
    try:
        shutil.move(temp_output_dir, final_output_dir)
        print(f"Experiment results saved to: {final_output_dir}")
    except Exception as e:
        print(f"Error moving results to final destination: {e}")

if __name__ == "__main__":
    main()









