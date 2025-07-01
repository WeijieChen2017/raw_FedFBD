import multiprocessing
import argparse
import os
from fbd_server import initialize_experiment, server_send_to_clients, server_collect_from_clients, end_experiment, evaluate_server_model
from fbd_client import client_task
from fbd_utils import load_config
from fbd_dataset import load_data, partition_data
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    output_dir = f"fbd_run/{args.experiment_name}_{args.model_flag}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = output_dir
    args.comm_dir = os.path.join(output_dir, "fbd_comm")
    
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
    #     evaluate_server_model(args, model_color, args.model_flag, args.experiment_name)

    # 2. Start client processes
    processes = []
    for i in range(args.num_clients):
        process = multiprocessing.Process(target=client_task, args=(i, partitions[i], args))
        processes.append(process)
        process.start()

    # 3. Run Rounds
    for r in range(args.num_rounds):
        # Server sends tasks to clients
        server_send_to_clients(r, args)
        
        # Server collects responses from clients
        server_collect_from_clients(r, args)

    # 4. Evaluate server models
    # print("Server: Evaluating final models from warehouse...")
    # for i in range(6):  # Evaluate models M0 to M5
    #     model_color = f"M{i}"
    #     evaluate_server_model(args, model_color, args.model_flag, args.experiment_name)

    # 5. End Experiment
    end_experiment(args)

    # Wait for all client processes to finish
    for process in processes:
        process.join()

    print("Framework execution complete.")

if __name__ == "__main__":
    main()









