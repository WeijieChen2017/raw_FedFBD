import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch training script for different alpha values")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., dermamnist, bloodmnist, etc.)")
    parser.add_argument("--fa", action="store_true", help="Enable FedAvg in the training command")
    args = parser.parse_args()

    alpha = [0.1, 0.25, 0.5, 1.0]

    for a in alpha:
        # Generate heterogeneous datasets
        command = f"python3 generate_hetero_datasets.py --dataset {args.dataset} --alpha {a}"
        print("running: ", command)
        os.system(command)
        
        # Run training with optional FedAvg
        fedavg_flag = "--FedAvg" if args.fa else ""
        command = f"python fbd_main_tau.py --experiment {args.dataset} --model_flag resnet18 --reg none {fedavg_flag} --save_affix _alpha_{a} --auto".strip()
        print("running: ", command)
        os.system(command)

if __name__ == "__main__":
    main()