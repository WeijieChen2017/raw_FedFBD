import os
import json
import random
import time
import shutil
import hashlib
import logging
import torch
from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import save_json, load_fbd_settings, FBDWarehouse, handle_dataset_cache, handle_weights_cache
from config.bloodmnist.generate_plans import main_generate_plans
import subprocess

import medmnist
from medmnist import INFO, Evaluator
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import PIL
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_SPECIFIC_RULES = {
    "bloodmnist": {"as_rgb": True},
    "breastmnist": {"as_rgb": True},
    "octmnist": {"as_rgb": True},
    "organcmnist": {"as_rgb": True},
    "tissuemnist": {"as_rgb": True},
    "pneumoniamnist": {"as_rgb": True},
    "chestmnist": {"as_rgb": True},
    "organamnist": {"as_rgb": True},
    "organsmnist": {"as_rgb": True},
}

def _test_model(model, evaluator, data_loader, task, criterion, device):
    """Core testing logic, adapted from train_and_eval_pytorch.py"""
    model.eval()
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

    y_score = y_score.detach().cpu().numpy()
    auc, acc = evaluator.evaluate(y_score, None, None) # No save folder or name tag needed for this server eval
    test_loss = sum(total_loss) / len(total_loss)
    return [test_loss, auc, acc]

def prepare_initial_model(args):
    """
    Prepares the initial model for the experiment.
    It downloads pretrained ImageNet weights, adapts them for the target dataset,
    and saves the initial model checkpoint to the cache directory.
    """
    model_path = os.path.join(args.cache_dir, f"initial_{args.model_flag}.pth")
    
    if os.path.exists(model_path):
        logging.info(f"Initial model found in cache: {model_path}")
        return

    logging.info(f"Preparing initial model '{args.model_flag}' with ImageNet weights.")
    
    # Handle weights caching before model loading
    sync_weights_back = handle_weights_cache(args.model_flag, args.cache_dir)

    model = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=args.norm,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        use_pretrained=True
    )
    
    # After model loading, cache the downloaded weights if necessary
    if sync_weights_back:
        sync_weights_back()

    # Save the initial model state
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved initial model to {model_path}")

def initialize_experiment(args):
    """
    Initializes the experiment by setting up directories, caching the dataset,
    and preparing the initial model.
    """
    logging.info("Server: Initializing experiment...")

    # 1. Setup cache and communication directories
    if not args.cache_dir:
        args.cache_dir = os.path.join(os.getcwd(), "cache")
        logging.info(f"Cache directory not set, using default: {args.cache_dir}")
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    if getattr(args, 'remove_communication', False) and os.path.exists(args.comm_dir):
        logging.info(f"Server: Clearing communication directory at {args.comm_dir}")
        shutil.rmtree(args.comm_dir)
    
    if not os.path.exists(args.comm_dir):
        os.makedirs(args.comm_dir)
        logging.info(f"Server: Created communication directory at {args.comm_dir}")

    # 2. Handle dataset caching
    logging.info("Server: Checking dataset cache...")
    handle_dataset_cache(args.experiment_name, args.cache_dir)

    # 3. Prepare and cache the initial model
    logging.info("Server: Preparing initial model...")
    prepare_initial_model(args)
    
    # 4. Generate FBD plans
    logging.info("Server: Generating FBD plans...")
    try:
        main_generate_plans()
        logging.info("Server: FBD plans generated successfully.")
    except Exception as e:
        logging.error("Server: Failed to generate FBD plans.", exc_info=True)
        raise e
        
    # 5. Initialize FBD Warehouse
    logging.info("Server: Initializing FBD Warehouse...")
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)

    model_template = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=args.norm,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        use_pretrained=True
    )
    initial_model_path = os.path.join(args.cache_dir, f"initial_{args.model_flag}.pth")
    model_template.load_state_dict(torch.load(initial_model_path))

    warehouse = FBDWarehouse(
        fbd_trace=fbd_trace,
        model_template=model_template,
        log_file_path=os.path.join(args.comm_dir, "warehouse.log")
    )
    
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")
    warehouse.save_warehouse(warehouse_path)
    logging.info(f"Server: FBD Warehouse initialized and saved to {warehouse_path}")

    # 6. Save training configuration
    config_path = os.path.join(args.comm_dir, "train_config.json")
    logging.info(f"Server: Saving training configuration to {config_path}")
    save_json(vars(args), config_path)
    
    logging.info("Server: Experiment initialization complete.")

def server_send_to_clients(r, args):
    """Server-side logic to create files for the current round."""
    print(f"Server: --- Round {r} ---")

    # 1. Load the warehouse
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)
    warehouse = FBDWarehouse(fbd_trace=fbd_trace)
    warehouse.load_warehouse(warehouse_path)

    # 2. Load the shipping and update plans
    shipping_plan_path = os.path.join("config", args.experiment_name, "shipping_plan.json")
    with open(shipping_plan_path, 'r') as f:
        shipping_plan = json.load(f)
    
    update_plan_path = os.path.join("config", args.experiment_name, "update_plan.json")
    with open(update_plan_path, 'r') as f:
        update_plan = json.load(f)

    # 3. Distribute goods to each client for the current round
    round_shipping_plan = shipping_plan.get(str(r + 1), {})
    round_update_plan = update_plan.get(str(r + 1), {})

    for i in range(args.num_clients):
        client_shipping_list = round_shipping_plan.get(str(i), [])
        client_update_plan = round_update_plan.get(str(i), {})
        
        if not client_shipping_list:
            print(f"Server: No shipping plan for client {i} in round {r}. Skipping.")
            continue

        # Get model weights from the warehouse
        model_weights = warehouse.get_shipping_weights(client_shipping_list)
        
        # Prepare data packet
        data_to_send = {
            "shipping_list": client_shipping_list,
            "update_plan": client_update_plan,
            "model_weights": model_weights,
            "round": r
        }
        
        # Save the data packet for the client atomically
        filepath = os.path.join(args.comm_dir, f"goods_round_{r}_client_{i}.pth")
        temp_filepath = f"{filepath}.tmp"
        torch.save(data_to_send, temp_filepath)
        os.rename(temp_filepath, filepath)

    print(f"Server: Sent goods for round {r} to {args.num_clients} clients.")

def server_collect_from_clients(r, args):
    """Server-side logic to collect responses from clients for the current round."""
    print(f"Server: Collecting responses for round {r}...")
    
    # 1. Load the warehouse
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)
    warehouse = FBDWarehouse(fbd_trace=fbd_trace)
    warehouse.load_warehouse(warehouse_path)

    collected_clients = 0
    round_losses = []
    
    while collected_clients < args.num_clients:
        # This is a simple polling mechanism. In a real scenario, you might want a more robust solution.
        for i in range(args.num_clients):
            filepath = os.path.join(args.comm_dir, f"response_round_{r}_client_{i}.pth")
            if os.path.exists(filepath):
                data = torch.load(filepath)
                
                # Process the client's update
                loss = data.get("train_loss")
                updated_weights = data.get("updated_weights")
                round_losses.append(loss)
                
                print(f"Server: Received update from client {i} for round {r}, loss: {loss:.4f}")

                if updated_weights:
                    warehouse.store_weights_batch(updated_weights)
                    # Save the warehouse so the evaluation function can load the latest state
                    warehouse.save_warehouse(warehouse_path)
                    print(f"Server: Warehouse updated by client {i} and saved.")

                    # Evaluate all models after this client's update
                    print(f"Server: Evaluating all models after update from client {i}...")
                    for model_idx in range(6):  # Evaluate models M0 to M5
                        model_color = f"M{model_idx}"
                        evaluate_server_model(args, model_color, args.model_flag, args.experiment_name)
                
                # Mark as collected by renaming or deleting the file to avoid recounting
                os.remove(filepath) 
                collected_clients += 1
        time.sleep(args.poll_interval)
    
    # After collecting from all clients, print summary
    avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
    print(f"Server: All responses for round {r} collected. Average loss: {avg_loss:.4f}")

def end_experiment(args):
    """After all rounds, send a shutdown signal to clients."""
    print("Server: All rounds complete. Sending shutdown signal.")
    for i in range(args.num_clients):
        filepath = os.path.join(args.comm_dir, f"last_round_client_{i}.json")
        with open(filepath, 'w') as f:
            json.dump({"secret": -1}, f) 

def evaluate_server_model(args, model_color, model_name, dataset):
    """
    Evaluates a model specified by its color (e.g., M0-M5) from the warehouse.
    This version integrates the evaluation logic directly, without calling an external script.

    Args:
        args: Experiment configuration arguments.
        model_color (str): The color of the model to evaluate (e.g., 'M0').
        model_name (str): The architecture of the model (e.g., 'resnet18').
        dataset (str): The dataset to evaluate on (e.g., 'bloodmnist').
    """
    logging.info(f"Starting evaluation for model {model_color} ({model_name}) on {dataset}...")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load the warehouse and reconstruct model
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")
    if not os.path.exists(warehouse_path):
        logging.error(f"Warehouse not found at {warehouse_path}.")
        return

    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)
    warehouse = FBDWarehouse(fbd_trace=fbd_trace)
    warehouse.load_warehouse(warehouse_path)
    model_weights = warehouse.get_model_weights(model_color)

    model = get_pretrained_fbd_model(
        architecture=model_name,
        norm=args.norm, in_channels=args.in_channels, num_classes=args.num_classes,
        use_pretrained=False
    )
    model.load_state_dict(model_weights)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 2. Prepare dataset and dataloader
    info = INFO[dataset]
    task = info['task']
    DataClass = getattr(medmnist, info['python_class'])
    
    dataset_rules = DATASET_SPECIFIC_RULES.get(dataset, {})
    as_rgb = dataset_rules.get("as_rgb", False)
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, size=args.size)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Setup for evaluation
    test_evaluator = Evaluator(dataset, 'test', size=args.size)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()

    # 4. Run evaluation
    test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)

    logging.info(f"Evaluation complete for model {model_color} on {dataset}.")
    logging.info(f"  └─ Test Loss: {test_metrics[0]:.5f}, Test AUC: {test_metrics[1]:.5f}, Test Acc: {test_metrics[2]:.5f}")

    # 5. Save results
    output_dir = os.path.join("eval_results", f"{dataset}/{model_name}/{model_color}")
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_dict = {
        "model_color": model_color, "model_name": model_name, "dataset": dataset,
        "test_loss": test_metrics[0], "test_auc": test_metrics[1], "test_acc": test_metrics[2]
    }
    
    save_name = os.path.join(output_dir, f"eval_metrics.json")
    with open(save_name, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"Metrics saved to {save_name}") 