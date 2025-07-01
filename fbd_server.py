import os
import json
import random
import time
import shutil
import hashlib
import logging
import torch
from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import save_json, load_fbd_settings, FBDWarehouse, handle_dataset_cache, handle_weights_cache, setup_logger
from fbd_dataset import DATASET_SPECIFIC_RULES
from config.bloodmnist.generate_plans import main_generate_plans
import subprocess

import medmnist
from medmnist import INFO, Evaluator
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import PIL
import numpy as np
import argparse

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
    # Create output directory and log directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_dir = os.path.join(args.output_dir, "fbd_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    logger.info("Server: Initializing experiment...")

    # 1. Setup cache and communication directories
    if not args.cache_dir:
        args.cache_dir = os.path.join(os.getcwd(), "cache")
        logger.info(f"Cache directory not set, using default: {args.cache_dir}")
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    if getattr(args, 'remove_communication', False) and os.path.exists(args.comm_dir):
        logger.info(f"Server: Clearing communication directory at {args.comm_dir}")
        shutil.rmtree(args.comm_dir)
    
    if not os.path.exists(args.comm_dir):
        os.makedirs(args.comm_dir)
        logger.info(f"Server: Created communication directory at {args.comm_dir}")

    # 2. Handle dataset caching
    logger.info("Server: Checking dataset cache...")
    handle_dataset_cache(args.experiment_name, args.cache_dir)

    # 3. Prepare and cache the initial model
    logger.info("Server: Preparing initial model...")
    prepare_initial_model(args)
    
    # 4. Generate FBD plans
    logger.info("Server: Generating FBD plans...")
    try:
        main_generate_plans()
        logger.info("Server: FBD plans generated successfully.")
    except Exception as e:
        logger.error("Server: Failed to generate FBD plans.", exc_info=True)
        raise e
        
    # 5. Initialize FBD Warehouse
    logger.info("Server: Initializing FBD Warehouse...")
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
    
    # Create a serializable copy of the configuration by removing complex objects
    args_to_save = vars(args).copy()
    args_to_save.pop('warehouse', None)
    args_to_save.pop('test_dataset', None)
    save_json(args_to_save, config_path)
    
    # 7. Prepare test dataset for evaluations
    logger.info("Server: Preparing test dataset for evaluations...")
    info = INFO[args.experiment_name]
    DataClass = getattr(medmnist, info['python_class'])
    dataset_rules = DATASET_SPECIFIC_RULES.get(args.experiment_name, {})
    as_rgb = dataset_rules.get("as_rgb", False)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    args.test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, size=args.size)
    logger.info("Server: Test dataset prepared.")
    
    logger.info("Server: Experiment initialization complete.")

def server_send_to_clients(r, args):
    """Server-side logic to create files for the current round."""
    log_dir = os.path.join(args.output_dir, "fbd_log")
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    logger.info(f"Server: --- Round {r} ---")

    # 1. Use the global warehouse
    warehouse = args.warehouse

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
            logger.info(f"Server: No shipping plan for client {i} in round {r}. Skipping.")
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
        logger.info(f"Server: Sent goods for round {r} to client {i}")

    logger.info(f"Server: Sent goods for round {r} to {args.num_clients} clients.")

def server_collect_from_clients(r, args):
    """Server-side logic to collect responses from clients for the current round."""
    log_dir = os.path.join(args.output_dir, "fbd_log")
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    logger.info(f"Server: Collecting responses for round {r}...")
    
    # 1. Use the global warehouse
    warehouse = args.warehouse
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")

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
                
                logger.info(f"Server: Received update from client {i} for round {r}, loss: {loss:.4f}")

                if updated_weights:
                    logger.info(f"Server: Received {len(updated_weights)} weight blocks from client {i}: {list(updated_weights.keys())}")
                    warehouse.store_weights_batch(updated_weights)
                    # Save the warehouse so the evaluation function can load the latest state
                    warehouse.save_warehouse(warehouse_path)
                    logger.info(f"Server: Warehouse updated by client {i} and saved.")
                    print(f"Server: Stored {len(updated_weights)} weight blocks from client {i}")
                else:
                    logger.warning(f"Server: Client {i} sent no updated weights!")
                    print(f"Server: WARNING - Client {i} sent no updated weights!")
                
                # Mark as collected by renaming or deleting the file to avoid recounting
                os.remove(filepath) 
                collected_clients += 1
        time.sleep(args.poll_interval)
    
    # After collecting from all clients, print summary
    avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
    logger.info(f"Server: All responses for round {r} collected. Average loss: {avg_loss:.4f}")
    
    # Evaluate all models once at the end of the round
    logger.info(f"Server: Evaluating all models at end of round {r}...")
    for model_idx in range(6):  # Evaluate models M0 to M5
        model_color = f"M{model_idx}"
        evaluate_server_model(args, model_color, args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    evaluate_server_model(args, "averaging", args.model_flag, args.experiment_name, args.test_dataset, warehouse)

def end_experiment(args):
    """After all rounds, send a shutdown signal to clients."""
    log_dir = os.path.join(args.output_dir, "fbd_log")
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    logger.info("Server: All rounds complete. Sending shutdown signal.")
    for i in range(args.num_clients):
        filepath = os.path.join(args.comm_dir, f"last_round_client_{i}.json")
        with open(filepath, 'w') as f:
            json.dump({"secret": -1}, f) 

def evaluate_server_model(args, model_color, model_name, dataset, test_dataset, warehouse):
    """
    Evaluates a model specified by its color (e.g., M0-M5) from the warehouse.
    This version integrates the evaluation logic directly, without calling an external script.

    Args:
        args: Experiment configuration arguments.
        model_color (str): The color of the model to evaluate (e.g., 'M0').
        model_name (str): The architecture of the model (e.g., 'resnet18').
        dataset (str): The dataset to evaluate on (e.g., 'bloodmnist').
        test_dataset (data.Dataset): The test dataset for evaluation.
        warehouse (FBDWarehouse): The global warehouse instance.
    """
    log_dir = os.path.join(args.output_dir, "fbd_log")
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    logger.info(f"Starting evaluation for model {model_color} ({model_name}) on {dataset}...")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Use the provided warehouse object to reconstruct the model
    if model_color == "averaging":
        logger.info("Creating and evaluating an averaged model from M0-M5.")
        all_model_weights = [warehouse.get_model_weights(f"M{i}") for i in range(6)]
        
        if not all(w is not None and len(w) > 0 for w in all_model_weights):
            logger.error("Could not retrieve weights for all models M0-M5. Skipping averaging.")
            return

        model_weights = {}
        param_keys = all_model_weights[0].keys()
        for key in param_keys:
            if all_model_weights[0][key].is_floating_point():
                model_weights[key] = torch.stack([weights[key] for weights in all_model_weights]).mean(dim=0)
            else:
                # For non-floating point tensors (e.g., num_batches_tracked in BatchNorm),
                # just take the value from the first model.
                model_weights[key] = all_model_weights[0][key]
        
        logger.info("Finished averaging model weights.")
    else:
        model_weights = warehouse.get_model_weights(model_color)
    
    logger.info(f"Loaded {len(model_weights)} parameters for model {model_color} from warehouse")
    # Check if weights look reasonable (not all zeros or same values)
    sample_param = list(model_weights.values())[0] if model_weights else None
    if sample_param is not None:
        logger.info(f"Sample parameter '{list(model_weights.keys())[0]}' stats: mean={sample_param.mean().item():.6f}, std={sample_param.std().item():.6f}")

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
    
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Setup for evaluation
    test_evaluator = Evaluator(dataset, 'test', size=args.size)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()

    # 4. Run evaluation
    test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)

    logger.info(f"Evaluation complete for model {model_color} on {dataset}.")
    logger.info(f"  └─ Test Loss: {test_metrics[0]:.5f}, Test AUC: {test_metrics[1]:.5f}, Test Acc: {test_metrics[2]:.5f}")
    print(f"Server - {model_color}: Test Loss = {test_metrics[0]:.5f}, Test AUC = {test_metrics[1]:.5f}, Test Acc = {test_metrics[2]:.5f}")

    # 5. Save results
    eval_results_dir = os.path.join(args.output_dir, "eval_results", f"{dataset}/{model_name}/{model_color}")
    os.makedirs(eval_results_dir, exist_ok=True)
    
    metrics_dict = {
        "model_color": model_color, "model_name": model_name, "dataset": dataset,
        "test_loss": test_metrics[0], "test_auc": test_metrics[1], "test_acc": test_metrics[2]
    }
    
    save_name = os.path.join(eval_results_dir, f"eval_metrics.json")
    with open(save_name, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Metrics saved to {save_name}") 

def main_server(args):
    """Main server process."""
    log_dir = os.path.join(args.output_dir, "fbd_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "server.log")
    logger = setup_logger("Server", log_file)
    
    logger.info("FBD Server starting...")
    logger.info(f"Arguments: {vars(args)}")

    # Setup directories
    comm_dir = getattr(args, 'comm_dir', 'fbd_comm')
    if not os.path.exists(comm_dir):
        os.makedirs(comm_dir)
        logger.info(f"Created communication directory: {comm_dir}")

    # Load all necessary FBD configurations
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    shipping_plan_path = os.path.join("config", args.experiment_name, "shipping_plan.json")
    update_plan_path = os.path.join("config", args.experiment_name, "update_plan.json")
    
    try:
        fbd_trace, fbd_info, model_parts = load_fbd_settings(fbd_settings_path)
        with open(shipping_plan_path, 'r') as f:
            shipping_plan = json.load(f)
        with open(update_plan_path, 'r') as f:
            update_plan = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Failed to load required plan file: {e}. Please run generate_plans.py.")
        return

    logger.info(f"Loaded FBD config for {args.experiment_name}")
    logger.info(f"Total rounds in plan: {len(shipping_plan)}")

    # Initialize the global model
    global_model = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=args.norm,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        use_pretrained=args.use_pretrained,
        logger=logger
    )
    logger.info(f"Initialized global model: {args.model_flag}")

    # Main server loop
    for round_num_str in sorted(shipping_plan.keys(), key=int):
        round_num = int(round_num_str)
        logger.info(f"\n----- Round {round_num} -----")
        print(f"\n{'='*20} Round {round_num} {'='*20}", flush=True)
        
        clients_in_round = shipping_plan[round_num_str]
        active_clients = list(clients_in_round.keys())
        logger.info(f"Active clients for round {round_num}: {active_clients}")

        # Distribute goods to each active client
        for client_id in active_clients:
            client_shipping_list = clients_in_round.get(client_id, [])
            if not client_shipping_list:
                logger.warning(f"No shipping list for client {client_id} in round {round_num}.")
                continue

            model_weights = {}
            for part_name in client_shipping_list:
                # This logic needs to be robust. Assuming fbd_trace maps a shipping part name to a model part prefix
                model_part_prefix = fbd_trace[part_name]['model_part']
                for param_name, param in global_model.state_dict().items():
                    if param_name.startswith(model_part_prefix):
                        model_weights[param_name] = param

            data_packet = {
                "shipping_list": client_shipping_list,
                "update_plan": update_plan[round_num_str].get(client_id, {}),
                "model_weights": model_weights,
                "round": round_num
            }
            
            goods_filepath = os.path.join(comm_dir, f"goods_round_{round_num}_client_{client_id}.pth")
            torch.save(data_packet, goods_filepath)
            logger.info(f"  > Dispatched goods to client {client_id}")

        # Wait for and collect responses from clients
        collected_updates = {}
        client_stats = {}
        
        start_time = time.time()
        timeout = getattr(args, 'poll_timeout', 300)
        while len(collected_updates) < len(active_clients):
            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for client responses in round {round_num}.")
                break

            for client_id in active_clients:
                if client_id not in collected_updates:
                    response_filepath = os.path.join(comm_dir, f"response_round_{round_num}_client_{client_id}.pth")
                    if os.path.exists(response_filepath):
                        try:
                            response_data = torch.load(response_filepath)
                            
                            collected_updates[client_id] = response_data.get("updated_weights", {})
                            client_stats[client_id] = response_data.get("dataset_stats", {})
                            
                            loss = response_data.get('train_loss', 'N/A')
                            loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss
                            logger.info(f"  < Received response from client {client_id} (Loss: {loss_str})")
                            
                            if getattr(args, 'remove_communication', False):
                                os.remove(response_filepath)
                        except Exception as e:
                            logger.error(f"Error processing response from client {client_id}: {e}")
                            collected_updates[client_id] = {}
            time.sleep(args.poll_interval)
        
        logger.info(f"All client responses for round {round_num} received.")

        # Aggregate weights
        aggregated_weights = {}
        for client_id, weights in collected_updates.items():
            if weights:
                aggregated_weights.update(weights)

        # Update global model
        if aggregated_weights:
            global_model.load_state_dict(aggregated_weights, strict=False)
            logger.info("Global model updated with aggregated weights.")
        else:
            logger.warning("No weights were aggregated in this round.")

    # Signal clients to shut down
    num_clients = len(fbd_info.get("clients", []))
    for i in range(num_clients):
        client_id_str = str(i) # Assuming client IDs are '0', '1', ...
        shutdown_filepath = os.path.join(comm_dir, f"last_round_client_{client_id_str}.json")
        with open(shutdown_filepath, 'w') as f:
            json.dump({"secret": -1}, f)
        logger.info(f"Sent shutdown signal to client {client_id_str}.")

    logger.info("FBD Server has completed all rounds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Block Design Server")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--model_flag", type=str, required=True, help="Model architecture")
    parser.add_argument("--norm", type=str, required=True, help="Normalization type")
    parser.add_argument("--in_channels", type=int, required=True, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
    parser.add_argument("--use_pretrained", action='store_true', help="Use pretrained weights")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--comm_dir", type=str, default="fbd_comm", help="Communication directory")
    parser.add_argument("--remove_communication", action='store_true', help="Remove communication files after processing")
    parser.add_argument("--poll_interval", type=float, default=1.0, help="Poll interval for client responses")

    args = parser.parse_args()

    main_server(args) 