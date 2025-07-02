import os
import json
import random
import time
import shutil
import hashlib
import logging
import torch
from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import save_json, load_fbd_settings, FBDWarehouse, handle_dataset_cache, handle_weights_cache, setup_logger, save_optimizer_state_by_block, build_optimizer_with_state
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
import importlib.util
from collections import Counter

def _get_scores(model, data_loader, task, device):
    """Runs the model on the data and returns the raw scores."""
    model.eval()
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                m = nn.Sigmoid()
                outputs = m(outputs)
            else:
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
            
            y_score = torch.cat((y_score, outputs), 0)
            
    return y_score.detach().cpu().numpy()

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
        
    # 4.5 Archive configuration files
    logger.info("Server: Archiving configuration files...")
    config_archive_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_archive_dir, exist_ok=True)
    
    source_config_dir = os.path.join("config", args.experiment_name)
    files_to_copy = [
        "config.json",
        "fbd_settings.py",
        "shipping_plan.json",
        "update_plan.json",
        "request_plan.json"
    ]
    
    for file_name in files_to_copy:
        source_path = os.path.join(source_config_dir, file_name)
        destination_path = os.path.join(config_archive_dir, file_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            logger.info(f"Copied {file_name} to {config_archive_dir}")
        else:
            logger.warning(f"Config file not found, skipping: {source_path}")

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
        
        # Get optimizer states from the warehouse
        optimizer_states = warehouse.get_shipping_optimizer_states(client_shipping_list)

        # Prepare data packet
        data_to_send = {
            "shipping_list": client_shipping_list,
            "update_plan": client_update_plan,
            "model_weights": model_weights,
            "optimizer_states": optimizer_states,
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
                updated_optimizer_states = data.get("updated_optimizer_states")
                round_losses.append(loss)
                
                logger.info(f"Server: Received update from client {i} for round {r}, loss: {loss:.4f}")

                if updated_weights:
                    logger.info(f"Server: Received {len(updated_weights)} weight blocks from client {i}: {list(updated_weights.keys())}")
                    warehouse.store_weights_batch(updated_weights)
                    print(f"Server: Stored {len(updated_weights)} weight blocks from client {i}")
                else:
                    logger.warning(f"Server: Client {i} sent no updated weights!")
                    print(f"Server: WARNING - Client {i} sent no updated weights!")
                
                if updated_optimizer_states:
                    logger.info(f"Server: Received {len(updated_optimizer_states)} optimizer states from client {i}.")
                    warehouse.store_optimizer_state_batch(updated_optimizer_states)
                
                if updated_weights or updated_optimizer_states:
                    # Save the warehouse so the evaluation function can load the latest state
                    warehouse.save_warehouse(warehouse_path)
                    logger.info(f"Server: Warehouse updated by client {i} and saved.")
                
                # Mark as collected by renaming or deleting the file to avoid recounting
                os.remove(filepath) 
                collected_clients += 1
        time.sleep(args.poll_interval)
    
    # After collecting from all clients, print summary
    avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
    logger.info(f"Server: All responses for round {r} collected. Average loss: {avg_loss:.4f}")
    
    # Evaluate all models once at the end of the round
    logger.info(f"Server: Evaluating all models at end of round {r}...")
    round_eval_results = {'round': r}
    for model_idx in range(6):  # Evaluate models M0 to M5
        model_color = f"M{model_idx}"
        metrics = evaluate_server_model(args, model_color, args.model_flag, args.experiment_name, args.test_dataset, warehouse)
        round_eval_results[model_color] = metrics
    
    avg_metrics = evaluate_server_model(args, "averaging", args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    round_eval_results["averaging"] = avg_metrics
    
    ensemble_metrics = evaluate_server_model(args, "ensemble", args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    round_eval_results["ensemble"] = ensemble_metrics
    
    return round_eval_results

def final_ensemble_test(args):
    """
    Assembles a final ensemble of models, evaluates them on the test set,
    and computes correctness (z), confidence (c), and standard deviation (s)
    metrics based on a majority vote.
    """
    log_dir = os.path.join(args.output_dir, "fbd_log")
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    logger.info("Server: Starting final ensemble test.")

    # 1. Load settings for the final test
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    try:
        spec = importlib.util.spec_from_file_location("fbd_settings", fbd_settings_path)
        fbd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fbd_module)
        final_test_colors = getattr(fbd_module, 'FINAL_TEST_COLORS', [])
        final_test_batch_size = getattr(fbd_module, 'FINAL_TEST_SIZE', 128)
        if not final_test_colors:
            logger.warning("FINAL_TEST_COLORS not found in fbd_settings.py. Skipping final test.")
            return None, None, None
    except FileNotFoundError:
        logger.warning(f"fbd_settings.py not found at {fbd_settings_path}. Skipping final test.")
        return None, None, None
    
    # 2. Prepare data and models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    info = INFO[args.experiment_name]
    task = info['task']
    test_loader = data.DataLoader(dataset=args.test_dataset, batch_size=final_test_batch_size, shuffle=False)
    warehouse = args.warehouse

    # 3. Get predictions from each model in the ensemble
    logger.info(f"Gathering predictions from {len(final_test_colors)} models: {final_test_colors}")
    all_model_scores = []
    for model_color in final_test_colors:
        model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        ).to(device)
        
        model_weights = warehouse.get_model_weights(model_color)
        if not model_weights:
            logger.warning(f"No weights found in warehouse for model {model_color}. Skipping.")
            continue
            
        model.load_state_dict(model_weights)
        scores = _get_scores(model, test_loader, task, device)
        all_model_scores.append(scores)

    if not all_model_scores:
        logger.error("Failed to gather any model predictions. Aborting final test.")
        return None, None, None

    # 4. Calculate final metrics via majority vote
    votes = np.stack([np.argmax(scores, axis=1) for scores in all_model_scores], axis=1)
    
    true_labels = args.test_dataset.labels.squeeze()
    num_samples = len(true_labels)
    num_models = votes.shape[1]
    num_classes = args.num_classes

    z_scores, c_scores, s_scores = [], [], []

    logger.info("Calculating z, c, and s scores for each test sample...")
    for i in range(num_samples):
        sample_votes = votes[i, :]
        vote_counts = Counter(sample_votes)
        
        majority_class, majority_count = vote_counts.most_common(1)[0]
        
        z = 1 if majority_class == true_labels[i] else 0
        c = majority_count / num_models
        
        vote_distribution = np.zeros(num_classes)
        for class_idx, count in vote_counts.items():
            vote_distribution[class_idx] = count
        s = np.std(vote_distribution)
        
        z_scores.append(z)
        c_scores.append(c)
        s_scores.append(s)

    # 5. Save results
    results = {
        'z_correctness': z_scores,
        'c_confidence': c_scores,
        's_std_dev': s_scores,
        'mean_correctness': np.mean(z_scores),
        'mean_confidence': np.mean(c_scores),
        'mean_std_dev': np.mean(s_scores)
    }
    
    results_path = os.path.join(args.output_dir, "final_test_metrics.json")
    save_json(results, results_path)
    logger.info(f"Final test metrics saved to {results_path}")

    return z_scores, c_scores, s_scores

def end_experiment(args):
    """After all rounds, run final tests and send a shutdown signal to clients."""
    log_dir = os.path.join(args.output_dir, "fbd_log")
    logger = setup_logger("Server", os.path.join(log_dir, "server.log"))
    
    # Run final ensemble test before shutting down
    final_ensemble_test(args)

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
    random.seed(args.seed)

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

    elif model_color == "ensemble":
        logger.info(f"Starting block-wise ensemble evaluation for {args.num_ensemble} models...")
        
        # Load settings from the experiment's FBD config file
        fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
        spec = importlib.util.spec_from_file_location("fbd_settings", fbd_settings_path)
        fbd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fbd_module)
        
        ensemble_colors_pool = getattr(fbd_module, 'ENSEMBLE_COLORS', [])
        model_parts_pool = getattr(fbd_module, 'MODEL_PARTS', [])
        fbd_trace = getattr(fbd_module, 'FBD_TRACE', {})

        if not all([ensemble_colors_pool, model_parts_pool, fbd_trace]):
            logger.error("Ensemble settings (ENSEMBLE_COLORS, MODEL_PARTS, FBD_TRACE) are missing or empty. Skipping evaluation.")
            return

        # Create a reverse map for easy lookup: (part, color) -> block_id
        part_color_to_block_id = {
            (info['model_part'], info['color']): block_id
            for block_id, info in fbd_trace.items()
        }

        # Prepare for evaluation
        info = INFO[dataset]
        task = info['task']
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        test_evaluator = Evaluator(dataset, 'test', size=args.size)
        
        all_y_scores = []
        logger.info(f"Generating {args.num_ensemble} hybrid models for the ensemble...")

        for i in range(args.num_ensemble):
            # 1. Create a random hybrid model configuration
            hybrid_config = {part: random.choice(ensemble_colors_pool) for part in model_parts_pool}
            logger.info(f"  Hybrid Model {i+1}/{args.num_ensemble} config: {hybrid_config}")
            
            # 2. Assemble weights for the hybrid model
            hybrid_weights = {}
            is_valid_config = True
            for part, color in hybrid_config.items():
                block_id = part_color_to_block_id.get((part, color))
                if block_id is None:
                    logger.error(f"Could not find block_id for part '{part}' and color '{color}'. Skipping this hybrid model.")
                    is_valid_config = False
                    break
                
                block_weights = warehouse.retrieve_weights(block_id)
                if not block_weights:
                    logger.error(f"Could not retrieve weights for block '{block_id}'. Skipping this hybrid model.")
                    is_valid_config = False
                    break
                
                hybrid_weights.update(block_weights)
            
            if not is_valid_config:
                continue

            # 3. Get predictions from the hybrid model
            model = get_pretrained_fbd_model(
                architecture=model_name,
                norm=args.norm, in_channels=args.in_channels, num_classes=args.num_classes,
                use_pretrained=False
            )
            model.load_state_dict(hybrid_weights)
            model.to(device)
            
            y_score = _get_scores(model, test_loader, task, device)
            all_y_scores.append(y_score)

        if not all_y_scores:
            logger.error("No valid hybrid model predictions were generated. Aborting ensemble evaluation.")
            return

        # Calculate Majority Vote Ratio
        true_labels = test_dataset.labels.flatten()
        member_predictions = np.argmax(np.array(all_y_scores), axis=2) # Shape: (num_ensemble, num_samples)
        
        # Calculate Majority Vote accuracy and Mean Member Accuracy
        num_correct_majority_vote = 0
        # The comparison below is likely buggy, but we calculate it for diagnostics.
        num_correct_individual_votes = np.sum(member_predictions == true_labels.reshape(1, -1))

        # Iterate through each sample to find the majority vote
        for i in range(member_predictions.shape[1]): # Iterate over samples
            sample_votes = member_predictions[:, i] # All votes for the i-th sample
            vote_counts = Counter(sample_votes)
            majority_class, _ = vote_counts.most_common(1)[0]
            
            if majority_class == true_labels[i]:
                num_correct_majority_vote += 1
        
        num_samples = len(true_labels)
        majority_vote_accuracy = num_correct_majority_vote / num_samples if num_samples > 0 else 0
        
        total_individual_votes = member_predictions.size
        mean_member_accuracy = num_correct_individual_votes / total_individual_votes if total_individual_votes > 0 else 0

        logger.info(f"Ensemble Majority Vote Accuracy: {majority_vote_accuracy:.5f}")
        logger.info(f"Ensemble Mean Member Accuracy: {mean_member_accuracy:.5f} ({num_correct_individual_votes}/{total_individual_votes} correct individual votes)")
        print(f"Server - Ensemble: Majority Vote Acc = {majority_vote_accuracy:.5f}, Mean Member Acc = {mean_member_accuracy:.5f}")

        # 4. Average the scores and evaluate
        avg_y_score = np.mean(all_y_scores, axis=0)
        
        auc, acc = test_evaluator.evaluate(avg_y_score, None, None)
        test_loss = float('nan') # Loss cannot be computed from scores alone
        
        logger.info(f"Ensemble evaluation complete.")
        logger.info(f"  └─ Test Loss: {test_loss:.5f}, Test AUC (from avg scores): {auc:.5f}, Test Acc (from avg scores): {acc:.5f}")
        print(f"Server - Ensemble: Test Loss = {test_loss:.5f}, Test AUC = {auc:.5f}, Test Acc = {acc:.5f}")
        
        # Return the metrics instead of saving them
        return {
            "model_color": model_color, "model_name": model_name, "dataset": dataset,
            "test_loss": test_loss, "test_auc": auc, "test_acc": acc,
            "majority_vote_accuracy": majority_vote_accuracy,
            "mean_member_accuracy": mean_member_accuracy
        }

    else: # This block handles single model evaluation ("M0", "M1", etc.)
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

    # Return the metrics instead of saving them
    return {
        "model_color": model_color, "model_name": model_name, "dataset": dataset,
        "test_loss": test_metrics[0], "test_auc": test_metrics[1], "test_acc": test_metrics[2]
    }

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
    parser.add_argument("--num_ensemble", type=int, default=24, help="Number of models in the random ensemble")

    args = parser.parse_args()

    main_server(args) 