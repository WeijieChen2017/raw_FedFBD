import os
import json
import time
import torch
import numpy as np
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import INFO

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import load_fbd_settings, setup_logger, save_optimizer_state_by_block, build_optimizer_with_state
from fbd_dataset import get_data_loader

def get_dataset_stats(data_partition):
    """Calculates and returns statistics for a given data partition."""
    full_dataset = data_partition.dataset
    partition_indices = data_partition.indices
    
    # MedMNIST datasets have a .labels attribute
    partition_labels = full_dataset.labels[partition_indices]
    
    # Get class distribution
    class_counts = Counter(partition_labels.flatten())
    
    # Convert keys to string for JSON serialization
    class_distribution = {str(k): v for k, v in class_counts.items()}
    
    stats = {
        "total_samples": len(partition_indices),
        "class_distribution": class_distribution
    }
    return stats

def train(model, train_loader, task, criterion, optimizer, epochs, device):
    """
    Trains the model for a specified number of epochs.
    Adapted from code_template/experiments/MedMNIST2D/train_and_eval_pytorch.py
    """
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            total_loss += (epoch_loss / num_batches)

    return total_loss / epochs if epochs > 0 else 0

def assemble_model_from_plan(model, received_weights, update_plan, fbd_trace):
    """
    Assembles a model by loading weights according to a specific update plan.
    
    Args:
        model (torch.nn.Module): The base model to load weights into.
        received_weights (dict): A dictionary of all weights received from the server.
        update_plan (dict): The specific plan for this client, detailing how to use the weights.
        fbd_trace (dict): The global FBD trace to map block IDs to model parts.
    """
    full_state_dict = {}
    
    # Process the model to be updated
    if 'model_to_update' in update_plan:
        for component_name, info in update_plan['model_to_update'].items():
            block_id = info['block_id']
            model_part = fbd_trace[block_id]['model_part']
            for param_name, param_value in received_weights.items():
                if param_name.startswith(model_part):
                    full_state_dict[param_name] = param_value
    
    # This example only assembles the 'model_to_update'.
    # A full implementation would also handle the 'model_as_regularizer' if needed.
    
    model.load_state_dict(full_state_dict, strict=False)
    # This print statement will be replaced by a logger call in the client_task
    # print(f"Assembled model with {len(full_state_dict)} tensors.")

def client_task(client_id, data_partition, args):
    """Client process that actively polls for round-based files, processes them, and sends a response."""
    log_dir = os.path.join(args.output_dir, "fbd_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"client_{client_id}.log")
    logger = setup_logger(f"Client-{client_id}", log_file)
    
    logger.info("Starting...")
    current_round = 0
    train_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[args.experiment_name]
    task = info['task']

    # Create the DataLoader for the client's partition
    train_loader = get_data_loader(data_partition, args.batch_size)
    logger.info(f"Created dataloader with {len(data_partition)} samples.")

    # Calculate and log dataset statistics
    stats = get_dataset_stats(data_partition)
    logger.info(f"Dataset stats: {stats}")

    # Load FBD trace for model assembly
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.py")
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)

    while True:
        # First, check for the shutdown signal
        shutdown_filepath = os.path.join(args.comm_dir, f"last_round_client_{client_id}.json")
        if os.path.exists(shutdown_filepath):
            with open(shutdown_filepath, 'r') as f:
                data = json.load(f)
                if data.get("secret") == -1:
                    logger.info("Shutdown signal received. Exiting.")
                    break

        # If no shutdown, look for the file for the current round
        round_filepath = os.path.join(args.comm_dir, f"goods_round_{current_round}_client_{client_id}.pth")
        if os.path.exists(round_filepath):
            # Load the data packet from the server
            data_packet = torch.load(round_filepath)
            model_weights = data_packet.get("model_weights")
            shipping_list = data_packet.get("shipping_list")
            update_plan = data_packet.get("update_plan")
            optimizer_states = data_packet.get("optimizer_states")
            
            logger.info(f"Round {current_round}: Received {len(shipping_list)} model parts.")

            # Create a base model instance
            model = get_pretrained_fbd_model(
                architecture=args.model_flag,
                norm=args.norm,
                in_channels=args.in_channels,
                num_classes=args.num_classes,
                use_pretrained=False  # Start with an empty model
            )

            # Assemble the model using the received plan and weights
            if update_plan:
                assemble_model_from_plan(model, model_weights, update_plan, fbd_trace)
                # Log after assembly
                num_tensors = len(model.state_dict())
                logger.info(f"Assembled model with {num_tensors} tensors.")

            # Set trainability of parameters and configure optimizer
            model.to(device)
            
            # Freeze all parameters by default
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze parameters of trainable parts
            model_to_update = update_plan.get('model_to_update', {})
            trainable_block_ids = []
            for component_name, info in model_to_update.items():
                if info['status'] == 'trainable':
                    trainable_block_ids.append(info['block_id'])
                    for name, param in model.named_parameters():
                        if name.startswith(component_name):
                            param.requires_grad = True
            
            criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
            
            # Create optimizer with only trainable parameters, loading state if available
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = build_optimizer_with_state(
                model, 
                optimizer_states, 
                list(trainable_params), 
                device, 
                default_lr=args.local_learning_rate
            )
            
            loss = train(model, train_loader, task, criterion, optimizer, args.local_epochs, device)
            logger.info(f"Round {current_round}: Training complete. Loss: {loss:.4f}")
            train_losses.append(loss)

            # Extract updated weights based on the update plan (only trainable parts)
            updated_weights = {}
            trained_state_dict = model.state_dict()
            logger.info(f"Extracting weights for trainable components: {list(model_to_update.keys())}")
            
            for component_name, info in model_to_update.items():
                if info['status'] == 'trainable':
                    block_id = info['block_id']
                    model_part = fbd_trace[block_id]['model_part']
                    block_weights = {}
                    for param_name, param_tensor in trained_state_dict.items():
                        if param_name.startswith(model_part + '.'):
                            block_weights[param_name] = param_tensor.cpu()
                    if block_weights:
                        updated_weights[block_id] = block_weights
                        logger.info(f"Extracted {len(block_weights)} weights for block {block_id} (component: {component_name})")
                    else:
                        logger.warning(f"No weights found for block {block_id} with model_part '{model_part}'")
            
            # Extract updated optimizer state for trainable blocks
            updated_optimizer_states = save_optimizer_state_by_block(optimizer, model, fbd_trace, trainable_block_ids)
            logger.info(f"Extracted optimizer states for {len(updated_optimizer_states)} trainable blocks.")

            logger.info(f"Total blocks with updated weights: {len(updated_weights)}")
            print(f"Client {client_id} - Round {current_round}: Training Loss = {loss:.4f} Sending {len(updated_weights)} updated weight blocks")

            # Process the data and write back a response
            response_data = {
                "client_id": client_id, 
                "round": current_round, 
                "train_loss": loss,
                "updated_weights": updated_weights,
                "updated_optimizer_states": updated_optimizer_states,
                "dataset_stats": stats
            }
            response_filepath = os.path.join(args.comm_dir, f"response_round_{current_round}_client_{client_id}.pth")
            
            temp_filepath = f"{response_filepath}.tmp"
            torch.save(response_data, temp_filepath)
            os.rename(temp_filepath, response_filepath)

            # Conditionally clean up the goods file
            if getattr(args, 'remove_communication', False):
                os.remove(round_filepath)
            
            current_round += 1
        else:
            # Wait before polling again
            time.sleep(args.poll_interval)

    # Save the training loss history
    loss_history_path = os.path.join(args.output_dir, f"client_{client_id}_train_losses.json")
    with open(loss_history_path, 'w') as f:
        json.dump(train_losses, f, indent=4)
    logger.info(f"Saved training loss history to {loss_history_path}")

    logger.info("Finished all tasks.") 