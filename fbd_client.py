import os
import json
import time
import torch
import numpy as np
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import load_fbd_settings, setup_logger, save_optimizer_state_by_block, build_optimizer_with_state
from fbd_dataset import get_data_loader, DATASET_SPECIFIC_RULES

def _test_model(model, evaluator, data_loader, task, criterion, device):
    """Core testing logic, adapted from fbd_server.py"""
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
    auc, acc = evaluator.evaluate(y_score, None, None)
    test_loss = sum(total_loss) / len(total_loss) if total_loss else 0
    return [test_loss, auc, acc]

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

def assemble_model_from_plan(model, received_weights, update_plan):
    """
    Assembles a model by loading weights according to a specific update plan.
    
    Args:
        model (torch.nn.Module): The base model to load weights into.
        received_weights (dict): A dictionary of all weights received from the server.
        update_plan (dict): The specific plan for this client, detailing how to use the weights.
    """
    full_state_dict = {}
    
    if 'model_to_update' in update_plan:
        for component_name, info in update_plan['model_to_update'].items():
            # component_name is the model_part, e.g., 'layer1'
            for param_name, param_value in received_weights.items():
                if param_name.startswith(component_name):
                    full_state_dict[param_name] = param_value
    
    model.load_state_dict(full_state_dict, strict=False)

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

    while True:
        # First, check for the shutdown signal
        shutdown_filepath = os.path.join(args.comm_dir, f"last_round_client_{client_id}.json")
        if os.path.exists(shutdown_filepath):
            with open(shutdown_filepath, 'r') as f:
                shutdown_data = json.load(f)
                if shutdown_data.get("secret") == -1:
                    logger.info("Shutdown signal received. Exiting.")
                    break

        # If no shutdown, look for the file for the current round
        round_filepath = os.path.join(args.comm_dir, f"goods_round_{current_round}_client_{client_id}.pth")
        if os.path.exists(round_filepath):
            # Calculate and log dataset statistics
            stats = get_dataset_stats(data_partition)
            logger.info(f"Dataset stats: {stats}")

            # Prepare test dataset for evaluations
            logger.info("Preparing test dataset for evaluations...")
            DataClass = getattr(medmnist, info['python_class'])
            dataset_rules = DATASET_SPECIFIC_RULES.get(args.experiment_name, {})
            as_rgb = dataset_rules.get("as_rgb", False)
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, size=args.size)
            test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
            test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
            logger.info("Test dataset prepared.")

            # Load the data packet from the server
            data_packet = torch.load(round_filepath, weights_only=False)
            model_weights = data_packet.get("model_weights")
            shipping_list = data_packet.get("shipping_list")
            update_plan = data_packet.get("update_plan")
            optimizer_states = data_packet.get("optimizer_states")
            
            logger.info(f"Round {current_round}: Received {len(shipping_list)} model parts.")

            # Build a local map from block_id to model_part from the update_plan
            block_id_to_model_part = {}
            if update_plan:
                model_to_update_plan = update_plan.get('model_to_update', {})
                for model_part, info in model_to_update_plan.items():
                    block_id_to_model_part[info['block_id']] = model_part

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
                assemble_model_from_plan(model, model_weights, update_plan)
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

            # Evaluate the model on the test set
            test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
            test_loss, test_auc, test_acc = test_metrics[0], test_metrics[1], test_metrics[2]
            logger.info(f"Round {current_round}: Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")

            # Extract updated weights based on the update plan (only trainable parts)
            updated_weights = {}
            trained_state_dict = model.state_dict()
            logger.info(f"Extracting weights for trainable components: {list(model_to_update.keys())}")
            
            for component_name, info in model_to_update.items():
                if info['status'] == 'trainable':
                    block_id = info['block_id']
                    model_part = component_name
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
            updated_optimizer_states = save_optimizer_state_by_block(optimizer, model, block_id_to_model_part, trainable_block_ids)
            logger.info(f"Extracted optimizer states for {len(updated_optimizer_states)} trainable blocks.")

            logger.info(f"Total blocks with updated weights: {len(updated_weights)}")
            print(f"Client {client_id} - Round {current_round}: Train Loss = {loss:.4f}, Test AUC = {test_auc:.4f}, Test Acc = {test_acc:.4f}. Sending {len(updated_weights)} updated weight blocks")

            # Process the data and write back a response
            response_data = {
                "client_id": client_id, 
                "round": current_round, 
                "train_loss": loss,
                "test_loss": test_loss,
                "test_auc": test_auc,
                "test_acc": test_acc,
                "updated_weights": updated_weights,
                "updated_optimizer_states": updated_optimizer_states,
                "dataset_stats": stats
            }
            response_filepath = os.path.join(args.comm_dir, f"response_round_{current_round}_client_{client_id}.pth")
            
            temp_filepath = f"{response_filepath}.tmp"
            torch.save(response_data, temp_filepath)
            os.rename(temp_filepath, response_filepath)

            # Save the cumulative training loss history after each round
            loss_history_path = os.path.join(args.output_dir, f"client_{client_id}_train_losses.json")
            with open(loss_history_path, 'w') as f:
                json.dump(train_losses, f, indent=4)
            logger.info(f"Updated training loss history for round {current_round} at {loss_history_path}")

            # Conditionally clean up the goods file
            if getattr(args, 'remove_communication', False):
                os.remove(round_filepath)

            # --- Memory Cleanup ---
            del model, optimizer, data_packet, model_weights, updated_weights, updated_optimizer_states
            del test_dataset, test_loader, test_evaluator
            torch.cuda.empty_cache()

            current_round += 1
        else:
            # Wait before polling again
            time.sleep(args.poll_interval)

    logger.info("Finished all tasks.") 