import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
import logging
import json
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import Counter
import os
import numpy as np
import gc
import copy
import shutil
from fbd_utils import (
    setup_logger,
    FBDWarehouse
)
from fbd_models_siim import get_siim_model
from fbd_models import get_pretrained_fbd_model
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader

# Define SIIM-specific INFO entry since SIIM is not a MedMNIST dataset
SIIM_INFO = {
    'siim': {
        'task': 'segmentation',
        'description': 'SIIM-ACR Pneumothorax Segmentation Challenge',
        'n_channels': 1,
        'label': {'0': 'background', '1': 'pneumothorax'},
        'license': 'SIIM'
    }
}

from fbd_dataset import get_data_loader, DATASET_SPECIFIC_RULES
from fbd_dataset_siim import get_siim_data_loader

# Suppress logging from fbd_model_ckpt to reduce noise
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)

def _test_model(model, evaluator, data_loader, task, criterion, device):
    """Core testing logic for client-side model evaluation"""
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


def _test_siim_model(model, data_loader, criterion, dice_metric, device):
    """Testing logic for SIIM segmentation model (client-side)"""
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss.append(loss.item())
            
            # Calculate Dice score
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid > 0.5).float()
            dice_metric(y_pred=outputs_binary, y=labels)
        
        # Aggregate Dice metric
        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()
    
    test_loss = sum(total_loss) / len(total_loss) if total_loss else 0
    return [test_loss, dice_score, dice_score]  # Return dice_score for both AUC and ACC slots

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

def build_regularizer_model(regularizer_spec, global_warehouse, args, device):
    """
    Build a regularizer model from block IDs specification using the global warehouse.
    
    Args:
        regularizer_spec (dict): Dictionary mapping model parts to block IDs
        global_warehouse: The global warehouse containing all model weights
        args: Training arguments
        device: PyTorch device
        
    Returns:
        torch.nn.Module: Assembled regularizer model
    """
    # Create a base model instance
    if args.experiment_name == "siim":
        reg_model = get_siim_model(
            architecture=args.model_flag,
            in_channels=args.n_channels,
            out_channels=args.num_classes,
            model_size=getattr(args, 'model_size', 'standard')
        )
    else:
        reg_model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        )
    
    # Get weights for the regularizer blocks from warehouse
    reg_block_ids = list(regularizer_spec.values())
    reg_weights = global_warehouse.get_shipping_weights(reg_block_ids)
    
    # Load the weights into the regularizer model
    reg_model.load_state_dict(reg_weights, strict=False)
    reg_model.to(device)
    reg_model.eval()  # Set to eval mode since we only need predictions
    return reg_model

def train(model, train_loader, task, criterion, optimizer, epochs, device, update_plan=None, global_warehouse=None, args=None):
    """Trains the model for a specified number of epochs."""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_reg_loss = 0
    
    # Check if we need to build regularizer models
    regularizer_models = []
    regularizer_coefficient = 0.0
    regularizer_type = None
    
    if update_plan and 'model_as_regularizer' in update_plan and global_warehouse and args:
        # Load regularizer settings
        try:
            # Check if command line args override the config
            if hasattr(args, 'reg') and args.reg is not None:
                # Map command line args to regularizer types
                if args.reg == 'none':
                    regularizer_type = None
                    regularizer_coefficient = 0.0
                else:
                    regularizer_type = 'weights distance' if args.reg == 'w' else 'consistency loss'
                    regularizer_coefficient = args.reg_coef if args.reg_coef is not None else 0.1
            else:
                # Fall back to config file settings
                fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
                with open(fbd_settings_path, 'r') as f:
                    fbd_settings = json.load(f)
                
                regularizer_params = fbd_settings.get('REGULARIZER_PARAMS', {})
                regularizer_type = regularizer_params.get('type')
                regularizer_coefficient = regularizer_params.get('coefficient', 0.1)
            
            if regularizer_type in ['consistency loss', 'weights distance']:
                
                # Build regularizer models
                for regularizer_spec in update_plan['model_as_regularizer']:
                    reg_model = build_regularizer_model(regularizer_spec, global_warehouse, args, device)
                    regularizer_models.append(reg_model)
                
                print(f"Built {len(regularizer_models)} regularizer models (type: {regularizer_type}) with coefficient {regularizer_coefficient}")
        except Exception as e:
            print(f"Warning: Could not load regularizer settings: {e}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_main_loss = 0
        epoch_reg_loss = 0
        num_batches = 0
        total_batches = len(train_loader)
        print(f"📊 Starting epoch {epoch+1}/{epochs} with {total_batches} batches")
        
        for batch_data in train_loader:
            # Handle different data formats
            if isinstance(batch_data, dict):
                # Dictionary format (SIIM dataset)
                inputs = batch_data["image"]
                targets = batch_data["label"]
            else:
                # Tuple format (original datasets)
                inputs, targets = batch_data
            
            optimizer.zero_grad()
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                main_loss = criterion(outputs, targets)
            elif task == 'segmentation':
                # For segmentation tasks (like SIIM), keep targets as float
                targets = targets.to(torch.float32).to(device)
                
                # Debug: Print tensor properties
                if num_batches == 0:  # Only print for first batch to avoid spam
                    print(f"🔍 Debug - Batch shapes and ranges:")
                    print(f"   Inputs: {inputs.shape}, range [{inputs.min():.3f}, {inputs.max():.3f}]")
                    print(f"   Targets: {targets.shape}, unique values: {torch.unique(targets)}")
                    print(f"   Outputs: {outputs.shape}, range [{outputs.min():.3f}, {outputs.max():.3f}]")
                    print(f"   Positive target voxels: {(targets > 0.5).sum().item()} / {targets.numel()} ({(targets > 0.5).sum().item() / targets.numel() * 100:.2f}%)")
                
                main_loss = criterion(outputs, targets)
                
                # Debug: Check loss value
                if num_batches == 0:
                    loss_type = getattr(args, 'loss_type', 'dice_ce')
                    print(f"   Loss function: {loss_type}")
                    print(f"   Computed loss: {main_loss.item():.6f}")
                    if main_loss.item() < 0.001:
                        print("   ⚠️  WARNING: Loss is extremely small!")
                    elif main_loss.item() > 10:
                        print("   ⚠️  WARNING: Loss is very large!")
                    else:
                        print("   ✅ Loss value seems normal")
            else:
                # For classification tasks
                targets = torch.squeeze(targets, 1).long().to(device)
                main_loss = criterion(outputs, targets)
            
            loss = main_loss
            reg_loss_value = 0.0
            
            # Add regularization loss if regularizer models exist
            if regularizer_models and regularizer_coefficient > 0:
                if regularizer_type == 'consistency loss':
                    # Consistency loss: L2 distance between model outputs
                    consistency_loss = 0.0
                    with torch.no_grad():
                        for reg_model in regularizer_models:
                            reg_outputs = reg_model(inputs.to(device))
                            # Compute L2 distance between predictions
                            consistency_loss += torch.norm(outputs - reg_outputs, p=2)
                    
                    # Average over regularizer models and add to main loss
                    consistency_loss = consistency_loss / len(regularizer_models)
                    reg_loss_value = regularizer_coefficient * consistency_loss
                    loss = loss + reg_loss_value
                
                elif regularizer_type == 'weights distance':
                    # Weights distance: L2 distance between model parameters
                    weights_distance = 0.0
                    main_params = dict(model.named_parameters())
                    
                    for reg_model in regularizer_models:
                        reg_params = dict(reg_model.named_parameters())
                        model_distance = 0.0
                        
                        # Compute L2 distance between corresponding parameters
                        for param_name, main_param in main_params.items():
                            if param_name in reg_params and main_param.requires_grad:
                                reg_param = reg_params[param_name]
                                param_distance = torch.norm(main_param - reg_param, p=2)
                                model_distance += param_distance
                        
                        weights_distance += model_distance
                    
                    # Average over regularizer models and add to main loss
                    weights_distance = weights_distance / len(regularizer_models)
                    reg_loss_value = regularizer_coefficient * weights_distance
                    loss = loss + reg_loss_value
            
            # Validate loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Client {client_id} Round {round_num}: Invalid loss detected: {loss.item():.6f} (main: {main_loss:.6f}, reg: {reg_loss:.6f})")
                print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}], Label range: [{labels.min():.3f}, {labels.max():.3f}]")
                print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                continue  # Skip this batch
            
            loss.backward()
            
            # Add gradient clipping to prevent divergence
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_main_loss += main_loss.item()
            epoch_reg_loss += reg_loss_value.item() if torch.is_tensor(reg_loss_value) else reg_loss_value
            num_batches += 1
        
        if num_batches > 0:
            total_loss += (epoch_loss / num_batches)
            total_main_loss += (epoch_main_loss / num_batches)
            total_reg_loss += (epoch_reg_loss / num_batches)

    avg_total_loss = total_loss / epochs if epochs > 0 else 0
    avg_main_loss = total_main_loss / epochs if epochs > 0 else 0
    avg_reg_loss = total_reg_loss / epochs if epochs > 0 else 0
    
    return avg_total_loss, avg_main_loss, avg_reg_loss

def assemble_model_from_plan(model, received_weights, update_plan):
    """Assembles a model by loading weights according to a specific update plan."""
    full_state_dict = {}
    
    # Get FBD parts and create mapping from component names to parameter names
    fbd_parts = model.get_fbd_parts()
    component_to_param_names = {}
    
    for component_name, component_module in fbd_parts.items():
        component_to_param_names[component_name] = []
        # Get all parameters from this component
        component_params = dict(component_module.named_parameters())
        
        # Find corresponding parameter names in the full model
        for full_name, full_param in model.named_parameters():
            for comp_param_name, comp_param in component_params.items():
                if full_param is comp_param:
                    component_to_param_names[component_name].append(full_name)
    
    if 'model_to_update' in update_plan:
        for component_name, info in update_plan['model_to_update'].items():
            # Get the actual parameter names for this component
            if component_name in component_to_param_names:
                component_param_names = component_to_param_names[component_name]
                
                for param_name, param_value in received_weights.items():
                    if param_name in component_param_names:
                        full_state_dict[param_name] = param_value
    
    model.load_state_dict(full_state_dict, strict=False)

def build_optimizer_with_state(model, optimizer_states, trainable_params, device, default_lr=0.001):
    """Build optimizer with state if available"""
    optimizer = optim.Adam(trainable_params, lr=default_lr, weight_decay=1e-5)
    
    if not optimizer_states:
        return optimizer
    
    # Create a mapping from parameter to its position in optimizer.param_groups
    param_to_index = {}
    param_index = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            param_to_index[param] = param_index
            param_index += 1
    
    # Create a mapping from parameter name to parameter object
    name_to_param = dict(model.named_parameters())
    
    # Load optimizer states from all relevant blocks
    for block_id, block_state in optimizer_states.items():
        if 'state' not in block_state:
            continue
            
        for param_name, param_state in block_state['state'].items():
            if param_name in name_to_param:
                param = name_to_param[param_name]
                if param in param_to_index:
                    param_idx = param_to_index[param]
                    
                    # Initialize optimizer state for this parameter
                    optimizer.state[param] = {}
                    
                    # Load momentum/Adam states
                    for state_key, state_value in param_state.items():
                        if isinstance(state_value, torch.Tensor):
                            optimizer.state[param][state_key] = state_value.to(device)
                        else:
                            optimizer.state[param][state_key] = state_value
    
    return optimizer

def save_optimizer_state_by_request_plan(optimizer, model, client_request_list, fbd_trace):
    """Save optimizer state according to request_plan"""
    optimizer_states = {}
    
    # Create mapping from parameter name to parameter object
    name_to_param = dict(model.named_parameters())
    
    # Save optimizer states for blocks in the request list
    for block_id in client_request_list:
        if block_id in fbd_trace:
            model_part = fbd_trace[block_id]['model_part']
            block_states = {
                'param_groups': [],
                'state': {}
            }
            
            # Save optimizer hyperparameters
            for group in optimizer.param_groups:
                group_info = {key: value for key, value in group.items() if key != 'params'}
                block_states['param_groups'].append(group_info)
            
            # Save parameter states for this block
            for param_name, param in name_to_param.items():
                if param_name.startswith(model_part) and param in optimizer.state:
                    param_state = {}
                    for state_key, state_value in optimizer.state[param].items():
                        if isinstance(state_value, torch.Tensor):
                            param_state[state_key] = state_value.cpu().clone()
                        else:
                            param_state[state_key] = state_value
                    
                    if param_state:
                        block_states['state'][param_name] = param_state
            
            if block_states['state']:
                optimizer_states[block_id] = block_states
    
    return optimizer_states

def save_model_state_to_disk(model, optimizer, client_id, round_num, assigned_model_color, output_dir):
    """
    Save model state dict and optimizer state dict to disk.
    Only keeps the latest round to save disk space.
    """
    import glob
    
    # Clean up old client directories for this client
    old_client_pattern = os.path.join(output_dir, f"client_{client_id}_round_*")
    old_client_dirs = glob.glob(old_client_pattern)
    for old_dir in old_client_dirs:
        try:
            shutil.rmtree(old_dir)
        except Exception as e:
            print(f"Warning: Could not remove old client directory {old_dir}: {e}")
    
    # Create client-specific directory for current round
    client_dir = os.path.join(output_dir, f"client_{client_id}_round_{round_num}")
    os.makedirs(client_dir, exist_ok=True)
    
    # Ensure model and optimizer are on CPU before saving to avoid GPU sync issues
    model.cpu()
    
    # Save model state dict
    model_path = os.path.join(client_dir, f"model_{assigned_model_color}.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save optimizer state dict
    optimizer_path = os.path.join(client_dir, f"optimizer_{assigned_model_color}.pth")
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return {
        "model_path": model_path,
        "optimizer_path": optimizer_path,
        "client_dir": client_dir
    }

def load_model_state_from_disk(model_path, optimizer_path, device):
    """
    Load model state dict and optimizer state dict from disk.
    
    Args:
        model_path: Path to saved model state dict
        optimizer_path: Path to saved optimizer state dict
        device: PyTorch device
    
    Returns:
        tuple: (model_state_dict, optimizer_state_dict)
    """
    model_state_dict = torch.load(model_path, map_location=device)
    optimizer_state_dict = torch.load(optimizer_path, map_location=device)
    
    return model_state_dict, optimizer_state_dict

def create_model_for_evaluation(args, device):
    """
    Create a model instance for evaluation purposes.
    
    Args:
        args: Training arguments
        device: PyTorch device
    
    Returns:
        PyTorch model
    """
    if args.experiment_name == "siim":
        from fbd_models_siim import get_siim_model
        model = get_siim_model(
            architecture=args.model_flag,
            in_channels=args.n_channels,
            out_channels=args.num_classes,
            model_size=getattr(args, 'model_size', 'standard')
        )
    else:
        model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        )
    
    model.to(device)
    return model

def simulate_client_task(model_or_reusable_model, client_id, client_dataset, args, round_num, warehouse, shipping_list, update_plan, use_disk=False, val_dataset=None):
    """
    Simulates a single client's task for a given round, now with a flag to control model copying.
    
    Args:
        model_or_reusable_model: Either a client-specific model instance (if use_disk=False) 
                                 or a reusable model to be copied (if use_disk=True).
        client_id (int): The client's ID.
        client_dataset: The client's local dataset.
        args: Configuration arguments.
        round_num (int): The current round number.
        warehouse: The central warehouse object.
        shipping_list (list): List of models to be shipped from the warehouse.
        update_plan (dict): Plan for updating models after training.
        use_disk (bool): If True, model_or_reusable_model is copied. If False, it's used directly.
        val_dataset: Optional validation dataset for this client.
        
    Returns:
        A dictionary containing the client's response.
    """
    if use_disk:
        # Disk mode: copy the reusable model for this client task
        model = copy.deepcopy(model_or_reusable_model)
    else:
        # In-memory mode: use the provided model instance directly
        model = model_or_reusable_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get info from MedMNIST INFO or SIIM_INFO
    if args.experiment_name in INFO:
        info = INFO[args.experiment_name]
    elif args.experiment_name in SIIM_INFO:
        info = SIIM_INFO[args.experiment_name]
    else:
        raise ValueError(f"Unknown experiment: {args.experiment_name}")
    
    task = info['task']
    
    # Load FBD settings to get the training schedule
    fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
    with open(fbd_settings_path, 'r') as f:
        fbd_settings = json.load(f)
    
    # Load request plan to determine what blocks to send back
    request_plan_path = f"config/{args.experiment_name}/request_plan.json"
    with open(request_plan_path, 'r') as f:
        request_plans = json.load(f)
    
    client_request_list = request_plans.get(str(round_num + 1), {}).get(str(client_id), [])
    
    # Get the assigned model color from what the update_plan tells us to train
    fbd_trace = fbd_settings.get('FBD_TRACE', {})
    assigned_model_color = None
    
    if update_plan and 'model_to_update' in update_plan:
        # Find the model color from the first trainable block in update_plan
        for component_name, component_info in update_plan['model_to_update'].items():
            if component_info.get('status') == 'trainable':
                block_id = component_info.get('block_id')
                if block_id in fbd_trace:
                    assigned_model_color = fbd_trace[block_id]['color']
                    break
    
    if not assigned_model_color:
        print(f"Client {client_id}: No trainable blocks in update_plan for round {round_num}. Skipping.")
        return None
    
    # Create the DataLoader for the client's partition
    if args.experiment_name == "siim":
        # Use balanced data loader to handle extreme class imbalance
        train_loader = get_siim_data_loader(
            client_dataset, 
            args.batch_size, 
            balanced=True,  # Enable balanced sampling
            positive_ratio=0.4  # 40% positive samples per batch
        )
    else:
        DataClass = getattr(medmnist, info['python_class'])
        as_rgb = getattr(args, 'as_rgb', False)
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        train_loader = DataLoader(DataClass(split='train', transform=data_transform, download=True, as_rgb=as_rgb, size=args.size), batch_size=args.batch_size, shuffle=True)
        test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)

    # Get model weights and optimizer states from the warehouse
    model_weights = warehouse.get_shipping_weights(shipping_list)
    all_optimizer_states = warehouse.get_shipping_optimizer_states(shipping_list)
    
    # Filter optimizer states to only include blocks for the assigned model color
    assigned_model_blocks = [block_id for block_id, info in fbd_trace.items() 
                           if info.get('color') == assigned_model_color]
    
    optimizer_states = {block_id: state for block_id, state in all_optimizer_states.items() 
                       if block_id in assigned_model_blocks}
    
    # Assemble the model using the received plan and weights
    num_tensors = 0
    if update_plan:
        assemble_model_from_plan(model, model_weights, update_plan)
        num_tensors = len(model.state_dict())
    
    # Set trainability of parameters and configure optimizer
    model.to(device)
    
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze parameters of trainable parts that belong to the assigned model color
    model_to_update = update_plan.get('model_to_update', {})
    trainable_block_ids = []
    trainable_components = []
    
    # Get FBD parts and create mapping from component names to parameter names
    fbd_parts = model.get_fbd_parts()
    component_to_param_names = {}
    
    for component_name, component_module in fbd_parts.items():
        component_to_param_names[component_name] = []
        # Get all parameters from this component
        component_params = dict(component_module.named_parameters())
        
        # Find corresponding parameter names in the full model
        for full_name, full_param in model.named_parameters():
            for comp_param_name, comp_param in component_params.items():
                if full_param is comp_param:
                    component_to_param_names[component_name].append(full_name)
    
    for component_name, component_info in model_to_update.items():
        if (component_info['status'] == 'trainable' and 
            component_info['block_id'] in assigned_model_blocks):
            trainable_block_ids.append(component_info['block_id'])
            trainable_components.append(component_name)
            
            # Unfreeze parameters for this component
            if component_name in component_to_param_names:
                for param_name in component_to_param_names[component_name]:
                    for name, param in model.named_parameters():
                        if name == param_name:
                            param.requires_grad = True
    
    if args.experiment_name == "siim":
        # Import alternative loss functions
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from alternative_loss_functions import get_siim_loss_function
        
        loss_type = getattr(args, 'loss_type', 'dice_ce')
        criterion = get_siim_loss_function(loss_type)
        print(f"🎯 Using loss function: {loss_type}")
    else:
        criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    
    # Create optimizer with only trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # If no parameters are trainable (empty update plan), make all parameters trainable
    if len(trainable_params) == 0:
        print(f"Client {client_id}: Round {round_num} with no trainable parameters. Making all parameters trainable.")
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = list(model.parameters())
    
    # Debug: Print number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Client {client_id} Round {round_num}: {trainable_count}/{total_params} parameters trainable ({trainable_count/total_params*100:.1f}%)")
    
    # Ensure we have trainable parameters before creating optimizer
    if len(trainable_params) == 0:
        raise ValueError(f"Client {client_id}: No trainable parameters found. Cannot create optimizer.")
    
    optimizer = build_optimizer_with_state(
        model, 
        optimizer_states, 
        trainable_params, 
        device, 
        default_lr=args.local_learning_rate
    )
    
    # Train the model
    loss, main_loss, reg_loss = train(model, train_loader, task, criterion, optimizer, args.local_epochs, device, update_plan, warehouse, args)
    
    # Save model state after training
    model_state_info = save_model_state_to_disk(model, optimizer, client_id, round_num, assigned_model_color, args.output_dir)
    
    # Load the trained weights back for evaluation
    model.load_state_dict(torch.load(model_state_info["model_path"]))
    model.to(device)

    # Evaluate the model on validation set if available, otherwise use training set
    val_metrics = None
    if val_dataset is not None and len(val_dataset) > 0:
        # Create validation data loader
        if args.experiment_name == "siim":
            # Use standard loader for validation (no balancing needed for evaluation)
            val_loader = get_siim_data_loader(val_dataset, args.batch_size, shuffle=False, balanced=False)
            from monai.metrics import DiceMetric
            dice_metric_val = DiceMetric(include_background=True, reduction="mean")
            val_metrics = _test_siim_model(model, val_loader, criterion, dice_metric_val, device)
            val_loss, val_dice, val_acc = val_metrics[0], val_metrics[1], val_metrics[2]
            val_auc = val_dice  # Use dice score as AUC placeholder
            print(f"Client {client_id} Validation: Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        else:
            # For non-SIIM datasets, create validation loader
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
            val_metrics = _test_model(model, test_evaluator, val_loader, task, criterion, device)
            val_loss, val_auc, val_acc = val_metrics[0], val_metrics[1], val_metrics[2]
            print(f"Client {client_id} Validation: Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, ACC: {val_acc:.4f}")
    
    # Also evaluate on training set for comparison (using last few batches)
    if args.experiment_name == "siim":
        from monai.metrics import DiceMetric
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        test_metrics = _test_siim_model(model, train_loader, criterion, dice_metric, device)
        test_loss, test_dice, test_acc = test_metrics[0], test_metrics[1], test_metrics[2]
        test_auc = test_dice  # Use dice score as AUC placeholder
    else:
        test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
        test_metrics = _test_model(model, test_evaluator, train_loader, task, criterion, device)
        test_loss, test_auc, test_acc = test_metrics[0], test_metrics[1], test_metrics[2]
    
    # Clean up GPU memory before the next client
    model.cpu()
    del optimizer
    torch.cuda.empty_cache()
    
    # Extract updated weights based on the request_plan (send back specific block IDs)
    updated_weights = {}
    # Load the trained state dict from disk
    trained_state_dict = torch.load(model_state_info["model_path"], map_location='cpu')
    
    # Combined comprehensive output line
    output_str = f"Client {client_id} Round {round_num}: {len(client_dataset)} samples, {len(shipping_list)} parts, {num_tensors} tensors | Train Loss: {loss:.4f} (Main: {main_loss:.4f}, Reg: {reg_loss:.4f}) | Train Eval: Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f}"
    
    # Add validation metrics if available
    if val_metrics is not None:
        if args.experiment_name == "siim":
            output_str += f" | Val: Loss: {val_loss:.4f}, Dice: {val_dice:.4f}"
        else:
            output_str += f" | Val: Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, ACC: {val_acc:.4f}"
    
    output_str += f" | Color: {assigned_model_color} | Trainable: {trainable_components}"
    print(output_str)
    
    # Send back weights according to request_plan (all requested blocks)
    # Also include metadata about which blocks were trainable
    trainable_block_ids_set = set()
    for component_name, component_info in model_to_update.items():
        if component_info['status'] == 'trainable':
            trainable_block_ids_set.add(component_info['block_id'])
    
    # For SIIM UNet, we treat the entire model as one unit
    # Since our get_fbd_parts() returns {'unet': self.unet}, we need to handle this specially
    if args.experiment_name == "siim" and client_request_list:
        # Send all model weights for any requested block (since we can't decompose the UNet properly)
        all_weights = {}
        for param_name, param_tensor in trained_state_dict.items():
            all_weights[param_name] = param_tensor.detach().clone()
        
        # Assign the same weights to all requested blocks
        for block_id in client_request_list:
            if block_id in fbd_trace:
                updated_weights[block_id] = all_weights.copy()
    else:
        # Original logic for other datasets
        for block_id in client_request_list:
            if block_id in fbd_trace:
                model_part = fbd_trace[block_id]['model_part']
                block_weights = {}
                
                # Extract parameters for this specific block
                for param_name, param_tensor in trained_state_dict.items():
                    if param_name.startswith(model_part):
                        block_weights[param_name] = param_tensor.detach().clone()
                
                if block_weights:
                    updated_weights[block_id] = block_weights
    
    # Create a dummy optimizer to save state from (model is on CPU)
    # The state was already saved to disk during training
    dummy_optimizer = build_optimizer_with_state(
        model, 
        optimizer_states, 
        list(filter(lambda p: p.requires_grad, model.parameters())), 
        'cpu', 
        default_lr=args.local_learning_rate
    )
    dummy_optimizer.load_state_dict(torch.load(model_state_info["optimizer_path"]))
    updated_optimizer_states = save_optimizer_state_by_request_plan(dummy_optimizer, model, client_request_list, fbd_trace)
    
    # Return client response
    response = {
        "train_loss": loss,
        "test_metrics": test_metrics,
        "updated_weights": updated_weights,
        "updated_optimizer_states": updated_optimizer_states,
        "trainable_block_ids": list(trainable_block_ids_set),
        "round": round_num,
        "model_state_info": model_state_info
    }
    
    # Add validation metrics if available
    if val_metrics is not None:
        response["val_metrics"] = val_metrics
    
    return response