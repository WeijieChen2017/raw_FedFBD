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

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_dataset import get_data_loader, DATASET_SPECIFIC_RULES

# Suppress logging from fbd_model_ckpt to reduce noise
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)

def compute_proximal_loss(local_params, global_params, mu):
    """
    Compute L2 proximal regularization term for FedProx.
    
    Args:
        local_params: Current local model parameters
        global_params: Global model parameters received from server
        mu: Proximal term coefficient
    
    Returns:
        torch.Tensor: Proximal loss term
    """
    proximal_loss = 0.0
    for local_p, global_p in zip(local_params, global_params):
        if local_p.requires_grad:  # Only compute for trainable parameters
            proximal_loss += torch.norm(local_p - global_p.detach()) ** 2
    return (mu / 2.0) * proximal_loss

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

def train(model, train_loader, task, criterion, optimizer, epochs, device, update_plan=None, global_warehouse=None, args=None, global_params=None, mu=0.1):
    """Trains the model for a specified number of epochs."""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_reg_loss = 0
    total_prox_loss = 0
    
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
        epoch_prox_loss = 0
        num_batches = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                main_loss = criterion(outputs, targets)
            else:
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
            
            # Add FedProx proximal term
            prox_loss_value = 0.0
            if global_params is not None and mu > 0:
                prox_loss_value = compute_proximal_loss(model.parameters(), global_params, mu)
                loss = loss + prox_loss_value
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_main_loss += main_loss.item()
            epoch_reg_loss += reg_loss_value.item() if torch.is_tensor(reg_loss_value) else reg_loss_value
            epoch_prox_loss += prox_loss_value.item() if torch.is_tensor(prox_loss_value) else prox_loss_value
            num_batches += 1
        
        if num_batches > 0:
            total_loss += (epoch_loss / num_batches)
            total_main_loss += (epoch_main_loss / num_batches)
            total_reg_loss += (epoch_reg_loss / num_batches)
            total_prox_loss += (epoch_prox_loss / num_batches)

    avg_total_loss = total_loss / epochs if epochs > 0 else 0
    avg_main_loss = total_main_loss / epochs if epochs > 0 else 0
    avg_reg_loss = total_reg_loss / epochs if epochs > 0 else 0
    avg_prox_loss = total_prox_loss / epochs if epochs > 0 else 0
    
    return avg_total_loss, avg_main_loss, avg_reg_loss, avg_prox_loss

def assemble_model_from_plan(model, received_weights, update_plan):
    """Assembles a model by loading weights according to a specific update plan."""
    full_state_dict = {}
    
    if 'model_to_update' in update_plan:
        for component_name, info in update_plan['model_to_update'].items():
            # component_name is the model_part, e.g., 'layer1'
            for param_name, param_value in received_weights.items():
                if param_name.startswith(component_name):
                    full_state_dict[param_name] = param_value
    
    model.load_state_dict(full_state_dict, strict=False)

def build_optimizer_with_state(model, optimizer_states, trainable_params, device, default_lr=0.001):
    """Build optimizer with state if available"""
    optimizer = optim.Adam(trainable_params, lr=default_lr)
    
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

def simulate_client_task(client_id, data_partition, args, round_num, global_warehouse, client_shipping_list, client_update_plan):
    """Simulated client task that processes one round"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[args.experiment_name]
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
    
    if client_update_plan and 'model_to_update' in client_update_plan:
        # Find the model color from the first trainable block in update_plan
        for component_name, component_info in client_update_plan['model_to_update'].items():
            if component_info.get('status') == 'trainable':
                block_id = component_info.get('block_id')
                if block_id in fbd_trace:
                    assigned_model_color = fbd_trace[block_id]['color']
                    break
    
    if not assigned_model_color:
        print(f"Client {client_id}: No trainable blocks in update_plan for round {round_num}. Skipping.")
        return None
    
    # Create the DataLoader for the client's partition
    train_loader = get_data_loader(data_partition, args.batch_size)
    
    if not client_shipping_list:
        print(f"Client {client_id}: No shipping plan for round {round_num}. Skipping.")
        return None
    
    # Prepare test dataset for evaluations
    DataClass = getattr(medmnist, info['python_class'])
    # Use as_rgb setting from config instead of hardcoded rules
    as_rgb = getattr(args, 'as_rgb', False)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, size=args.size)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
    
    # Get model weights and optimizer states from the warehouse
    model_weights = global_warehouse.get_shipping_weights(client_shipping_list)
    all_optimizer_states = global_warehouse.get_shipping_optimizer_states(client_shipping_list)
    
    # Filter optimizer states to only include blocks for the assigned model color
    fbd_trace = fbd_settings.get('FBD_TRACE', {})
    assigned_model_blocks = [block_id for block_id, info in fbd_trace.items() 
                           if info.get('color') == assigned_model_color]
    
    optimizer_states = {block_id: state for block_id, state in all_optimizer_states.items() 
                       if block_id in assigned_model_blocks}
    
    # Build a local map from block_id to model_part from the update_plan
    block_id_to_model_part = {}
    if client_update_plan:
        model_to_update_plan = client_update_plan.get('model_to_update', {})
        for model_part, component_info in model_to_update_plan.items():
            block_id_to_model_part[component_info['block_id']] = model_part
    
    # Create a base model instance
    model = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=args.norm,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        use_pretrained=False  # Start with an empty model
    )
    
    # Assemble the model using the received plan and weights
    num_tensors = 0
    if client_update_plan:
        assemble_model_from_plan(model, model_weights, client_update_plan)
        num_tensors = len(model.state_dict())
    
    # Set trainability of parameters and configure optimizer
    model.to(device)
    
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze parameters of trainable parts that belong to the assigned model color
    model_to_update = client_update_plan.get('model_to_update', {})
    trainable_block_ids = []
    trainable_components = []
    
    for component_name, component_info in model_to_update.items():
        if (component_info['status'] == 'trainable' and 
            component_info['block_id'] in assigned_model_blocks):
            trainable_block_ids.append(component_info['block_id'])
            trainable_components.append(component_name)
            for name, param in model.named_parameters():
                if name.startswith(component_name):
                    param.requires_grad = True
    
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    
    # Create optimizer with only trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer_with_state(
        model, 
        optimizer_states, 
        list(trainable_params), 
        device, 
        default_lr=args.local_learning_rate
    )
    
    # Store global model parameters for FedProx (before training)
    global_params = [param.detach().clone() for param in model.parameters()]
    
    # FedProx hyperparameter (can be configurable)
    fedprox_mu = getattr(args, 'fedprox_mu', 0.1)
    
    # Train the model with FedProx
    loss, main_loss, reg_loss, prox_loss = train(model, train_loader, task, criterion, optimizer, args.local_epochs, device, client_update_plan, global_warehouse, args, global_params, fedprox_mu)
    
    # Evaluate the model on the test set
    test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
    test_loss, test_auc, test_acc = test_metrics[0], test_metrics[1], test_metrics[2]
    
    # Extract updated weights based on the request_plan (send back specific block IDs)
    updated_weights = {}
    trained_state_dict = model.state_dict()
    
    # Combined comprehensive output line with FedProx
    print(f"Client {client_id} Round {round_num}: {len(data_partition)} samples, {len(client_shipping_list)} parts, {num_tensors} tensors | Train Loss: {loss:.4f} (Main: {main_loss:.4f}, Reg: {reg_loss:.4f}, Prox: {prox_loss:.4f}, μ={fedprox_mu}) | Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f} | Color: {assigned_model_color} | Trainable: {trainable_components}")
    
    # Send back weights according to request_plan (all requested blocks)
    # Also include metadata about which blocks were trainable
    trainable_block_ids_set = set()
    for component_name, component_info in model_to_update.items():
        if component_info['status'] == 'trainable':
            trainable_block_ids_set.add(component_info['block_id'])
    
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
    
    # Save optimizer states according to request_plan
    updated_optimizer_states = save_optimizer_state_by_request_plan(optimizer, model, client_request_list, fbd_trace)
    
    # Return client response
    return {
        "train_loss": loss,
        "test_metrics": test_metrics,
        "updated_weights": updated_weights,
        "updated_optimizer_states": updated_optimizer_states,
        "trainable_block_ids": list(trainable_block_ids_set),
        "round": round_num
    }