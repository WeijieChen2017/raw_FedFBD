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

def train(model, train_loader, task, criterion, optimizer, epochs, device):
    """Trains the model for a specified number of epochs."""
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
    """Build optimizer with state if available - simplified for simulation"""
    optimizer = optim.Adam(trainable_params, lr=default_lr)
    return optimizer

def save_optimizer_state_by_block(optimizer, model, trainable_block_ids):
    """Save optimizer state by block - simplified for simulation"""
    return {}

def simulate_client_task(client_id, data_partition, args, round_num, global_warehouse, client_shipping_list, client_update_plan):
    """Simulated client task that processes one round"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[args.experiment_name]
    task = info['task']
    
    # Load FBD settings to get the training schedule
    fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
    with open(fbd_settings_path, 'r') as f:
        fbd_settings = json.load(f)
    
    # Get the assigned model color for this client and round
    training_schedule = fbd_settings.get('FBD_INFO', {}).get('training_plan', {}).get('schedule', {})
    assigned_model_color = training_schedule.get(str(round_num), {}).get(str(client_id), None)
    
    if not assigned_model_color:
        print(f"Client {client_id}: No assigned model color for round {round_num}. Skipping.")
        return None
    
    # Create the DataLoader for the client's partition
    train_loader = get_data_loader(data_partition, args.batch_size)
    
    if not client_shipping_list:
        print(f"Client {client_id}: No shipping plan for round {round_num}. Skipping.")
        return None
    
    # Prepare test dataset for evaluations
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
    
    # Get model weights and optimizer states from the warehouse
    model_weights = global_warehouse.get_shipping_weights(client_shipping_list)
    optimizer_states = global_warehouse.get_shipping_optimizer_states(client_shipping_list)
    
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
    
    # Filter blocks to only train those belonging to the assigned model color
    fbd_trace = fbd_settings.get('FBD_TRACE', {})
    assigned_model_blocks = [block_id for block_id, info in fbd_trace.items() 
                           if info.get('color') == assigned_model_color]
    
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
    
    # Train the model
    loss = train(model, train_loader, task, criterion, optimizer, args.local_epochs, device)
    
    # Evaluate the model on the test set
    test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
    test_loss, test_auc, test_acc = test_metrics[0], test_metrics[1], test_metrics[2]
    
    # Extract updated weights based on the update plan (only trainable parts for assigned model color)
    updated_weights = {}
    trained_state_dict = model.state_dict()
    
    # Combined comprehensive output line
    print(f"Client {client_id} Round {round_num}: {len(data_partition)} samples, {len(client_shipping_list)} parts, {num_tensors} tensors | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f} | Color: {assigned_model_color} | Trainable: {trainable_components}")
    
    for component_name, component_info in model_to_update.items():
        if (component_info['status'] == 'trainable' and 
            component_info['block_id'] in assigned_model_blocks):
            block_id = component_info['block_id']
            for param_name, param_tensor in trained_state_dict.items():
                if param_name.startswith(component_name):
                    updated_weights[param_name] = param_tensor.detach().clone()
    
    # Save optimizer states for trainable components
    updated_optimizer_states = save_optimizer_state_by_block(optimizer, model, trainable_block_ids)
    
    # Return client response
    return {
        "train_loss": loss,
        "test_metrics": test_metrics,
        "updated_weights": updated_weights,
        "updated_optimizer_states": updated_optimizer_states,
        "round": round_num
    }