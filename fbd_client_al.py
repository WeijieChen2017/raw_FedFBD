import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
import logging
import json
import numpy as np
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

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
            
            loss.backward()
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

def calculate_entropy(prob_matrix):
    """Calculate entropy for each sample from probability matrix."""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    prob_matrix = np.clip(prob_matrix, epsilon, 1.0)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(prob_matrix * np.log(prob_matrix), axis=1)
    return entropy

def calculate_ensemble_ratio(prob_matrix, ensemble_predictions):
    """
    Calculate ensemble ratio metric for active learning.
    
    The ensemble ratio measures disagreement among ensemble members.
    Higher ratio indicates more disagreement and uncertainty.
    
    Args:
        prob_matrix: Average probability matrix from ensemble [N, K]
        ensemble_predictions: Individual predictions from each model [M, N, K]
    
    Returns:
        np.array: Ensemble ratio scores for each sample
    """
    # Get predicted classes from average probabilities
    avg_predictions = np.argmax(prob_matrix, axis=1)
    
    # Count how many models agree with the ensemble prediction
    agreement_counts = []
    for i in range(len(avg_predictions)):
        # Count models that predict the same class as ensemble average
        model_predictions = np.argmax(ensemble_predictions[:, i, :], axis=1)
        agreement = np.sum(model_predictions == avg_predictions[i])
        agreement_counts.append(agreement)
    
    agreement_counts = np.array(agreement_counts)
    num_models = ensemble_predictions.shape[0]
    
    # Convert agreement to disagreement ratio
    # Higher disagreement = higher uncertainty
    disagreement_ratio = 1.0 - (agreement_counts / num_models)
    
    return disagreement_ratio

def extract_features(model, data_loader, device):
    """Extract features from the last layer before classification."""
    model.eval()
    features = []
    indices = []
    
    # Hook to capture features
    activation = {}
    def hook(module, input, output):
        activation['features'] = input[0]
    
    # Register hook on the last linear layer
    if hasattr(model, 'out_layer'):
        # FBD models use out_layer
        handle = model.out_layer.register_forward_hook(hook)
    elif hasattr(model, 'fc'):
        handle = model.fc.register_forward_hook(hook)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            handle = model.classifier[-1].register_forward_hook(hook)
        else:
            handle = model.classifier.register_forward_hook(hook)
    else:
        raise ValueError("Could not find final classification layer")
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            _ = model(inputs.to(device))
            batch_features = activation['features'].cpu().numpy()
            features.append(batch_features)
            
            # Track indices
            batch_size = inputs.size(0)
            start_idx = batch_idx * data_loader.batch_size
            batch_indices = list(range(start_idx, start_idx + batch_size))
            indices.extend(batch_indices)
    
    handle.remove()
    features = np.vstack(features)
    return features, indices

def kmeans_plus_plus_selection(features, indices, n_select):
    """Select diverse samples using k-means++ initialization."""
    if len(indices) <= n_select:
        return indices
    
    # Use cosine distance for diversity
    distances = cosine_distances(features)
    
    selected = []
    # First point: random selection
    first_idx = np.random.randint(len(indices))
    selected.append(indices[first_idx])
    
    # Remaining points: k-means++ style selection
    for _ in range(n_select - 1):
        # Calculate min distance to already selected points
        min_distances = np.min(distances[:, [indices.index(s) for s in selected]], axis=1)
        
        # Probability proportional to distance
        probabilities = min_distances / min_distances.sum()
        
        # Select next point
        next_idx = np.random.choice(len(indices), p=probabilities)
        selected.append(indices[next_idx])
    
    return selected

def active_learning_round(client_id, unlabeled_indices, labeled_indices, model, data_loader, 
                         device, budget_frac, global_N, enable_diversity=True, 
                         metric='entropy', warehouse=None, model_colors=None):
    """
    Perform active learning selection for a client.
    
    Args:
        client_id: Client identifier
        unlabeled_indices: Indices of unlabeled samples available to this client
        labeled_indices: Indices of already labeled samples for this client
        model: Current model for scoring
        data_loader: DataLoader for the full dataset
        device: PyTorch device
        budget_frac: Fraction of global dataset to label this round
        global_N: Total number of samples across all clients
        enable_diversity: Whether to use diversity filtering
    
    Returns:
        selected_indices: Indices selected for labeling
    """
    if len(unlabeled_indices) == 0:
        return []
    
    model.eval()
    
    # Get the base dataset from the data_loader
    if hasattr(data_loader.dataset, 'dataset'):
        # This is already a Subset, get the base dataset
        base_dataset = data_loader.dataset.dataset
    else:
        base_dataset = data_loader.dataset
    
    # Create a subset data loader for unlabeled data
    unlabeled_dataset = torch.utils.data.Subset(base_dataset, unlabeled_indices)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=data_loader.batch_size, shuffle=False
    )
    
    # Determine task type
    if hasattr(model, 'task'):
        task = model.task
    else:
        # Try to infer from a test forward pass
        try:
            test_input = next(iter(unlabeled_loader))[0][:1]  # Get one sample
            test_output = model(test_input.to(device))
            task = 'multi-label, binary-class' if test_output.shape[1] > 1 else 'multi-class'
        except (StopIteration, IndexError):
            # If no unlabeled data, default to multi-class
            task = 'multi-class'
    
    # Step 1: Score samples based on selected metric
    if metric == 'ensemble_ratio' and warehouse and model_colors:
        # Use ensemble predictions for scoring
        ensemble_probs = []
        
        # Get predictions from each model in the ensemble
        for color in model_colors:
            try:
                # Get model weights from warehouse
                from fbd_model_ckpt import get_pretrained_fbd_model
                # Get the actual number of classes from the current model
                if hasattr(model, 'out_layer'):
                    num_classes = model.out_layer.out_features
                elif hasattr(model, 'fc'):
                    num_classes = model.fc.out_features
                elif hasattr(model, 'classifier'):
                    if isinstance(model.classifier, nn.Sequential):
                        num_classes = model.classifier[-1].out_features
                    else:
                        num_classes = model.classifier.out_features
                else:
                    num_classes = 8  # Default for bloodmnist
                
                ensemble_model = get_pretrained_fbd_model(
                    architecture='resnet18',  # Use the actual architecture
                    norm=True,
                    in_channels=3,
                    num_classes=num_classes,
                    use_pretrained=False
                )
                
                model_weights = warehouse.get_model_weights(color)
                ensemble_model.load_state_dict(model_weights)
                ensemble_model.to(device)
                ensemble_model.eval()
                
                # Get predictions
                model_probs = []
                with torch.no_grad():
                    for inputs, _ in unlabeled_loader:
                        outputs = ensemble_model(inputs.to(device))
                        
                        if task == 'multi-label, binary-class':
                            probs = torch.sigmoid(outputs)
                        else:
                            probs = torch.softmax(outputs, dim=1)
                        
                        model_probs.append(probs.cpu().numpy())
                
                ensemble_probs.append(np.vstack(model_probs))
                
            except Exception as e:
                print(f"Warning: Could not get predictions from model {color}: {e}")
        
        if len(ensemble_probs) >= 2:
            # Stack ensemble predictions [M, N, K]
            ensemble_predictions = np.stack(ensemble_probs)
            # Average probabilities
            prob_matrix = np.mean(ensemble_predictions, axis=0)
            # Calculate ensemble ratio scores
            uncertainty_scores = calculate_ensemble_ratio(prob_matrix, ensemble_predictions)
        else:
            # Fall back to entropy if ensemble not available
            print("Falling back to entropy metric (insufficient ensemble models)")
            metric = 'entropy'
    
    if metric == 'entropy' or (metric == 'ensemble_ratio' and 'uncertainty_scores' not in locals()):
        # Use entropy scoring
        all_probs = []
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                outputs = model(inputs.to(device))
                
                # Apply appropriate activation
                if hasattr(model, 'task'):
                    task = model.task
                else:
                    # Infer from output shape
                    task = 'multi-label, binary-class' if outputs.shape[1] > 1 else 'multi-class'
                
                if task == 'multi-label, binary-class':
                    probs = torch.sigmoid(outputs)
                else:
                    probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
        
        prob_matrix = np.vstack(all_probs)
        uncertainty_scores = calculate_entropy(prob_matrix)
    
    # Step 2: Decide local quota
    local_N = len(unlabeled_indices) + len(labeled_indices)
    q_i = int(np.floor(budget_frac * local_N / global_N * global_N))
    q_i = min(q_i, len(unlabeled_indices))  # Can't select more than available
    
    if q_i == 0:
        return []
    
    # Step 3: Pick top uncertain examples
    uncertainty_indices = np.argsort(uncertainty_scores)[::-1]  # Descending order
    
    if enable_diversity and q_i > 10:
        # Optional diversity filter
        # Pre-select 10x budget
        n_candidates = min(10 * q_i, len(unlabeled_indices))
        candidate_relative_indices = uncertainty_indices[:n_candidates]
        candidate_indices = [unlabeled_indices[i] for i in candidate_relative_indices]
        
        # Extract features for diversity selection
        try:
            features, _ = extract_features(model, unlabeled_loader, device)
            candidate_features = features[candidate_relative_indices]
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not extract features for diversity filtering: {e}")
            print("Using random selection instead of diversity filtering")
            # Fall back to random selection from candidates
            np.random.shuffle(candidate_indices)
            selected_indices = candidate_indices[:q_i]
            print(f"Client {client_id}: Selected {len(selected_indices)} samples from {len(unlabeled_indices)} unlabeled (random fallback)")
            return selected_indices
        
        # Select diverse subset
        selected_indices = kmeans_plus_plus_selection(
            candidate_features, candidate_indices, q_i
        )
    else:
        # Without diversity: just take top-q_i
        selected_relative_indices = uncertainty_indices[:q_i]
        selected_indices = [unlabeled_indices[i] for i in selected_relative_indices]
    
    print(f"Client {client_id}: Selected {len(selected_indices)} samples from {len(unlabeled_indices)} unlabeled")
    
    return selected_indices

def simulate_client_task(client_id, data_partition, args, round_num, global_warehouse, client_shipping_list, client_update_plan, active_learning_state=None):
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
    
    # Train the model
    loss, main_loss, reg_loss = train(model, train_loader, task, criterion, optimizer, args.local_epochs, device, client_update_plan, global_warehouse, args)
    
    # Evaluate the model on the test set
    test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
    test_loss, test_auc, test_acc = test_metrics[0], test_metrics[1], test_metrics[2]
    
    # Extract updated weights based on the request_plan (send back specific block IDs)
    updated_weights = {}
    trained_state_dict = model.state_dict()
    
    # Combined comprehensive output line
    print(f"Client {client_id} Round {round_num}: {len(data_partition)} samples, {len(client_shipping_list)} parts, {num_tensors} tensors | Train Loss: {loss:.4f} (Main: {main_loss:.4f}, Reg: {reg_loss:.4f}) | Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f} | Color: {assigned_model_color} | Trainable: {trainable_components}")
    
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
    
    # Handle active learning if enabled
    active_learning_info = {}
    if active_learning_state is not None:
        # Extract current labeled/unlabeled indices for this client
        labeled_indices = active_learning_state.get('labeled_indices', {}).get(str(client_id), [])
        unlabeled_indices = active_learning_state.get('unlabeled_indices', {}).get(str(client_id), [])
        
        # Only perform active learning if we have unlabeled data and it's an AL round
        if unlabeled_indices and active_learning_state.get('perform_selection', False):
            global_N = active_learning_state.get('global_N', 1)
            budget_frac = active_learning_state.get('budget_frac', 0.05)
            
            # Get the original full partition from active learning state
            original_partition = active_learning_state.get('original_partition')
            if original_partition is None:
                # Fallback: try to get from data_partition
                if hasattr(data_partition, 'dataset'):
                    original_partition = data_partition.dataset
                else:
                    original_partition = data_partition
            
            full_partition_loader = torch.utils.data.DataLoader(
                original_partition, 
                batch_size=args.batch_size, 
                shuffle=False
            )
            
            # Perform active learning selection
            selected_indices = active_learning_round(
                client_id=client_id,
                unlabeled_indices=unlabeled_indices,
                labeled_indices=labeled_indices,
                model=model,
                data_loader=full_partition_loader,
                device=device,
                budget_frac=budget_frac,
                global_N=global_N,
                enable_diversity=args.al_diversity if hasattr(args, 'al_diversity') else True,
                metric=args.al_metric if hasattr(args, 'al_metric') else 'entropy',
                warehouse=global_warehouse,
                model_colors=['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
            )
            
            active_learning_info = {
                'selected_indices': selected_indices,
                'num_labeled': len(labeled_indices),
                'num_unlabeled': len(unlabeled_indices),
                'num_selected': len(selected_indices)
            }
    
    # Return client response
    return {
        "train_loss": loss,
        "test_metrics": test_metrics,
        "updated_weights": updated_weights,
        "updated_optimizer_states": updated_optimizer_states,
        "trainable_block_ids": list(trainable_block_ids_set),
        "round": round_num,
        "active_learning_info": active_learning_info
    }