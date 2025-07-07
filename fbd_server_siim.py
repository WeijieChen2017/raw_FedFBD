import json
import os
import torch
import torch.nn as nn
import numpy as np
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms
import logging
import gc

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

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_models_siim import get_siim_model
from fbd_utils import load_fbd_settings, FBDWarehouse
from fbd_dataset import DATASET_SPECIFIC_RULES

# Suppress logging from fbd_model_ckpt to reduce noise
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)

def save_server_model_to_disk(model, model_color, round_num, output_dir):
    """
    Save server model state dict to disk to free GPU memory.
    Only keeps the latest round to save disk space.
    
    Args:
        model: PyTorch model
        model_color: Model color identifier
        round_num: Round number
        output_dir: Output directory to save states
    
    Returns:
        str: Path to saved model file
    """
    import glob
    import shutil
    
    # Clean up old server directories
    old_server_pattern = os.path.join(output_dir, "server_round_*")
    old_server_dirs = glob.glob(old_server_pattern)
    for old_dir in old_server_dirs:
        try:
            shutil.rmtree(old_dir)
        except Exception as e:
            print(f"Warning: Could not remove old server directory {old_dir}: {e}")
    
    # Create server model directory for current round
    server_dir = os.path.join(output_dir, f"server_round_{round_num}")
    os.makedirs(server_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(server_dir, f"model_{model_color}.pth")
    torch.save(model.state_dict(), model_path)
    
    # Move model to CPU and clear GPU memory
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return model_path

def fix_state_dict_prefixes(saved_state_dict, model):
    """
    Fix state dict key prefix mismatches.
    
    Args:
        saved_state_dict: The state dict loaded from disk
        model: The target model
        
    Returns:
        Fixed state dict with corrected key names
    """
    model_state_dict = model.state_dict()
    fixed_state_dict = {}
    
    for model_key in model_state_dict.keys():
        # Try different prefix variations
        possible_keys = [
            model_key,  # Exact match
            model_key.replace("unet.", ""),  # Remove unet prefix -> "model.0.conv..."
            model_key.replace("unet.", "model."),  # Replace unet with model -> "model.model.0.conv..." 
            model_key.replace("unet.model.", "model."),  # Replace "unet.model." with "model." -> "model.0.conv..."
            model_key.replace("unet.model.", ""),  # Remove "unet.model." -> "0.conv..."
        ]
        
        key_found = False
        for possible_key in possible_keys:
            if possible_key in saved_state_dict:
                fixed_state_dict[model_key] = saved_state_dict[possible_key]
                key_found = True
                break
        
        if not key_found:
            print(f"Warning: Could not find mapping for key: {model_key}")
            # Keep the original parameter (random initialization)
            fixed_state_dict[model_key] = model_state_dict[model_key]
    
    return fixed_state_dict

def detect_model_size_from_weights(model_path):
    """
    Detect model size (small/standard) from saved weights by checking tensor dimensions.
    
    Args:
        model_path: Path to saved model state dict
    
    Returns:
        str: 'small' or 'standard'
    """
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        return detect_model_size_from_state_dict(state_dict)
        
    except Exception as e:
        print(f"Warning: Could not detect model size from weights: {e}")
        return 'standard'

def detect_model_size_from_state_dict(state_dict):
    """
    Detect model size from state dict by checking tensor dimensions.
    
    Args:
        state_dict: PyTorch state dict
    
    Returns:
        str: 'small' or 'standard'
    """
    try:
        # Check the first conv layer to determine feature size
        # Small models have 64 features, standard have 128
        first_conv_key = None
        for key in state_dict.keys():
            if 'conv.unit0.conv.weight' in key and 'model.0' in key:
                first_conv_key = key
                break
        
        if first_conv_key and first_conv_key in state_dict:
            # Shape should be [features, in_channels, ...] 
            # Small: [48, 1, 3, 3, 3], Standard: [96, 1, 3, 3, 3], Large: [192, 1, 3, 3, 3], etc.
            first_conv_shape = state_dict[first_conv_key].shape
            features = first_conv_shape[0]
            if features == 48:
                return 'small'
            elif features == 96:
                return 'standard'
            elif features == 192:
                return 'large'
            elif features == 384:
                return 'xlarge'
            elif features == 576:
                return 'xxlarge'
            elif features == 768:
                return 'mega'
            # Legacy support for old feature sizes
            elif features == 64:
                return 'small'  # Old small
            elif features == 128:
                return 'standard'  # Old standard
            elif features == 256:
                return 'large'  # Old large
            elif features == 512:
                return 'xlarge'  # Old xlarge
            elif features == 1024:
                return 'mega'  # Old mega
        
        # Fallback: assume standard if we can't detect
        return 'standard'
        
    except Exception as e:
        print(f"Warning: Could not detect model size from state dict: {e}")
        return 'standard'

def detect_model_size_from_warehouse(warehouse):
    """
    Detect model size from warehouse by checking any available model weights.
    
    Args:
        warehouse: FBDWarehouse instance
    
    Returns:
        str: 'small' or 'standard'
    """
    try:
        # Try to get M0 weights first, then any available model
        for model_color in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
            model_weights = warehouse.get_model_weights(model_color)
            if model_weights:
                return detect_model_size_from_state_dict(model_weights)
        
        # Fallback: assume standard if no weights found
        return 'standard'
        
    except Exception as e:
        print(f"Warning: Could not detect model size from warehouse: {e}")
        return 'standard'

def load_server_model_from_disk(model_path, args, experiment_name, device):
    """
    Load server model state dict from disk with automatic model size detection.
    
    Args:
        model_path: Path to saved model state dict
        args: Training arguments
        experiment_name: Name of the experiment
        device: PyTorch device
    
    Returns:
        PyTorch model with loaded weights
    """
    # For SIIM, auto-detect model size from saved weights to avoid dimension mismatches
    if experiment_name == "siim":
        detected_size = detect_model_size_from_weights(model_path)
        current_size = getattr(args, 'model_size', 'standard')
        
        if detected_size != current_size:
            print(f"Model size mismatch detected: weights are '{detected_size}' but args specify '{current_size}'. Using '{detected_size}' to match weights.")
        
        model = get_siim_model(
            architecture=args.model_flag,
            in_channels=args.n_channels,
            out_channels=args.num_classes,
            model_size=detected_size  # Use detected size instead of args
        )
    else:
        model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        )
    
    # Load weights with error handling for prefix mismatches
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e):
            print(f"Warning: State dict mismatch detected. Attempting to fix key prefixes...")
            saved_state_dict = torch.load(model_path, map_location=device)
            fixed_state_dict = fix_state_dict_prefixes(saved_state_dict, model)
            model.load_state_dict(fixed_state_dict, strict=False)
            print("Successfully loaded state dict with prefix corrections.")
        else:
            raise e
    
    model.to(device)
    return model

def _get_scores(model, data_loader, task, device):
    """Runs the model on the data and returns the raw scores."""
    model.eval()
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle both dictionary format (SIIM) and tuple format (MedMNIST)
            if isinstance(batch_data, dict):
                inputs = batch_data["image"].to(device)
            else:
                inputs, _ = batch_data
                inputs = inputs.to(device)
                
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class' or task == 'segmentation':
                m = nn.Sigmoid()
                outputs = m(outputs)
            else:
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
            
            y_score = torch.cat((y_score, outputs), 0)
            
    return y_score.detach().cpu().numpy()

def _test_model(model, evaluator, data_loader, task, criterion, device):
    """Core testing logic for server-side model evaluation"""
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
    """Testing logic for SIIM segmentation model"""
    model.eval()
    total_loss = []
    dice_scores = []
    
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

def evaluate_server_model(args, model_color, model_flag, experiment_name, test_dataset, warehouse):
    """Evaluate a server model"""
    # Allow forcing evaluation on CPU to save GPU memory
    if getattr(args, 'eval_on_cpu', False):
        device = torch.device("cpu")
        print(f"  Evaluating {model_color} on CPU (--eval_on_cpu enabled)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle SIIM dataset separately
    if experiment_name == "siim":
        task = "segmentation"
        from monai.losses import DiceCELoss
        from monai.metrics import DiceMetric
        
        # Create test loader, ensuring num_workers=0 to prevent deadlocks
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        criterion = DiceCELoss(to_onehot_y=False, sigmoid=True).to(device)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        
        # Create model with auto-detected size to match existing weights
        detected_size = detect_model_size_from_warehouse(warehouse)
        current_size = getattr(args, 'model_size', 'standard')
        
        if detected_size != current_size:
            print(f"Auto-detected model size '{detected_size}' from warehouse (args specify '{current_size}')")
        
        model = get_siim_model(
            architecture=args.model_flag,
            in_channels=args.n_channels,
            out_channels=args.num_classes,
            model_size=detected_size
        )
    else:
        # Original MedMNIST handling
        if experiment_name in INFO:
            info = INFO[experiment_name]
        elif experiment_name in SIIM_INFO:
            info = SIIM_INFO[experiment_name]
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        task = info['task']
        
        # Create test loader, ensuring num_workers=0 to prevent deadlocks
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_evaluator = Evaluator(experiment_name, 'test', size=args.size)
        criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
        
        # Create model
        model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        )
    
    try:
        
        if model_color == "averaging":
            # Get weights from all models M0-M5 and average them
            all_model_weights = []
            for i in range(6):
                try:
                    model_weights = warehouse.get_model_weights(f"M{i}")
                    all_model_weights.append(model_weights)
                except Exception as e:
                    print(f"Warning: Could not get weights for M{i}: {e}")
            
            if all_model_weights:
                # Average the weights
                averaged_weights = {}
                for key in all_model_weights[0].keys():
                    # Get all tensors for this key
                    tensors = [weights[key] for weights in all_model_weights]
                    
                    # Check if all tensors have the same dtype and are floating point
                    first_tensor = tensors[0]
                    if all(t.dtype == first_tensor.dtype for t in tensors) and first_tensor.dtype.is_floating_point:
                        # Safe to average floating point tensors
                        averaged_weights[key] = torch.stack(tensors).mean(dim=0)
                    else:
                        # For non-floating point tensors (like indices) or mixed dtypes, use the first model's weights
                        averaged_weights[key] = first_tensor.clone()
                        # Only warn for unexpected non-floating point tensors (skip common BatchNorm counters)
                        if not first_tensor.dtype.is_floating_point and not key.endswith('num_batches_tracked'):
                            print(f"Warning: Cannot average non-floating point tensor '{key}' (dtype: {first_tensor.dtype}). Using first model's weights.")
                
                # Load averaged weights with error handling for prefix mismatches
                try:
                    model.load_state_dict(averaged_weights)
                except RuntimeError as e:
                    if "Missing key(s) in state_dict" in str(e):
                        print(f"Warning: State dict mismatch in averaging. Attempting to fix key prefixes...")
                        averaged_weights = fix_state_dict_prefixes(averaged_weights, model)
                        model.load_state_dict(averaged_weights, strict=False)
                    else:
                        raise e
                
                # Save averaged model to disk and clean up GPU memory
                model_path = save_server_model_to_disk(model, model_color, getattr(args, 'current_round', 0), args.output_dir)
                
                # Load model back for evaluation
                model = load_server_model_from_disk(model_path, args, experiment_name, device)
            else:
                # Fallback to M0 if averaging fails
                model_weights = warehouse.get_model_weights("M0")
                # Load M0 weights with error handling for prefix mismatches
                try:
                    model.load_state_dict(model_weights)
                except RuntimeError as e:
                    if "Missing key(s) in state_dict" in str(e):
                        print(f"Warning: State dict mismatch in M0 fallback. Attempting to fix key prefixes...")
                        model_weights = fix_state_dict_prefixes(model_weights, model)
                        model.load_state_dict(model_weights, strict=False)
                    else:
                        raise e
                
                # Save M0 model to disk and clean up GPU memory
                model_path = save_server_model_to_disk(model, model_color, getattr(args, 'current_round', 0), args.output_dir)
                
                # Load model back for evaluation
                model = load_server_model_from_disk(model_path, args, experiment_name, device)
                
        elif model_color == "ensemble":
            # Implement block-wise ensemble evaluation
            # Load settings from the experiment's FBD config file
            fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
            with open(fbd_settings_path, 'r') as f:
                fbd_settings = json.load(f)
            
            ensemble_size = fbd_settings.get('ENSEMBLE_SIZE', args.num_ensemble)
            ensemble_colors_pool = fbd_settings.get('ENSEMBLE_COLORS', [])
            model_parts_pool = fbd_settings.get('MODEL_PARTS', [])
            fbd_trace = fbd_settings.get('FBD_TRACE', {})
            
            print(f"Starting block-wise ensemble evaluation for {ensemble_size} models...")

            if not all([ensemble_colors_pool, model_parts_pool, fbd_trace]):
                print("Ensemble settings missing. Falling back to M0.")
                model_weights = warehouse.get_model_weights("M0")
                model.load_state_dict(model_weights)
                model.to(device)
                if experiment_name == "siim":
                    metrics = _test_siim_model(model, test_loader, criterion, dice_metric, device)
                    return {"test_loss": metrics[0], "test_dice": metrics[1], "test_acc": metrics[2]}
                else:
                    metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
                    return {"test_loss": metrics[0], "test_auc": metrics[1], "test_acc": metrics[2]}

            # Create a reverse map for easy lookup: (part, color) -> block_id
            part_color_to_block_id = {
                (info['model_part'], info['color']): block_id
                for block_id, info in fbd_trace.items()
            }

            # Generate ensemble predictions
            import random
            from scipy.stats import mode
            import numpy as np
            from tqdm import tqdm
            
            all_y_scores = []
            print(f"Generating {ensemble_size} hybrid models for ensemble evaluation...")
            
            # Temporarily suppress model creation logging during ensemble generation
            model_logger_level = logging.getLogger('fbd_model_ckpt').level
            logging.getLogger('fbd_model_ckpt').setLevel(logging.ERROR)

            for i in tqdm(range(ensemble_size), desc="Generating hybrid models"):
                # Create a random hybrid model configuration
                hybrid_config = {part: random.choice(ensemble_colors_pool) for part in model_parts_pool}
                
                # Assemble weights for the hybrid model
                hybrid_weights = {}
                is_valid_config = True
                for part, color in hybrid_config.items():
                    block_id = part_color_to_block_id.get((part, color))
                    if block_id is None:
                        print(f"Could not find block_id for part '{part}' and color '{color}'. Skipping this hybrid model.")
                        is_valid_config = False
                        break
                    
                    block_weights = warehouse.retrieve_weights(block_id)
                    if not block_weights:
                        print(f"Could not retrieve weights for block '{block_id}'. Skipping this hybrid model.")
                        is_valid_config = False
                        break
                    
                    hybrid_weights.update(block_weights)
                
                if not is_valid_config:
                    continue
                
                # Create and evaluate hybrid model
                if experiment_name == "siim":
                    # Use the same detected size for consistency
                    detected_size = detect_model_size_from_state_dict(hybrid_weights)
                    hybrid_model = get_siim_model(
                        architecture=args.model_flag,
                        in_channels=args.n_channels,
                        out_channels=args.num_classes,
                        model_size=detected_size
                    )
                else:
                    hybrid_model = get_pretrained_fbd_model(
                        architecture=args.model_flag,
                        norm=args.norm, 
                        in_channels=args.in_channels, 
                        num_classes=args.num_classes,
                        use_pretrained=False
                    )
                hybrid_model.load_state_dict(hybrid_weights)
                hybrid_model.to(device)
                
                # Get scores from hybrid model
                y_score = _get_scores(hybrid_model, test_loader, task, device)
                all_y_scores.append(y_score)
                
                # Clean up hybrid model to free GPU memory
                hybrid_model.cpu()
                del hybrid_model
                torch.cuda.empty_cache()
                gc.collect()

            # Restore original logging level
            logging.getLogger('fbd_model_ckpt').setLevel(model_logger_level)
            
            if not all_y_scores:
                print("No valid hybrid model predictions were generated. Falling back to M0.")
                model_weights = warehouse.get_model_weights("M0")
                model.load_state_dict(model_weights)
                
                # Save M0 model to disk and clean up GPU memory
                model_path = save_server_model_to_disk(model, model_color, getattr(args, 'current_round', 0), args.output_dir)
                
                # Load model back for evaluation
                model = load_server_model_from_disk(model_path, args, experiment_name, device)
                
                metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
                return {"test_loss": metrics[0], "test_auc": metrics[1], "test_acc": metrics[2]}

            # Calculate ensemble metrics using majority voting
            all_y_scores_array = np.array(all_y_scores)
            
            # Handle different task types
            if task == 'segmentation':
                # For segmentation tasks (like SIIM), we can't easily do ensemble voting on segmentation masks
                # because they're too large and complex. Instead, we'll use averaging of scores.
                # For SIIM, we'll skip the complex voting and just use averaged scores for evaluation
                print("Segmentation task: Using averaged scores for ensemble evaluation")
                majority_vote_accuracy = 0.0  # Not meaningful for segmentation
                mean_member_accuracy = 0.0
                mean_confidence = 0.0
                min_confidence = 0.0
                max_confidence = 0.0
                high_confidence_samples = 0
                medium_confidence_samples = 0
                low_confidence_samples = 0
                
            elif task == 'multi-label, binary-class':
                # For multi-label, we need to handle each label independently
                # all_y_scores shape: (num_models, num_samples, num_classes)
                # true_labels shape: (num_samples, num_classes)
                true_labels = test_dataset.labels.squeeze()
                if len(true_labels.shape) == 1:
                    true_labels = true_labels.reshape(-1, 1)
                
                # Convert sigmoid outputs to binary predictions for each label
                member_predictions = (all_y_scores_array > 0.5).astype(int)
                # Shape: (num_models, num_samples, num_classes)
                
                # Calculate majority vote for each label independently
                # Mean across models, then threshold at 0.5
                ensemble_predictions = np.mean(member_predictions, axis=0) > 0.5
                # Shape: (num_samples, num_classes)
                
                # Calculate accuracy (exact match - all labels must match)
                exact_match = np.all(ensemble_predictions == true_labels, axis=1)
                num_correct_majority = np.sum(exact_match)
                num_samples = len(true_labels)
                majority_vote_accuracy = num_correct_majority / num_samples if num_samples > 0 else 0
                
                # Also calculate Hamming accuracy (per-label accuracy)
                hamming_accuracy = np.mean(ensemble_predictions == true_labels)
                
                mean_member_accuracy = hamming_accuracy  # Use Hamming accuracy for multi-label
                
            elif task == 'binary-class':
                # For binary classification, scores are shape (num_models, num_samples, 1) or (num_models, num_samples, 2)
                true_labels = test_dataset.labels.flatten()
                # Convert outputs to binary predictions
                if len(all_y_scores_array.shape) == 3:
                    if all_y_scores_array.shape[2] == 1:
                        # Single output (sigmoid)
                        member_predictions = (all_y_scores_array.squeeze(axis=2) > 0.5).astype(int)
                    else:
                        # Two outputs (softmax) - take argmax
                        member_predictions = np.argmax(all_y_scores_array, axis=2)
                else:
                    # 2D array (num_models, num_samples)
                    member_predictions = (all_y_scores_array > 0.5).astype(int)
            else:
                # For multi-class, scores are shape (num_models, num_samples, num_classes)
                true_labels = test_dataset.labels.flatten()
                member_predictions = np.argmax(all_y_scores_array, axis=2)
            
            # Skip single-label voting logic for multi-label tasks and segmentation
            if task == 'multi-label, binary-class' or task == 'segmentation':
                # Multi-label metrics and segmentation metrics were already calculated above
                if task == 'multi-label, binary-class':
                    mean_confidence = 0.0
                    min_confidence = 0.0
                    max_confidence = 0.0
                    high_confidence_samples = 0
                    medium_confidence_samples = 0
                    low_confidence_samples = 0
                # For segmentation, the values were already set above
            else:
                # Debug shapes
                print(f"Debug - all_y_scores_array shape: {all_y_scores_array.shape}")
                print(f"Debug - member_predictions shape before processing: {member_predictions.shape}")
                print(f"Debug - true_labels shape: {true_labels.shape}")
                
                # Ensure member_predictions is 2D for voting
                if len(member_predictions.shape) == 1:
                    # If member_predictions is 1D, it means we only have 1 model
                    member_predictions = member_predictions.reshape(1, -1)
                elif len(member_predictions.shape) > 2:
                    print(f"ERROR: member_predictions has unexpected shape {member_predictions.shape}")
                    # This shouldn't happen after our fixes above, but just in case
                    raise ValueError(f"member_predictions should be 2D but has shape {member_predictions.shape}")
                
                votes_by_sample = member_predictions.T  # Shape: (num_samples, num_ensemble_members)
                
                # Calculate majority vote for each sample and voting confidence
                majority_votes = []
                vote_confidences = []
                total_ensemble_members = votes_by_sample.shape[1]
                
                for sample_idx in range(votes_by_sample.shape[0]):
                    sample_votes = votes_by_sample[sample_idx]
                    
                    # Count votes for each class
                    unique_votes, vote_counts = np.unique(sample_votes, return_counts=True)
                    
                    # Find the majority vote (class with most votes)
                    majority_class_idx = np.argmax(vote_counts)
                    majority_class = unique_votes[majority_class_idx]
                    majority_count = vote_counts[majority_class_idx]
                    
                    majority_votes.append(majority_class)
                    
                    # Calculate confidence as ratio of majority votes to total votes
                    confidence = majority_count / total_ensemble_members
                    vote_confidences.append(confidence)
                
                majority_votes = np.array(majority_votes)
                vote_confidences = np.array(vote_confidences)
                
                # Debug voting results
                print(f"Debug - votes_by_sample shape: {votes_by_sample.shape}")
                print(f"Debug - majority_votes shape: {majority_votes.shape}")
                print(f"Debug - vote_confidences shape: {vote_confidences.shape}")
                
                # Ensure shapes match before comparison
                if len(majority_votes) != len(true_labels):
                    print(f"ERROR: Shape mismatch - majority_votes: {majority_votes.shape}, true_labels: {true_labels.shape}")
                    # Fall back to a safe default
                    majority_vote_accuracy = 0.0
                    mean_confidence = 0.0
                    min_confidence = 0.0
                    max_confidence = 0.0
                    high_confidence_samples = 0
                    medium_confidence_samples = 0
                    low_confidence_samples = 0
                    mean_member_accuracy = 0.0
                    num_correct_majority = 0
                else:
                    # Calculate accuracy
                    num_correct_majority = np.sum(majority_votes == true_labels)
                    num_samples = len(true_labels)
                    majority_vote_accuracy = num_correct_majority / num_samples if num_samples > 0 else 0
                    
                    # Calculate voting statistics
                    mean_confidence = np.mean(vote_confidences)
                    min_confidence = np.min(vote_confidences)
                    max_confidence = np.max(vote_confidences)
                    
                    # Count samples by confidence level
                    high_confidence_samples = np.sum(vote_confidences >= 0.8)  # 80%+ agreement
                    medium_confidence_samples = np.sum((vote_confidences >= 0.6) & (vote_confidences < 0.8))  # 60-80% agreement
                    low_confidence_samples = np.sum(vote_confidences < 0.6)  # <60% agreement
                    
                    # Calculate mean member accuracy for diagnostics
                    print(f"Debug - member_predictions final shape: {member_predictions.shape}")
                    print(f"Debug - true_labels shape: {true_labels.shape}")
                    
                    # Fix broadcasting issue: member_predictions is (num_models, num_samples), true_labels is (num_samples,)
                    # We need to reshape true_labels to (1, num_samples) to broadcast correctly
                    true_labels_reshaped = true_labels.reshape(1, -1)
                    print(f"Debug - true_labels_reshaped shape: {true_labels_reshaped.shape}")
                    
                    # Ensure shapes are compatible for broadcasting
                    if member_predictions.shape[1] != true_labels_reshaped.shape[1]:
                        print(f"Warning: Shape mismatch - member_predictions: {member_predictions.shape}, true_labels: {true_labels_reshaped.shape}")
                        mean_member_accuracy = 0.0
                    else:
                        num_correct_individual = np.sum(member_predictions == true_labels_reshaped)
                        total_individual_votes = member_predictions.size
                        mean_member_accuracy = num_correct_individual / total_individual_votes if total_individual_votes > 0 else 0

            if task == 'segmentation':
                print(f"Ensemble complete: {len(all_y_scores)} hybrid models for segmentation task")
                # For segmentation, we can't use the standard evaluator, so return simplified metrics
                return {
                    "test_loss": 0.0,  # Loss not meaningful for ensemble 
                    "test_auc": 0.0,   # Not meaningful for segmentation ensemble
                    "test_acc": 0.0    # Majority vote not meaningful for segmentation masks
                }
            elif task == 'multi-label, binary-class':
                print(f"Ensemble complete: {len(all_y_scores)} hybrid models, Majority Vote Acc: {majority_vote_accuracy:.4f}, Hamming Acc: {hamming_accuracy:.4f}")
            else:
                print(f"Ensemble complete: {total_ensemble_members} hybrid models, Majority Vote Acc: {majority_vote_accuracy:.4f}, Mean Confidence: {mean_confidence:.3f}")

            # Average the scores and evaluate (for loss and AUC metrics) - only for non-segmentation tasks
            if task != 'segmentation':
                averaged_scores = np.mean(all_y_scores, axis=0)
                
                # Create evaluator and calculate metrics
                auc, _ = test_evaluator.evaluate(averaged_scores, None, None)
                
                # For ensemble, we'll use majority vote accuracy and averaged AUC
                return {
                    "test_loss": 0.0,  # Loss not meaningful for ensemble 
                    "test_auc": auc,
                    "test_acc": majority_vote_accuracy
                }
        else:
            # Get weights for specific model color (M0, M1, M2, etc.)
            model_weights = warehouse.get_model_weights(model_color)
            if model_weights:
                model.load_state_dict(model_weights)
                # Debug: Check a sample weight to see if models are actually different
                sample_param = next(iter(model_weights.values()))
                param_sum = float(sample_param.sum()) if hasattr(sample_param, 'sum') else 0
                print(f"  Evaluating {model_color}: loaded {len(model_weights)} parameters, sample param sum: {param_sum:.6f}")
                
                # Save individual model to disk and clean up GPU memory
                model_path = save_server_model_to_disk(model, model_color, getattr(args, 'current_round', 0), args.output_dir)
                
                # Load model back for evaluation
                model = load_server_model_from_disk(model_path, args, experiment_name, device)
            else:
                print(f"  Warning: No weights found for {model_color}")
                return {"test_loss": 0.0, "test_auc": 0.0, "test_acc": 0.0}
        if experiment_name == "siim":
            metrics = _test_siim_model(model, test_loader, criterion, dice_metric, device)
            return {"test_loss": metrics[0], "test_dice": metrics[1], "test_acc": metrics[2]}
        else:
            metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
            return {"test_loss": metrics[0], "test_auc": metrics[1], "test_acc": metrics[2]}
    except Exception as e:
        print(f"Error evaluating model {model_color}: {e}")
        return {"test_loss": 0.0, "test_auc": 0.0, "test_acc": 0.0}

def initialize_server_simulation(args):
    """Initialize the server simulation environment"""
    print("Server: Initializing simulation...")
    
    # Set up cache directory if not provided
    if not args.cache_dir:
        args.cache_dir = os.path.join(os.getcwd(), "cache")
        print(f"Cache directory not set, using default: {args.cache_dir}")
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    # Initialize FBD Warehouse
    fbd_settings_path = f"config/{args.experiment_name}/fbd_settings.json"
    fbd_trace, _, _ = load_fbd_settings(fbd_settings_path)
    
    # Create model template with shared initial weights for reproducible federated learning
    if args.experiment_name == "siim":
        model_size = getattr(args, 'model_size', 'standard')
        model_template = get_siim_model(
            architecture=args.model_flag,
            in_channels=args.n_channels,
            out_channels=args.num_classes,
            model_size=model_size
        )
        
        # Handle different initialization methods
        init_method = getattr(args, 'init_method', 'pretrained')
        
        if init_method == "pretrained":
            # Try to load pretrained weights in order of preference
            weight_files_to_try = [
                f"siim_unet_pretrained_monai_{model_size}.pth",  # Medical imaging optimized
                f"siim_unet_pretrained_lungmask_{model_size}.pth",  # Lung-specific
                f"siim_unet_pretrained_chest_foundation_{model_size}.pth",  # Chest imaging
                f"siim_unet_initial_weights_{model_size}.pth"  # Basic shared weights
            ]
            
            weights_loaded = False
            for weights_file in weight_files_to_try:
                if os.path.exists(weights_file):
                    try:
                        initial_weights = torch.load(weights_file, map_location='cpu')
                        model_template.load_state_dict(initial_weights)
                        print(f"✅ Loaded pretrained weights from {weights_file}")
                        weights_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️  Could not load weights from {weights_file}: {e}")
            
            if not weights_loaded:
                print(f"📌 No pretrained weights found, falling back to shared random initialization")
                print("   All clients will start with same random weights due to shared seed")
        
        elif init_method == "shared_random":
            print(f"🎲 Using shared random initialization (seed: {getattr(args, 'seed', 42)})")
            print("   All clients will start with identical random weights")
        
        elif init_method == "random":
            print(f"🎰 Using different random initialization for each client")
            print("   Each client will start with different random weights")
        
        else:
            print(f"⚠️  Unknown initialization method: {init_method}, using shared random")
    else:
        model_template = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=True
        )
    
    warehouse = FBDWarehouse(
        fbd_trace=fbd_trace,
        model_template=model_template,
        log_file_path=f"{args.output_dir}/warehouse.log"
    )
    
    print("Server: FBD Warehouse initialized.")
    return warehouse

def load_simulation_plans(args):
    """Load shipping and update plans for simulation"""
    shipping_plan_path = f"config/{args.experiment_name}/shipping_plan.json"
    update_plan_path = f"config/{args.experiment_name}/update_plan.json"
    
    # Check if plans exist, if not generate them
    if not os.path.exists(shipping_plan_path) or not os.path.exists(update_plan_path):
        print(f"Plans not found for {args.experiment_name}. Generating plans...")
        try:
            import subprocess
            result = subprocess.run([
                "python3", "fbd_generate_plan.py", 
                "--experiment_name", args.experiment_name
            ], capture_output=True, text=True, check=True)
            print("Plans generated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error generating plans: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    # Load shipping and update plans
    with open(shipping_plan_path, 'r') as f:
        shipping_plans = json.load(f)
    
    with open(update_plan_path, 'r') as f:
        update_plans = json.load(f)
    
    return shipping_plans, update_plans

def prepare_test_dataset(args):
    """Prepare test dataset for server evaluations"""
    print("Server: Preparing test dataset for evaluations...")
    
    if args.experiment_name == "siim":
        # Load SIIM test dataset
        from fbd_dataset_siim import load_siim_data
        _, test_dataset = load_siim_data(args)
    else:
        # Original MedMNIST handling
        if args.experiment_name in INFO:
            info = INFO[args.experiment_name]
        elif args.experiment_name in SIIM_INFO:
            info = SIIM_INFO[args.experiment_name]
        else:
            raise ValueError(f"Unknown experiment: {args.experiment_name}")
        DataClass = getattr(medmnist, info['python_class'])
        # Use as_rgb setting from config instead of hardcoded rules
        as_rgb = getattr(args, 'as_rgb', False)
        # Use 3-channel normalization when in_channels=3 (for ResNet)
        if args.in_channels == 3:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        else:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
        test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, size=args.size)
    
    print("Server: Test dataset prepared.")
    return test_dataset

def collect_and_evaluate_round(round_num, args, warehouse, client_responses):
    """Collect client responses and evaluate models for a round"""
    print(f"Server: Collecting responses for round {round_num}...")
    
    round_losses = []
    
    # Process responses from all clients
    for client_id, response in client_responses.items():
        if response is None:
            continue
            
        loss = response.get("train_loss")
        updated_weights = response.get("updated_weights")
        updated_optimizer_states = response.get("updated_optimizer_states")
        trainable_block_ids = response.get("trainable_block_ids", [])
        round_losses.append(loss)
        
        if updated_weights:
            # Only store blocks that were actually trainable (trained)
            trainable_weights = {block_id: weights for block_id, weights in updated_weights.items() 
                               if block_id in trainable_block_ids}
            
            if trainable_weights:
                warehouse.store_weights_batch(trainable_weights)
                # Debug: Check which block IDs are being updated
                all_block_ids = list(updated_weights.keys())[:3]
                trainable_ids = list(trainable_weights.keys())[:3]
                print(f"Server: Received {len(updated_weights)} blocks from client {client_id} (e.g., {all_block_ids}), stored {len(trainable_weights)} trainable blocks (e.g., {trainable_ids}), loss: {loss:.4f}")
            else:
                print(f"Server: WARNING - Client {client_id} sent no trainable weights!")
        else:
            print(f"Server: WARNING - Client {client_id} sent no updated weights!")
        
        if updated_optimizer_states:
            # Only store optimizer states for trainable blocks
            trainable_optimizer_states = {block_id: state for block_id, state in updated_optimizer_states.items() 
                                        if block_id in trainable_block_ids}
            if trainable_optimizer_states:
                warehouse.store_optimizer_state_batch(trainable_optimizer_states)
    
    # Print summary
    avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
    print(f"Server: All responses for round {round_num} collected. Average loss: {avg_loss:.4f}")
    
    # Evaluate all models once at the end of the round
    print(f"Server: Evaluating all models at end of round {round_num}...")
    round_eval_results = {'round': round_num}
    
    # Store results for summary output
    model_results = []
    
    # Set current round for disk-based model management
    args.current_round = round_num
    
    # Evaluate individual models M0 to M5
    for model_idx in range(6):
        model_color = f"M{model_idx}"
        metrics = evaluate_server_model(args, model_color, args.model_flag, args.experiment_name, args.test_dataset, warehouse)
        round_eval_results[model_color] = metrics
        auc = metrics.get("test_auc", 0.0)
        acc = metrics.get("test_acc", 0.0)
        model_results.append(f"{model_color}: AUC={auc:.4f}, ACC={acc:.4f}")
    
    # Evaluate averaged model
    avg_metrics = evaluate_server_model(args, "averaging", args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    round_eval_results["averaging"] = avg_metrics
    auc = avg_metrics.get("test_auc", 0.0)
    acc = avg_metrics.get("test_acc", 0.0)
    model_results.append(f"Averaging: AUC={auc:.4f}, ACC={acc:.4f}")
    
    # Evaluate ensemble model
    ensemble_metrics = evaluate_server_model(args, "ensemble", args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    round_eval_results["ensemble"] = ensemble_metrics
    auc = ensemble_metrics.get("test_auc", 0.0)
    acc = ensemble_metrics.get("test_acc", 0.0)
    model_results.append(f"Ensemble: AUC={auc:.4f}, ACC={acc:.4f}")
    
    # Print summary
    print(f"\n🔍 Round {round_num} Evaluation:")
    for result in model_results:
        print(f"  {result}")
    print()
    
    return round_eval_results

def get_client_plans_for_round(round_num, client_id, shipping_plans, update_plans):
    """Get shipping and update plans for a specific client and round"""
    round_shipping_plan = shipping_plans.get(str(round_num + 1), {})
    round_update_plan = update_plans.get(str(round_num + 1), {})
    
    client_shipping_list = round_shipping_plan.get(str(client_id), [])
    client_update_plan = round_update_plan.get(str(client_id), {})
    
    return client_shipping_list, client_update_plan