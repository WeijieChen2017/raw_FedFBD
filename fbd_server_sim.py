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

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import load_fbd_settings, FBDWarehouse
from fbd_dataset import DATASET_SPECIFIC_RULES

# Suppress logging from fbd_model_ckpt to reduce noise
logging.getLogger('fbd_model_ckpt').setLevel(logging.WARNING)

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

def evaluate_server_model(args, model_color, model_flag, experiment_name, test_dataset, warehouse):
    """Evaluate a server model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[experiment_name]
    task = info['task']
    
    # Create test loader
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    test_evaluator = Evaluator(experiment_name, 'test', size=args.size)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    
    try:
        # Create a fresh model instance
        model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False  # Don't use pretrained weights, we'll load from warehouse
        )
        
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
                
                model.load_state_dict(averaged_weights)
            else:
                # Fallback to M0 if averaging fails
                model_weights = warehouse.get_model_weights("M0")
                model.load_state_dict(model_weights)
                
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

            # Restore original logging level
            logging.getLogger('fbd_model_ckpt').setLevel(model_logger_level)
            
            if not all_y_scores:
                print("No valid hybrid model predictions were generated. Falling back to M0.")
                model_weights = warehouse.get_model_weights("M0")
                model.load_state_dict(model_weights)
                model.to(device)
                metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
                return {"test_loss": metrics[0], "test_auc": metrics[1], "test_acc": metrics[2]}

            # Calculate ensemble metrics using majority voting
            all_y_scores_array = np.array(all_y_scores)
            
            # Handle different task types
            if task == 'multi-label, binary-class':
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
                # For binary classification, scores are shape (num_models, num_samples, 1) or (num_models, num_samples)
                true_labels = test_dataset.labels.flatten()
                # Convert sigmoid outputs to binary predictions
                # Handle both shapes: (num_models, num_samples, 1) and (num_models, num_samples)
                if len(all_y_scores_array.shape) == 3 and all_y_scores_array.shape[2] == 1:
                    member_predictions = (all_y_scores_array.squeeze(axis=2) > 0.5).astype(int)
                else:
                    member_predictions = (all_y_scores_array > 0.5).astype(int)
            else:
                # For multi-class, scores are shape (num_models, num_samples, num_classes)
                true_labels = test_dataset.labels.flatten()
                member_predictions = np.argmax(all_y_scores_array, axis=2)
            
            # Skip single-label voting logic for multi-label tasks
            if task == 'multi-label, binary-class':
                # Multi-label metrics were already calculated above
                mean_confidence = 0.0
                min_confidence = 0.0
                max_confidence = 0.0
                high_confidence_samples = 0
                medium_confidence_samples = 0
                low_confidence_samples = 0
            else:
                # Debug shapes
                print(f"Debug - all_y_scores_array shape: {all_y_scores_array.shape}")
                print(f"Debug - member_predictions shape before transpose: {member_predictions.shape}")
                print(f"Debug - true_labels shape: {true_labels.shape}")
                
                # Ensure member_predictions has the right shape
                if len(member_predictions.shape) == 1:
                    # If member_predictions is 1D, it means we only have 1 model
                    member_predictions = member_predictions.reshape(1, -1)
                
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

            if task == 'multi-label, binary-class':
                print(f"Ensemble complete: {len(all_y_scores)} hybrid models, Majority Vote Acc: {majority_vote_accuracy:.4f}, Hamming Acc: {hamming_accuracy:.4f}")
            else:
                print(f"Ensemble complete: {total_ensemble_members} hybrid models, Majority Vote Acc: {majority_vote_accuracy:.4f}, Mean Confidence: {mean_confidence:.3f}")

            # Average the scores and evaluate (for loss and AUC metrics)
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
            else:
                print(f"  Warning: No weights found for {model_color}")
                return {"test_loss": 0.0, "test_auc": 0.0, "test_acc": 0.0}
        
        model.to(device)
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
    
    # Create model template directly without loading from file
    # In simulation, we don't need to save/load - just create fresh each time
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
    info = INFO[args.experiment_name]
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