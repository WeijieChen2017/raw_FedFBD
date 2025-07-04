#!/usr/bin/env python3

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
import logging

import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import load_fbd_settings, FBDWarehouse, setup_logger
from fbd_dataset import DATASET_SPECIFIC_RULES

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

def _test_model_with_targets(model, evaluator, data_loader, task, device):
    """Proper evaluation function that processes scores correctly (fixed target handling)"""
    model.eval()
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_score = torch.cat((y_score, outputs), 0)

    y_score = y_score.detach().cpu().numpy()
    
    # MedMNIST evaluator loads true labels internally - just pass None, None
    auc, acc = evaluator.evaluate(y_score, None, None)
    return auc, acc

def diagnose_warehouse_weights(warehouse, logger, final_test_colors):
    """Diagnostic function to check warehouse weights and compare with server simulation approach"""
    logger.info("=== WAREHOUSE DIAGNOSTICS ===")
    
    # Check if we can retrieve weights for each color
    for color in final_test_colors:
        try:
            weights = warehouse.get_model_weights(color)
            if weights:
                # Get a sample parameter to check if weights are reasonable
                sample_param = next(iter(weights.values()))
                param_sum = float(sample_param.sum()) if hasattr(sample_param, 'sum') else 0
                param_mean = float(sample_param.mean()) if hasattr(sample_param, 'mean') else 0
                param_std = float(sample_param.std()) if hasattr(sample_param, 'std') else 0
                logger.info(f"  {color}: ✅ {len(weights)} params, sample stats: sum={param_sum:.6f}, mean={param_mean:.6f}, std={param_std:.6f}")
            else:
                logger.warning(f"  {color}: ❌ No weights found")
        except Exception as e:
            logger.error(f"  {color}: ❌ Error retrieving weights: {e}")
    
    logger.info("=== END DIAGNOSTICS ===")

def perform_final_ensemble_prediction(args):
    """
    Performs final ensemble prediction on test data and returns detailed results.
    
    Returns:
        pandas.DataFrame: Table with columns 'sample_id', 'c_i', 'z_i', 'true_label', 'predicted_label'
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "fbd_test_sim.log")
    logger = setup_logger("FBD_Test", log_file)
    logger.info("Starting final ensemble prediction...")
    logger.info("🔧 FIXED: Now properly processing targets during forward pass (was ignoring targets completely)")
    
    # Load warehouse - check for both standard and round-specific names
    warehouse_path = os.path.join(args.comm_dir, "fbd_warehouse.pth")
    if not os.path.exists(warehouse_path):
        # Look for warehouse files with round numbers
        import glob
        warehouse_pattern = os.path.join(args.comm_dir, "fbd_warehouse_round_*.pth")
        warehouse_files = glob.glob(warehouse_pattern)
        if warehouse_files:
            # Use the latest round warehouse file
            warehouse_path = max(warehouse_files, key=lambda x: int(x.split('_round_')[1].split('.')[0]))
            logger.info(f"Using warehouse file: {warehouse_path}")
        else:
            logger.error(f"No warehouse file found in {args.comm_dir}")
            raise FileNotFoundError(f"No warehouse file found in {args.comm_dir}")
    
    # Load FBD settings
    fbd_settings_path = os.path.join("config", args.experiment_name, "fbd_settings.json")
    try:
        with open(fbd_settings_path, 'r') as f:
            fbd_settings = json.load(f)
        
        fbd_trace = fbd_settings.get('FBD_TRACE', {})
        final_test_colors = fbd_settings.get('FINAL_TEST_COLORS', [])
        
        if not final_test_colors:
            logger.warning("FINAL_TEST_COLORS not found in fbd_settings.json. Using default ensemble colors.")
            final_test_colors = fbd_settings.get('ENSEMBLE_COLORS', ['M0', 'M1', 'M2', 'M3', 'M4', 'M5'])
            
    except FileNotFoundError:
        logger.error(f"FBD settings file not found: {fbd_settings_path}")
        raise
    
    # Initialize warehouse
    model_template = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=args.norm,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        use_pretrained=False
    )
    
    warehouse = FBDWarehouse(
        fbd_trace=fbd_trace,
        model_template=model_template,
        log_file_path=os.path.join(args.comm_dir, "warehouse.log")
    )
    warehouse.load_warehouse(warehouse_path)
    logger.info(f"Loaded warehouse from {warehouse_path}")
    
    # Add diagnostic check
    diagnose_warehouse_weights(warehouse, logger, final_test_colors)
    
    # Prepare test dataset
    info = INFO[args.experiment_name]
    DataClass = getattr(medmnist, info['python_class'])
    dataset_rules = DATASET_SPECIFIC_RULES.get(args.experiment_name, {})
    as_rgb = dataset_rules.get("as_rgb", False)
    
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
    
    test_dataset = DataClass(
        split='test', 
        transform=data_transform, 
        download=True, 
        as_rgb=as_rgb, 
        size=args.size
    )
    
    test_loader = data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Setup device and task
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    task = info['task']
    
    logger.info(f"Using device: {device}")
    logger.info(f"Task: {task}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Ensemble colors: {final_test_colors}")
    
    # Get predictions from each model in the ensemble
    logger.info("Gathering predictions from ensemble models...")
    all_model_scores = []
    
    # First, add individual color models
    for model_color in final_test_colors:
        logger.info(f"Processing model {model_color}...")
        
        model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        ).to(device)
        
        model_weights = warehouse.get_model_weights(model_color)
        if not model_weights:
            logger.warning(f"No weights found for model {model_color}. Skipping.")
            continue
            
        model.load_state_dict(model_weights)
        scores = _get_scores(model, test_loader, task, device)
        all_model_scores.append(scores)
        
        # Evaluate individual model using MedMNIST evaluator
        test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
        individual_auc, individual_acc = _test_model_with_targets(model, test_evaluator, test_loader, task, device)
        logger.info(f"Individual {model_color} - AUC: {individual_auc:.4f}, ACC: {individual_acc:.4f}")
        
        logger.info(f"Got predictions from {model_color}, shape: {scores.shape}")
    
    # Add the averaging model
    logger.info("Creating and evaluating averaged model from all colors...")
    all_color_models = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']  # All possible colors
    all_model_weights = []
    
    for color in all_color_models:
        weights = warehouse.get_model_weights(color)
        if weights:
            all_model_weights.append(weights)
        else:
            logger.warning(f"No weights found for color {color} in averaging")
    
    if len(all_model_weights) > 0:
        # Average the weights
        averaged_weights = {}
        param_keys = all_model_weights[0].keys()
        
        for key in param_keys:
            if all_model_weights[0][key].is_floating_point():
                # Average floating point parameters
                averaged_weights[key] = torch.stack([w[key] for w in all_model_weights]).mean(dim=0)
            else:
                # For non-floating point tensors (e.g., BatchNorm counters), use first model
                averaged_weights[key] = all_model_weights[0][key]
        
        # Create averaged model and get predictions
        averaged_model = get_pretrained_fbd_model(
            architecture=args.model_flag,
            norm=args.norm,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            use_pretrained=False
        ).to(device)
        
        averaged_model.load_state_dict(averaged_weights)
        averaged_scores = _get_scores(averaged_model, test_loader, task, device)
        all_model_scores.append(averaged_scores)
        
        # Evaluate averaged model using MedMNIST evaluator
        test_evaluator_avg = Evaluator(args.experiment_name, 'test', size=args.size)
        avg_auc, avg_acc = _test_model_with_targets(averaged_model, test_evaluator_avg, test_loader, task, device)
        logger.info(f"Individual Averaging model - AUC: {avg_auc:.4f}, ACC: {avg_acc:.4f}")
        
        logger.info(f"Got predictions from averaged model, shape: {averaged_scores.shape}")
        logger.info(f"Averaged {len(all_model_weights)} models: {[f'M{i}' for i in range(len(all_model_weights))]}")

    if not all_model_scores:
        logger.error("No valid model predictions gathered. Aborting.")
        raise RuntimeError("No valid model predictions gathered.")
    
    # Calculate ensemble predictions using majority vote
    logger.info("Calculating ensemble predictions...")
    
    # Convert scores to predictions (argmax for each model)
    votes = np.stack([np.argmax(scores, axis=1) for scores in all_model_scores], axis=1)
    # votes shape: (num_samples, num_models)
    
    true_labels = test_dataset.labels.squeeze()
    num_samples = len(true_labels)
    num_models = votes.shape[1]
    
    # Also evaluate using MedMNIST evaluator for comparison
    test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
    
    # Calculate ensemble average scores for MedMNIST evaluation
    ensemble_avg_scores = np.mean(all_model_scores, axis=0)
    # MedMNIST evaluator loads true labels internally - just pass None, None
    medmnist_auc, medmnist_acc = test_evaluator.evaluate(ensemble_avg_scores, None, None)
    
    logger.info(f"MedMNIST Evaluator results:")
    logger.info(f"  Ensemble Avg AUC: {medmnist_auc:.4f}")
    logger.info(f"  Ensemble Avg ACC: {medmnist_acc:.4f}")
    
    # Save predictions for analysis
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save individual model predictions
    for i, (model_color, scores) in enumerate(zip(final_test_colors + ['averaging'], all_model_scores)):
        pred_file = os.path.join(predictions_dir, f"{model_color}_predictions.csv")
        pred_df = pd.DataFrame(scores)
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Saved {model_color} predictions to {pred_file}")
    
    # Save ensemble average predictions
    ensemble_pred_file = os.path.join(predictions_dir, "ensemble_avg_predictions.csv")
    ensemble_pred_df = pd.DataFrame(ensemble_avg_scores)
    ensemble_pred_df.to_csv(ensemble_pred_file, index=False)
    logger.info(f"Saved ensemble average predictions to {ensemble_pred_file}")
    
    # Calculate c_i (confidence) and z_i (correctness) for each sample
    results_data = []
    
    for i in range(num_samples):
        sample_votes = votes[i, :]
        vote_counts = Counter(sample_votes)
        
        # Get majority vote
        majority_class, majority_count = vote_counts.most_common(1)[0]
        
        # Calculate c_i: ratio of majority votes
        c_i = majority_count / num_models
        
        # Calculate z_i: 1 if majority vote matches true label, 0 otherwise
        z_i = 1 if majority_class == true_labels[i] else 0
        
        results_data.append({
            'sample_id': i,
            'c_i': c_i,
            'z_i': z_i,
            'true_label': int(true_labels[i]),
            'predicted_label': int(majority_class),
            'majority_count': majority_count,
            'total_votes': num_models
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Calculate summary statistics
    overall_accuracy = results_df['z_i'].mean()
    mean_confidence = results_df['c_i'].mean()
    
    logger.info(f"Final ensemble results:")
    logger.info(f"  Majority Vote accuracy: {overall_accuracy:.4f}")
    logger.info(f"  MedMNIST Ensemble Avg accuracy: {medmnist_acc:.4f}")
    logger.info(f"  Mean confidence: {mean_confidence:.4f}")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Number of models: {num_models}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="FBD Final Ensemble Test")
    parser.add_argument("--experiment_name", type=str, required=True, 
                       help="Name of the experiment")
    parser.add_argument("--model_flag", type=str, required=True, 
                       help="Model architecture (e.g., resnet18)")
    parser.add_argument("--norm", type=str, required=True, 
                       help="Normalization type")
    parser.add_argument("--in_channels", type=int, required=True, 
                       help="Number of input channels")
    parser.add_argument("--num_classes", type=int, required=True, 
                       help="Number of output classes")
    parser.add_argument("--size", type=int, default=224, 
                       help="Image size")
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="Batch size for evaluation")
    parser.add_argument("--comm_dir", type=str, default="fbd_comm", 
                       help="Communication directory containing warehouse")
    parser.add_argument("--output_dir", type=str, default="final_results", 
                       help="Output directory for results")
    parser.add_argument("--final_ensemble", action="store_true", 
                       help="Perform final ensemble prediction")
    
    args = parser.parse_args()
    
    if not args.final_ensemble:
        print("Use --final_ensemble flag to perform final ensemble prediction")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Perform final ensemble prediction
        results_df = perform_final_ensemble_prediction(args)
        
        # Save results to CSV
        results_csv_path = os.path.join(args.output_dir, "final_ensemble_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")
        
        # Save detailed results to JSON
        results_json_path = os.path.join(args.output_dir, "final_ensemble_results.json")
        
        # Convert DataFrame to dict and ensure all values are JSON serializable
        per_sample_results = []
        for _, row in results_df.iterrows():
            per_sample_results.append({
                'sample_id': int(row['sample_id']),
                'c_i': float(row['c_i']),
                'z_i': int(row['z_i']),
                'true_label': int(row['true_label']),
                'predicted_label': int(row['predicted_label']),
                'majority_count': int(row['majority_count']),
                'total_votes': int(row['total_votes'])
            })
        
        results_dict = {
            'summary': {
                'overall_accuracy': float(results_df['z_i'].mean()),
                'mean_confidence': float(results_df['c_i'].mean()),
                'num_samples': len(results_df),
                'num_models': int(results_df['total_votes'].iloc[0]) if len(results_df) > 0 else 0
            },
            'per_sample_results': per_sample_results
        }
        
        with open(results_json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Detailed results saved to: {results_json_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("FINAL ENSEMBLE RESULTS SUMMARY")
        print("="*50)
        print(f"Majority Vote Accuracy: {results_df['z_i'].mean():.4f}")
        print(f"Mean Confidence: {results_df['c_i'].mean():.4f}")
        print(f"Number of samples: {len(results_df)}")
        print(f"Number of models: {results_df['total_votes'].iloc[0] if len(results_df) > 0 else 0}")
        print(f"\nFor comparison - MedMNIST Ensemble Avg Accuracy will be logged in the log file.")
        
        # Show first few samples
        print("\nFirst 10 samples:")
        print(results_df[['sample_id', 'c_i', 'z_i', 'true_label', 'predicted_label']].head(10))
        
    except Exception as e:
        print(f"Error during final ensemble prediction: {e}")
        raise

if __name__ == "__main__":
    main()