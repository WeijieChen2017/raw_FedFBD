import argparse
import os
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np

import medmnist
from medmnist import INFO, Evaluator

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import load_config, setup_logger, handle_weights_cache
from fbd_dataset import DATASET_SPECIFIC_RULES, load_data

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
    # For this standalone script, we don't need to save the scores to a file.
    auc, acc = evaluator.evaluate(y_score, save_folder=None, run_name=None) 
    test_loss = sum(total_loss) / len(total_loss)
    return [test_loss, auc, acc]


def main():
    parser = argparse.ArgumentParser(description="Standalone training and evaluation script.")
    parser.add_argument("--experiment_name", type=str, default="bloodmnist", help="Name of the experiment dataset.")
    parser.add_argument("--model_flag", type=str, default="resnet18", help="Model architecture.")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Path to the model and weights cache.")
    parser.add_argument("--output_dir", type=str, default="fbd_single_run_results", help="Directory to save results.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()

    # --- 1. Setup ---
    # Load config and merge into args
    config = load_config(args.experiment_name, args.model_flag)
    args_dict = vars(args)
    args_dict.update(vars(config))
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_dir = os.path.join(args.output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = setup_logger("SingleRun", os.path.join(log_dir, "run.log"))
    logger.info("Starting standalone training run.")
    logger.info(f"Arguments: {args}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- 2. Data Loading ---
    logger.info("Loading data...")
    train_dataset, test_dataset = load_data(args)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # --- 3. Model Preparation ---
    logger.info(f"Preparing model '{args.model_flag}' with ImageNet weights.")
    
    sync_weights_back = handle_weights_cache(args.model_flag, args.cache_dir)

    model = get_pretrained_fbd_model(
        architecture=args.model_flag,
        norm=args.norm,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        use_pretrained=True
    )
    
    if sync_weights_back:
        sync_weights_back()
    
    model.to(device)

    # --- 4. Training ---
    info = INFO[args.experiment_name]
    task = info['task']
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                 targets = targets.to(torch.float32)
                 loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        logger.info(f"--- Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} ---")

    logger.info("Training finished.")

    # --- 5. Evaluation ---
    logger.info("Starting final evaluation...")
    test_evaluator = Evaluator(args.experiment_name, 'test', size=args.size)
    test_metrics = _test_model(model, test_evaluator, test_loader, task, criterion, device)
    
    logger.info(f"Evaluation complete.")
    logger.info(f"  └─ Test Loss: {test_metrics[0]:.5f}, Test AUC: {test_metrics[1]:.5f}, Test Acc: {test_metrics[2]:.5f}")
    print(f"\nFinal Results:\n  Test Loss = {test_metrics[0]:.5f}\n  Test AUC  = {test_metrics[1]:.5f}\n  Test Acc  = {test_metrics[2]:.5f}")

    # --- 6. Save Results ---
    results_dict = {
        "model_name": args.model_flag,
        "dataset": args.experiment_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "test_loss": test_metrics[0], 
        "test_auc": test_metrics[1], 
        "test_acc": test_metrics[2]
    }
    
    results_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Final metrics saved to {results_path}")

    model_path = os.path.join(args.output_dir, f"{args.model_flag}_{args.experiment_name}_final.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Trained model saved to {model_path}")


if __name__ == '__main__':
    main() 