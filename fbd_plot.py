import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import argparse
import glob

def is_valid_metric(value):
    """Check if a metric is a valid number for plotting."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))

def generate_plots(experiment_dir):
    """
    Generates and saves plots for server-side and client-side metrics from a given experiment directory.

    Args:
        experiment_dir (str): The path to the experiment's output directory.
    """
    print(f"Generating plots for experiment: {experiment_dir}")
    save_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # --- Plot Server-Side Metrics ---
    json_path = os.path.join(experiment_dir, 'eval_results', 'server_evaluation_history.json')
    if not os.path.exists(json_path):
        print(f"Error: Server evaluation file not found at {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    metrics = defaultdict(lambda: defaultdict(list))
    for round_data in data:
        for model_name, model_data in round_data.items():
            if isinstance(model_data, dict):
                metrics[model_name]['test_loss'].append(model_data.get('test_loss'))
                metrics[model_name]['test_acc'].append(model_data.get('test_acc'))
                metrics[model_name]['test_auc'].append(model_data.get('test_auc'))
                metrics[model_name]['majority_vote_ratio'].append(model_data.get('majority_vote_ratio'))

    rounds = [item['round'] + 1 for item in data]

    # Plot 1: Loss for M0 to M5
    plt.figure(figsize=(12, 8))
    for i in range(6):
        model = f'M{i}'
        if model in metrics:
            valid_rounds = [r for r, v in zip(rounds, metrics[model]['test_loss']) if is_valid_metric(v)]
            valid_losses = [v for v in metrics[model]['test_loss'] if is_valid_metric(v)]
            if valid_losses:
                plt.plot(valid_rounds, valid_losses, label=model)
    plt.title(f'Server Model Test Loss\n(Experiment: {os.path.basename(experiment_dir)})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'server_loss.png'))
    plt.close()
    print("Generated server_loss.png")

    # Plot 2: Accuracy for all models
    plt.figure(figsize=(12, 8))
    models_to_plot = [f'M{i}' for i in range(6)] + ['averaging', 'ensemble']
    for model in models_to_plot:
        if model in metrics:
            valid_rounds = [r for r, v in zip(rounds, metrics[model]['test_acc']) if is_valid_metric(v)]
            valid_accs = [v for v in metrics[model]['test_acc'] if is_valid_metric(v)]
            if valid_accs:
                plt.plot(valid_rounds, valid_accs, label=model)
    plt.title(f'Server Model Test Accuracy\n(Experiment: {os.path.basename(experiment_dir)})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'server_accuracy.png'))
    plt.close()
    print("Generated server_accuracy.png")

    # Plot 3: AUC for all models
    plt.figure(figsize=(12, 8))
    for model in models_to_plot:
        if model in metrics:
            valid_rounds = [r for r, v in zip(rounds, metrics[model]['test_auc']) if is_valid_metric(v)]
            valid_aucs = [v for v in metrics[model]['test_auc'] if is_valid_metric(v)]
            if valid_aucs:
                plt.plot(valid_rounds, valid_aucs, label=model)
    plt.title(f'Server Model Test AUC\n(Experiment: {os.path.basename(experiment_dir)})')
    plt.xlabel('Epoch')
    plt.ylabel('Test AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'server_auc.png'))
    plt.close()
    print("Generated server_auc.png")

    # Plot 4: Ensemble accuracy with uncertainty
    plt.figure(figsize=(12, 8))
    if 'ensemble' in metrics and 'majority_vote_ratio' in metrics['ensemble']:
        ensemble_acc_raw = metrics['ensemble']['test_acc']
        majority_ratio_raw = metrics['ensemble']['majority_vote_ratio']
        plot_rounds, plot_accs, lower_bounds, upper_bounds = [], [], [], []

        for i, r in enumerate(rounds):
            acc = ensemble_acc_raw[i]
            ratio = majority_ratio_raw[i]
            if is_valid_metric(acc) and is_valid_metric(ratio):
                uncertainty = 1 - ratio
                lower_bound = acc - 0.5 * uncertainty
                upper_bound = acc + 0.5 * uncertainty
                plot_rounds.append(r)
                plot_accs.append(acc)
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

        if plot_accs:
            plt.plot(plot_rounds, plot_accs, label='Ensemble Accuracy', color='blue', zorder=5)
            plt.fill_between(plot_rounds, lower_bounds, upper_bounds, color='lightblue', alpha=0.5, label='Uncertainty (1 - Majority Ratio)')

        plt.title(f'Ensemble Accuracy with Uncertainty\n(Experiment: {os.path.basename(experiment_dir)})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'ensemble_accuracy_with_uncertainty.png'))
        plt.close()
        print("Generated ensemble_accuracy_with_uncertainty.png")
    else:
        print("Skipping ensemble plot: 'ensemble' model data or 'majority_vote_ratio' not found.")

    # --- Plot Client-Side Metrics ---
    loss_files = sorted(glob.glob(os.path.join(experiment_dir, 'client_*_train_losses.json')))
    if not loss_files:
        print(f"No client loss files found in {experiment_dir}")
    else:
        plt.figure(figsize=(10, 6))
        for loss_file in loss_files:
            client_id = os.path.basename(loss_file).split('_')[1]
            with open(loss_file, 'r') as f:
                losses = json.load(f)
            plt.plot(range(1, len(losses) + 1), losses, label=f'Client {client_id}')
        
        plt.title(f'Client Training Losses in Experiment: {os.path.basename(experiment_dir)}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plot_filename = os.path.join(save_dir, 'client_losses.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate plots from federated learning experiment results.")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to the experiment output directory. If not provided, finds the latest one."
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        # Find the latest experiment folder in the current directory, ignoring hidden folders
        all_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
        if not all_dirs:
            print("No experiment folders found in the current directory.")
            return
        experiment_dir = max(all_dirs, key=lambda d: os.path.getmtime(d))
        print(f"No experiment directory provided, found latest: {experiment_dir}")

    generate_plots(experiment_dir)

if __name__ == "__main__":
    main()