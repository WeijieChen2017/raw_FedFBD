import json
import os
import torch
import torch.nn as nn
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms

from fbd_model_ckpt import get_pretrained_fbd_model
from fbd_utils import load_fbd_settings, FBDWarehouse
from fbd_dataset import DATASET_SPECIFIC_RULES

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
        if model_color == "averaging":
            model = warehouse.get_averaged_model()
        elif model_color == "ensemble":
            # For ensemble, use M0 as placeholder
            model = warehouse.get_model_by_color("M0")
        else:
            model = warehouse.get_model_by_color(model_color)
        
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
    # Load shipping and update plans
    shipping_plan_path = f"config/{args.experiment_name}/shipping_plan.json"
    with open(shipping_plan_path, 'r') as f:
        shipping_plans = json.load(f)
    
    update_plan_path = f"config/{args.experiment_name}/update_plan.json"
    with open(update_plan_path, 'r') as f:
        update_plans = json.load(f)
    
    return shipping_plans, update_plans

def prepare_test_dataset(args):
    """Prepare test dataset for server evaluations"""
    print("Server: Preparing test dataset for evaluations...")
    info = INFO[args.experiment_name]
    DataClass = getattr(medmnist, info['python_class'])
    dataset_rules = DATASET_SPECIFIC_RULES.get(args.experiment_name, {})
    as_rgb = dataset_rules.get("as_rgb", False)
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
        round_losses.append(loss)
        
        print(f"Server: Received update from client {client_id} for round {round_num}, loss: {loss:.4f}")
        
        if updated_weights:
            print(f"Server: Received {len(updated_weights)} weight blocks from client {client_id}: {list(updated_weights.keys())}")
            warehouse.store_weights_batch(updated_weights)
            print(f"Server: Stored {len(updated_weights)} weight blocks from client {client_id}")
        else:
            print(f"Server: WARNING - Client {client_id} sent no updated weights!")
        
        if updated_optimizer_states:
            print(f"Server: Received {len(updated_optimizer_states)} optimizer states from client {client_id}.")
            warehouse.store_optimizer_state_batch(updated_optimizer_states)
    
    # Print summary
    avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
    print(f"Server: All responses for round {round_num} collected. Average loss: {avg_loss:.4f}")
    
    # Evaluate all models once at the end of the round
    print(f"Server: Evaluating all models at end of round {round_num}...")
    round_eval_results = {'round': round_num}
    
    # Evaluate individual models M0 to M5
    for model_idx in range(6):
        model_color = f"M{model_idx}"
        metrics = evaluate_server_model(args, model_color, args.model_flag, args.experiment_name, args.test_dataset, warehouse)
        round_eval_results[model_color] = metrics
    
    # Evaluate averaged model
    avg_metrics = evaluate_server_model(args, "averaging", args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    round_eval_results["averaging"] = avg_metrics
    
    # Evaluate ensemble model
    ensemble_metrics = evaluate_server_model(args, "ensemble", args.model_flag, args.experiment_name, args.test_dataset, warehouse)
    round_eval_results["ensemble"] = ensemble_metrics
    
    return round_eval_results

def get_client_plans_for_round(round_num, client_id, shipping_plans, update_plans):
    """Get shipping and update plans for a specific client and round"""
    round_shipping_plan = shipping_plans.get(str(round_num + 1), {})
    round_update_plan = update_plans.get(str(round_num + 1), {})
    
    client_shipping_list = round_shipping_plan.get(str(client_id), [])
    client_update_plan = round_update_plan.get(str(client_id), {})
    
    return client_shipping_list, client_update_plan