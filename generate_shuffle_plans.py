#!/usr/bin/env python3
"""
Generate shuffled shipping and update plans for federated learning.
Creates random assignments of model parts to clients instead of systematic rotation.
"""

import json
import random
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def load_fbd_settings(config_path):
    """Load FBD settings to understand model structure and client assignments"""
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_model_structure(fbd_settings):
    """Parse model structure from FBD_TRACE and FBD_INFO"""
    fbd_trace = fbd_settings['FBD_TRACE']
    fbd_info = fbd_settings['FBD_INFO']
    
    # Group blocks by model color
    models = defaultdict(list)
    for block_id, info in fbd_trace.items():
        color = info['color']
        models[color].append(block_id)
    
    # Get client assignments for each model
    client_models = fbd_info['clients']
    
    # Create reverse mapping: model -> eligible clients
    model_clients = defaultdict(list)
    for client_id, model_list in client_models.items():
        for model in model_list:
            model_clients[model].append(int(client_id))
    
    return models, model_clients, fbd_trace

def generate_random_shipping_plan(models, model_clients, fbd_trace, num_rounds, seed=42, update_plan=None):
    """
    Generate shipping plan that aligns with update plan assignments.
    Clients receive blocks they need for training and regularization.
    
    Args:
        models: Dict mapping model colors to block IDs
        model_clients: Dict mapping model colors to eligible client IDs  
        fbd_trace: FBD_TRACE mapping for block metadata
        num_rounds: Number of rounds to generate
        seed: Random seed for reproducibility
        update_plan: Update plan to align shipping with (if provided)
    
    Returns:
        Dict: Shipping plan in the required JSON format
    """
    random.seed(seed)
    
    shipping_plan = {}
    
    for round_num in range(1, num_rounds + 1):
        round_plan = {str(i): [] for i in range(6)}  # Initialize empty lists for all clients
        
        if update_plan and str(round_num) in update_plan:
            # Align with update plan
            round_update_plan = update_plan[str(round_num)]
            
            for client_id in range(6):
                client_blocks = set()
                client_update = round_update_plan[str(client_id)]
                
                # Add blocks from model_to_update (trainable)
                for model_part_info in client_update["model_to_update"].values():
                    client_blocks.add(model_part_info["block_id"])
                
                # Add blocks from model_as_regularizer 
                for regularizer_spec in client_update["model_as_regularizer"]:
                    for block_id in regularizer_spec.values():
                        client_blocks.add(block_id)
                
                round_plan[str(client_id)] = sorted(list(client_blocks))
        else:
            # Fallback: For each model, randomly assign its blocks to one eligible client
            assigned_blocks = set()
            for model_color, model_blocks in models.items():
                eligible_clients = model_clients[model_color]
                selected_client = random.choice(eligible_clients)
                
                for block_id in model_blocks:
                    if block_id not in assigned_blocks:
                        round_plan[str(selected_client)].append(block_id)
                        assigned_blocks.add(block_id)
            
            # Sort blocks for each client for consistency
            for client_id in round_plan:
                round_plan[client_id] = sorted(round_plan[client_id])
        
        shipping_plan[str(round_num)] = round_plan
    
    return shipping_plan

def random_mapping(table, seed=None):
    """
    Parameters
    ----------
    table : 2-D array-like
        Numeric entries are allowed.  Either put np.inf or the string "X"
        for forbidden cells.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[int, int]]
        One admissible row-column assignment.  Each row and column appears once.

    Raises
    ------
    ValueError
        If no perfect assignment exists.
    """
    rng = np.random.default_rng(seed)

    # 1) Convert to a float matrix, inf = forbidden
    table_array = np.array(table, dtype='object')
    
    # Create cost matrix by replacing X with inf and converting numbers to float
    cost = np.full(table_array.shape, np.inf, dtype=float)
    
    for i in range(table_array.shape[0]):
        for j in range(table_array.shape[1]):
            if table_array[i,j] != 'X' and table_array[i,j] is not None:
                cost[i,j] = float(table_array[i,j])

    # 2) Add a tiny random tie-breaker to every *allowed* entry
    noise = rng.random(cost.shape) * 1e-6
    cost = np.where(np.isinf(cost), cost, cost + noise)

    # 3) Hungarian algorithm on the noised matrix
    rows, cols = linear_sum_assignment(cost)

    # 4) Verify feasibility (all picked entries are finite)
    if np.isinf(cost[rows, cols]).any():
        raise ValueError("No legal one-to-one mapping exists.")

    return list(zip(rows, cols))

def create_part_mapping_table(fbd_settings):
    """Create mapping table for model parts using FBD settings"""
    
    schedule = fbd_settings['FBD_INFO']['training_plan']['schedule']
    clients = fbd_settings['FBD_INFO']['clients']
    
    # Create the base mapping table (same for all parts)
    mapping_table = []
    
    for model_idx in range(6):  # M0-M5
        model_name = f'M{model_idx}'
        row = []
        
        for client_idx in range(6):  # C0-C5
            # Check if this client is eligible for this model
            if model_name in clients[str(client_idx)]:
                # Find which round%3 this model gets assigned to this client
                assigned_round = None
                for round_mod in range(3):
                    if schedule[str(round_mod)][str(client_idx)] == model_name:
                        assigned_round = round_mod
                        break
                
                if assigned_round is not None:
                    row.append(assigned_round)
                else:
                    row.append('X')  # Eligible but not in rotation
            else:
                row.append('X')  # Not eligible
        
        mapping_table.append(row)
    
    return mapping_table

def generate_systematic_update_plan(models, model_clients, fbd_trace, fbd_settings, num_rounds, seed=42):
    """
    Generate systematic update plan using round % 3 and random_mapping:
    1. Each unique block is trainable by exactly ONE client per round (model_to_update)
    2. Model parts are systematically distributed using FBD settings rotation
    3. Regularizer models can be shared among all eligible clients
    
    Args:
        models: Dict mapping model colors to block IDs
        model_clients: Dict mapping model colors to eligible client IDs
        fbd_trace: FBD_TRACE mapping for block metadata
        fbd_settings: FBD settings with training schedule
        num_rounds: Number of rounds to generate
        seed: Random seed for reproducibility
    
    Returns:
        Dict: Update plan in the required JSON format
    """
    
    # Get the base mapping table for systematic assignment
    base_mapping_table = create_part_mapping_table(fbd_settings)
    
    update_plan = {}
    
    for round_num in range(1, num_rounds + 1):
        round_plan = {}
        
        # Generate assignments for all 6 parts using round % 3 approach
        round_mod = round_num % 3
        
        # Create cost table for this round - prefer assignments that match round_mod
        cost_table = []
        for row in base_mapping_table:
            cost_row = []
            for val in row:
                if val == 'X':
                    cost_row.append('X')
                else:
                    # Lower cost (better) if this matches our round_mod
                    if val == round_mod:
                        cost_row.append(0)  # Preferred assignment
                    else:
                        cost_row.append(1)  # Less preferred but still valid
            cost_table.append(cost_row)
        
        # Get assignments for all 6 parts
        part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
        part_assignments = {}
        
        for i, part_name in enumerate(part_names):
            try:
                # Use different seed for each part to get variety
                part_seed = seed + round_num * 10 + i
                assignment = random_mapping(cost_table, seed=part_seed)
                part_assignments[part_name] = {model_idx: client_idx for model_idx, client_idx in assignment}
            except ValueError:
                print(f"Warning: Could not generate assignment for {part_name} in round {round_num}")
                part_assignments[part_name] = {}
        
        # Build client plans from part assignments
        for client_id in range(6):
            client_plan = {
                "model_to_update": {},
                "model_as_regularizer": []
            }
            
            # Collect all blocks this client should train
            client_blocks = []
            for part_name in part_names:
                if part_name in part_assignments:
                    for model_idx, assigned_client in part_assignments[part_name].items():
                        if assigned_client == client_id:
                            # Find the block ID for this model and part
                            model_color = f'M{model_idx}'
                            if model_color in models:
                                for block_id in models[model_color]:
                                    if fbd_trace[block_id]['model_part'] == part_name:
                                        client_blocks.append((block_id, part_name))
                                        break
            
            # Add trainable blocks for this client with unique keys
            part_counter = {}
            for block_id, part_name in client_blocks:
                # Create unique key if multiple blocks have same model_part
                if part_name in part_counter:
                    part_counter[part_name] += 1
                    unique_key = f"{part_name}_{part_counter[part_name]}"
                else:
                    part_counter[part_name] = 0
                    unique_key = part_name
                
                client_plan["model_to_update"][unique_key] = {
                    "block_id": block_id,
                    "status": "trainable"
                }
            
            # Add regularizer models (all eligible models this client isn't training)
            training_models = set()
            for block_id, _ in client_blocks:
                for model_color, model_blocks in models.items():
                    if block_id in model_blocks:
                        training_models.add(model_color)
                        break
            
            for model_color, eligible_clients in model_clients.items():
                if client_id in eligible_clients and model_color not in training_models:
                    regularizer_spec = {}
                    for block_id in models[model_color]:
                        block_info = fbd_trace[block_id]
                        model_part = block_info['model_part']
                        regularizer_spec[model_part] = block_id
                    client_plan["model_as_regularizer"].append(regularizer_spec)
            
            round_plan[str(client_id)] = client_plan
        
        update_plan[str(round_num)] = round_plan
    
    return update_plan

def generate_shuffle_plans(config_dir, num_rounds=30, seed=42):
    """
    Generate shuffled shipping and update plans for the given config directory.
    
    Args:
        config_dir: Path to config directory (e.g., "config/organamnist_shuffle")
        num_rounds: Number of rounds to generate plans for
        seed: Random seed for reproducibility
    """
    print(f"Generating shuffle plans for {config_dir}")
    print(f"Rounds: {num_rounds}, Seed: {seed}")
    
    # Load FBD settings
    fbd_settings_path = f"{config_dir}/fbd_settings.json"
    fbd_settings = load_fbd_settings(fbd_settings_path)
    
    # Parse model structure
    models, model_clients, fbd_trace = parse_model_structure(fbd_settings)
    
    print(f"\nModel structure:")
    for model_color, blocks in models.items():
        clients = model_clients[model_color]
        print(f"  {model_color}: {len(blocks)} blocks -> clients {clients}")
    
    # Generate random update plan first (defines trainable assignments)
    print(f"\nGenerating random update plan...")
    update_plan = generate_random_update_plan(models, model_clients, fbd_trace, num_rounds, seed)
    
    # Generate shipping plan aligned with update plan
    print(f"Generating shipping plan aligned with update plan...")
    shipping_plan = generate_random_shipping_plan(models, model_clients, fbd_trace, num_rounds, seed, update_plan)
    
    # Save plans
    shipping_plan_path = f"{config_dir}/shipping_plan.json"
    update_plan_path = f"{config_dir}/update_plan.json"
    
    with open(shipping_plan_path, 'w') as f:
        json.dump(shipping_plan, f, indent=2)
    print(f"✅ Saved shipping plan: {shipping_plan_path}")
    
    with open(update_plan_path, 'w') as f:
        json.dump(update_plan, f, indent=2)
    print(f"✅ Saved update plan: {update_plan_path}")
    
    # Print sample of generated plan
    print(f"\n📋 Sample shipping plan (Round 1):")
    for client_id, blocks in shipping_plan['1'].items():
        print(f"  Client {client_id}: {len(blocks)} blocks")
    
    print(f"\n📋 Sample update plan (Round 1):")
    for client_id, plan in update_plan['1'].items():
        trainable_blocks = len(plan['model_to_update'])
        regularizers = len(plan['model_as_regularizer'])
        print(f"  Client {client_id}: {trainable_blocks} trainable parts, {regularizers} regularizers")
    
    print(f"\n🎉 Shuffle plans generated successfully!")
    return shipping_plan, update_plan

def main():
    """Main function to generate shuffle plans"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate shuffled federated learning plans")
    parser.add_argument("--config_dir", type=str, default="config/organamnist_shuffle",
                        help="Config directory path")
    parser.add_argument("--num_rounds", type=int, default=30,
                        help="Number of rounds to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generate_shuffle_plans(args.config_dir, args.num_rounds, args.seed)

if __name__ == "__main__":
    main()