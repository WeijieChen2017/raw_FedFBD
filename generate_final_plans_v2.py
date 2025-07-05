#!/usr/bin/env python3
"""
Generate final training plans using the working random_mapping approach from generate_random_plans.py
Translates the assignments to proper update plan format and creates both model and client views.
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

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
    cost = np.asarray(table, dtype="object")
    cost = np.where(
        np.isin(cost, ["X", None]),                 # forbidden markers (NOT -1)
        np.inf,
        cost.astype(float)
    )

    # 2) Add a tiny random tie-breaker to every *allowed* entry
    noise = rng.random(cost.shape) * 1e-6
    cost = np.where(np.isinf(cost), cost, cost + noise)

    # 3) Hungarian algorithm on the noised matrix
    rows, cols = linear_sum_assignment(cost)

    # 4) Verify feasibility (all picked entries are finite)
    if np.isinf(cost[rows, cols]).any():
        raise ValueError("No legal one-to-one mapping exists.")

    return list(zip(rows, cols))

def create_mapping_table():
    """Create the specific mapping table using -1 for forbidden cells (like generate_random_plans.py)"""
    
    # The table from generate_random_plans.py
    table = [
        [0, 2, 1, -1, -1, -1],  # M0: eligible for C0,C1,C2
        [1, -1, -1, 2, -1, 0],  # M1: eligible for C0,C3,C5  
        [2, -1, -1, -1, 0, 1],  # M2: eligible for C0,C4,C5
        [-1, 0, -1, 1, 2, -1],  # M3: eligible for C1,C3,C4
        [-1, 1, 2, 0, -1, -1],  # M4: eligible for C1,C2,C3
        [-1, -1, 0, -1, 1, 2]   # M5: eligible for C2,C4,C5
    ]
    
    return table

def load_fbd_settings():
    """Load FBD settings to understand block structure"""
    with open('config/organamnist_shuffle/fbd_settings.json', 'r') as f:
        return json.load(f)

def get_block_structure(fbd_settings):
    """Parse block structure from FBD settings"""
    fbd_trace = fbd_settings['FBD_TRACE']
    
    # Group blocks by model color and part
    model_blocks = defaultdict(lambda: defaultdict(list))
    for block_id, info in fbd_trace.items():
        color = info['color']
        model_part = info['model_part']
        model_blocks[color][model_part].append(block_id)
    
    return model_blocks, fbd_trace

def generate_all_assignments(num_rounds=30, seed_base=42):
    """
    Generate assignments for all rounds and all parts using the working random_mapping.
    
    Returns:
        dict: {round_num: {part_name: [(model_idx, client_idx), ...]}}
    """
    
    # Get the mapping table (same structure as generate_random_plans.py)
    mapping_table = create_mapping_table()
    
    # All part names
    part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
    
    all_assignments = {}
    
    for round_num in range(1, num_rounds + 1):
        round_assignments = {}
        
        for i, part_name in enumerate(part_names):
            try:
                # Use different seed for each round and part combination
                # Create more variation like generate_random_plans.py does
                seed = (round_num - 1) * 6 + i  # This will give us seeds 0,1,2,3,4,5 for round 1, then 6,7,8,9,10,11 for round 2, etc.
                assignment = random_mapping(mapping_table, seed=seed)
                round_assignments[part_name] = assignment
            except ValueError as e:
                print(f"Warning: Could not generate assignment for {part_name} in round {round_num}: {e}")
                round_assignments[part_name] = []
        
        all_assignments[round_num] = round_assignments
    
    return all_assignments

def generate_update_plan(all_assignments, num_rounds=30):
    """
    Generate the update plan in the proper JSON format.
    
    Args:
        all_assignments: Dict of assignments from generate_all_assignments
        num_rounds: Number of rounds
        
    Returns:
        Dict: Update plan in proper JSON format
    """
    
    # Load FBD settings and get block structure
    fbd_settings = load_fbd_settings()
    model_blocks, fbd_trace = get_block_structure(fbd_settings)
    
    # Get eligible clients for each model
    clients = fbd_settings['FBD_INFO']['clients']
    model_clients = defaultdict(list)
    for client_id, model_list in clients.items():
        for model in model_list:
            model_clients[model].append(int(client_id))
    
    update_plan = {}
    
    for round_num in range(1, num_rounds + 1):
        round_plan = {}
        round_assignments = all_assignments[round_num]
        
        # Initialize all clients
        for client_id in range(6):
            round_plan[str(client_id)] = {
                "model_to_update": {},
                "model_as_regularizer": []
            }
        
        # Process assignments for each part
        part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
        
        # Track which models each client is training
        client_training_models = defaultdict(set)
        
        for part_name in part_names:
            if part_name in round_assignments:
                for model_idx, client_idx in round_assignments[part_name]:
                    model_name = f'M{model_idx}'
                    client_training_models[client_idx].add(model_name)
                    
                    # Find the block ID for this model and part
                    if model_name in model_blocks and part_name in model_blocks[model_name]:
                        block_ids = model_blocks[model_name][part_name]
                        if block_ids:
                            block_id = block_ids[0]  # Take first block if multiple
                            
                            # Create unique key if multiple parts with same name
                            base_key = part_name
                            key_counter = 0
                            unique_key = base_key
                            while unique_key in round_plan[str(client_idx)]["model_to_update"]:
                                key_counter += 1
                                unique_key = f"{base_key}_{key_counter}"
                            
                            round_plan[str(client_idx)]["model_to_update"][unique_key] = {
                                "block_id": block_id,
                                "status": "trainable"
                            }
        
        # Add regularizer models (eligible models that client is not training)
        for client_id in range(6):
            eligible_models = set(clients[str(client_id)])
            training_models = client_training_models[client_id]
            regularizer_models = eligible_models - training_models
            
            for model_name in regularizer_models:
                if model_name in model_blocks:
                    regularizer_spec = {}
                    for part_name in part_names:
                        if part_name in model_blocks[model_name]:
                            block_ids = model_blocks[model_name][part_name]
                            if block_ids:
                                regularizer_spec[part_name] = block_ids[0]
                    
                    if regularizer_spec:
                        round_plan[str(client_id)]["model_as_regularizer"].append(regularizer_spec)
        
        update_plan[str(round_num)] = round_plan
    
    return update_plan

def generate_model_view_files(all_assignments, num_rounds=30):
    """
    Generate model view files showing which client trains each part of each model.
    Format: training_plan_human_M0.txt to training_plan_human_M5.txt
    """
    
    part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
    
    for model_idx in range(6):
        model_name = f'M{model_idx}'
        filename = f'config/organamnist_shuffle/training_plan_human_{model_name}.txt'
        
        with open(filename, 'w') as f:
            # Write header
            f.write(' | '.join(part_names) + '\n')
            
            # Write assignments for each round
            for round_num in range(1, num_rounds + 1):
                round_assignments = all_assignments[round_num]
                
                row_parts = []
                for part_name in part_names:
                    # Find which client trains this model's part
                    assigned_client = None
                    for m_idx, c_idx in round_assignments[part_name]:
                        if m_idx == model_idx:
                            assigned_client = f'C{c_idx}'
                            break
                    
                    if assigned_client is None:
                        assigned_client = 'None'
                    
                    row_parts.append(assigned_client)
                
                f.write(' | '.join(row_parts) + '\n')
        
        print(f"Generated: {filename}")

def generate_client_view_files(all_assignments, num_rounds=30):
    """
    Generate client view files showing which model each part comes from.
    Format: client_view_C0.txt to client_view_C5.txt
    """
    
    part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
    
    for client_idx in range(6):
        client_name = f'C{client_idx}'
        filename = f'config/organamnist_shuffle/client_view_{client_name}.txt'
        
        with open(filename, 'w') as f:
            # Write header
            f.write(' | '.join(part_names) + '\n')
            
            # Write assignments for each round
            for round_num in range(1, num_rounds + 1):
                round_assignments = all_assignments[round_num]
                
                row_parts = []
                for part_name in part_names:
                    # Find which models this client trains for this part
                    assigned_models = []
                    for m_idx, c_idx in round_assignments[part_name]:
                        if c_idx == client_idx:
                            assigned_models.append(f'M{m_idx}')
                    
                    if assigned_models:
                        row_parts.append('+'.join(assigned_models))
                    else:
                        row_parts.append('None')
                
                f.write(' | '.join(row_parts) + '\n')
        
        print(f"Generated: {filename}")

def main():
    """Main function to generate all training plans and update plan"""
    
    print("=== GENERATING FINAL TRAINING PLANS V2 ===")
    print("Using working random_mapping approach from generate_random_plans.py")
    
    # Generate all assignments
    print("\n1. Generating assignments for all rounds and parts...")
    all_assignments = generate_all_assignments(num_rounds=30)
    
    # Generate update plan
    print("\n2. Generating update plan...")
    update_plan = generate_update_plan(all_assignments, num_rounds=30)
    
    # Save update plan
    update_plan_path = "config/organamnist_shuffle/update_plan.json"
    with open(update_plan_path, 'w') as f:
        json.dump(update_plan, f, indent=2)
    print(f"Generated: {update_plan_path}")
    
    # Generate model view files
    print("\n3. Generating model view files...")
    generate_model_view_files(all_assignments, num_rounds=30)
    
    # Generate client view files  
    print("\n4. Generating client view files...")
    generate_client_view_files(all_assignments, num_rounds=30)
    
    print("\n✅ All training plans and update plan generated successfully!")
    
    # Show sample from first few rounds
    print("\n📋 Sample assignments for first 3 rounds:")
    for round_num in [1, 2, 3]:
        print(f"\nRound {round_num}:")
        round_data = all_assignments[round_num]
        for part_name in ['in_layer', 'layer1']:  # Just show first two parts
            assignments = round_data[part_name]
            assignments_str = ', '.join([f'M{m}->C{c}' for m, c in assignments])
            print(f"  {part_name}: {assignments_str}")

if __name__ == "__main__":
    main()