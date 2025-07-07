#!/usr/bin/env python3
"""
Generate final training plans using exact random_mapping approach.
For each round and each model part, use random_mapping with the specific table 
to assign model parts to clients with one-to-one mapping.
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment

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

    # 2) Add significant random noise to every *allowed* entry to create variation
    # Use larger noise (up to 0.5) to actually affect assignment decisions
    noise = rng.random(cost.shape) * 0.5
    cost = np.where(np.isinf(cost), cost, cost + noise)

    # 3) Hungarian algorithm on the noised matrix
    rows, cols = linear_sum_assignment(cost)

    # 4) Verify feasibility (all picked entries are finite)
    if np.isinf(cost[rows, cols]).any():
        raise ValueError("No legal one-to-one mapping exists.")

    return list(zip(rows, cols))

def create_mapping_table():
    """Create the specific mapping table from FBD settings"""
    
    # Load FBD settings
    with open('config/organamnist_shuffle/fbd_settings.json', 'r') as f:
        settings = json.load(f)
    
    schedule = settings['FBD_INFO']['training_plan']['schedule']
    clients = settings['FBD_INFO']['clients']
    
    # Create the mapping table (same for all parts)
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

def generate_all_assignments(num_rounds=30, seed_base=42):
    """
    Generate assignments for all rounds and all parts using random_mapping.
    
    Returns:
        dict: {round_num: {part_name: [(model_idx, client_idx), ...]}}
    """
    
    # Get the mapping table
    mapping_table = create_mapping_table()
    
    # All part names
    part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
    
    all_assignments = {}
    
    for round_num in range(1, num_rounds + 1):
        round_assignments = {}
        
        for i, part_name in enumerate(part_names):
            try:
                # Use different seed for each round and part combination
                # Make the seed more varied by using larger multipliers
                seed = seed_base + round_num * 1000 + i * 100 + round_num * i
                assignment = random_mapping(mapping_table, seed=seed)
                round_assignments[part_name] = assignment
            except ValueError as e:
                print(f"Warning: Could not generate assignment for {part_name} in round {round_num}: {e}")
                round_assignments[part_name] = []
        
        all_assignments[round_num] = round_assignments
    
    return all_assignments

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
    """Main function to generate all training plans"""
    
    print("=== GENERATING FINAL TRAINING PLANS ===")
    print("Using exact random_mapping approach for each round and part")
    
    # Generate all assignments
    print("\n1. Generating assignments for all rounds and parts...")
    all_assignments = generate_all_assignments(num_rounds=30)
    
    # Generate model view files
    print("\n2. Generating model view files...")
    generate_model_view_files(all_assignments, num_rounds=30)
    
    # Generate client view files  
    print("\n3. Generating client view files...")
    generate_client_view_files(all_assignments, num_rounds=30)
    
    print("\n✅ All training plans generated successfully!")
    
    # Show sample from first round
    print("\n📋 Sample assignments for Round 1:")
    round1 = all_assignments[1]
    for part_name, assignments in round1.items():
        print(f"  {part_name}: {assignments}")

if __name__ == "__main__":
    main()