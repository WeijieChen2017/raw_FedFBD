#!/usr/bin/env python3
"""
Generate systematic assignments for all model parts using round % 3 and random_mapping
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

    # 2) Add a tiny random tie-breaker to every *allowed* entry
    noise = rng.random(cost.shape) * 1e-6
    cost = np.where(np.isinf(cost), cost, cost + noise)

    # 3) Hungarian algorithm on the noised matrix
    rows, cols = linear_sum_assignment(cost)

    # 4) Verify feasibility (all picked entries are finite)
    if np.isinf(cost[rows, cols]).any():
        raise ValueError("No legal one-to-one mapping exists.")

    return list(zip(rows, cols))

def create_part_mapping_table():
    """Create mapping table for a model part using FBD settings"""
    
    # Load FBD settings
    with open('config/organamnist_shuffle/fbd_settings.json', 'r') as f:
        settings = json.load(f)
    
    schedule = settings['FBD_INFO']['training_plan']['schedule']
    clients = settings['FBD_INFO']['clients']
    
    # Create the base mapping table (same for all parts)
    # This table shows which round%3 each model gets assigned to each client
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

def generate_assignments_for_round(round_num, seed_base=42):
    """Generate assignments for a specific round using round % 3"""
    
    # Get the base mapping table
    base_table = create_part_mapping_table()
    
    # Determine which round mod 3 we're in
    round_mod = round_num % 3
    
    # Create cost table for this round - prefer assignments that match round_mod
    cost_table = []
    for row in base_table:
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
    
    # Get assignments for all 6 parts using different seeds
    part_names = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
    assignments = {}
    
    for i, part_name in enumerate(part_names):
        try:
            # Use different seed for each part to get variety
            seed = seed_base + round_num * 10 + i
            assignment = random_mapping(cost_table, seed=seed)
            assignments[part_name] = assignment
        except ValueError as e:
            print(f"Warning: Could not generate assignment for {part_name} in round {round_num}: {e}")
            assignments[part_name] = None
    
    return assignments

def print_assignments_for_round(round_num):
    """Print assignments for a specific round"""
    print(f"\n=== ROUND {round_num} (round%3={round_num%3}) ASSIGNMENTS ===")
    
    assignments = generate_assignments_for_round(round_num)
    
    for part_name, assignment in assignments.items():
        if assignment:
            print(f"\n{part_name.upper()}:")
            for model_idx, client_idx in assignment:
                print(f"  M{model_idx} -> C{client_idx}")
        else:
            print(f"\n{part_name.upper()}: No valid assignment found")

def test_systematic_assignments():
    """Test the systematic assignment generation"""
    
    print("=== SYSTEMATIC ASSIGNMENT GENERATION TEST ===")
    
    # Show the base mapping table
    base_table = create_part_mapping_table()
    print("\nBase mapping table (round%3 assignments):")
    print("     C0  C1  C2  C3  C4  C5")
    for i, row in enumerate(base_table):
        row_str = f'M{i} | '
        for val in row:
            if val == 'X':
                row_str += ' X |'
            else:
                row_str += f' {val} |'
        print(row_str)
    
    # Test assignments for first few rounds
    for round_num in [1, 2, 3, 4, 5]:
        print_assignments_for_round(round_num)

if __name__ == "__main__":
    test_systematic_assignments()