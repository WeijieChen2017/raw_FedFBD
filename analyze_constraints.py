#!/usr/bin/env python3
"""
Analyze the constraint structure to understand if multiple valid assignments exist
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations

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

def find_all_valid_assignments():
    """Find all valid assignments by brute force"""
    
    mapping_table = create_mapping_table()
    
    print("Base mapping table:")
    print("     C0  C1  C2  C3  C4  C5")
    for i, row in enumerate(mapping_table):
        row_str = f'M{i} | '
        for val in row:
            if val == 'X':
                row_str += ' X |'
            else:
                row_str += f' {val} |'
        print(row_str)
    print()
    
    # Create boolean matrix of valid assignments
    valid_matrix = []
    for i, row in enumerate(mapping_table):
        valid_row = []
        for j, val in enumerate(row):
            valid_row.append(val != 'X')
        valid_matrix.append(valid_row)
    
    # Find all valid permutations
    valid_assignments = []
    
    # Try all possible permutations of columns (clients)
    for perm in permutations(range(6)):
        # Check if this permutation is valid
        valid = True
        for model_idx in range(6):
            client_idx = perm[model_idx]
            if not valid_matrix[model_idx][client_idx]:
                valid = False
                break
        
        if valid:
            assignment = [(i, perm[i]) for i in range(6)]
            valid_assignments.append(assignment)
    
    print(f"Found {len(valid_assignments)} valid assignments:")
    for i, assignment in enumerate(valid_assignments):
        print(f"\nAssignment {i+1}:")
        for model_idx, client_idx in assignment:
            cost = mapping_table[model_idx][client_idx]
            print(f"  M{model_idx} -> C{client_idx} (cost={cost})")
        
        # Calculate total cost
        total_cost = sum(mapping_table[model_idx][client_idx] for model_idx, client_idx in assignment)
        print(f"  Total cost: {total_cost}")
    
    return valid_assignments

def test_random_selection():
    """Test if we can create variation by randomly selecting among valid assignments"""
    
    valid_assignments = find_all_valid_assignments()
    
    if len(valid_assignments) > 1:
        print(f"\nWe have {len(valid_assignments)} valid assignments to choose from!")
        print("We can create variation by randomly selecting among these.")
    else:
        print(f"\nOnly {len(valid_assignments)} valid assignment exists.")
        print("The constraints are too tight to create variation.")

if __name__ == "__main__":
    test_random_selection()