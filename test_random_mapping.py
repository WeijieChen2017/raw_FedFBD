#!/usr/bin/env python3
"""
Test the random_mapping function with in_layer table
"""

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

def test_in_layer_mapping():
    """Test the random_mapping function with in_layer table"""
    
    # Create the in_layer mapping table from FBD settings
    in_layer_table = [
        [0, 2, 1, 'X', 'X', 'X'],  # M0: eligible for C0,C1,C2
        [1, 'X', 'X', 2, 'X', 0],  # M1: eligible for C0,C3,C5  
        [2, 'X', 'X', 'X', 0, 1],  # M2: eligible for C0,C4,C5
        ['X', 0, 'X', 1, 2, 'X'],  # M3: eligible for C1,C3,C4
        ['X', 1, 2, 0, 'X', 'X'],  # M4: eligible for C1,C2,C3
        ['X', 'X', 0, 'X', 1, 2]   # M5: eligible for C2,C4,C5
    ]

    print('Testing random_mapping function with in_layer table:')
    print('Input table:')
    print('     C0  C1  C2  C3  C4  C5')
    for i, row in enumerate(in_layer_table):
        row_str = f'M{i} | '
        for val in row:
            if val == 'X':
                row_str += ' X |'
            else:
                row_str += f' {val} |'
        print(row_str)

    try:
        assignment = random_mapping(in_layer_table, seed=42)
        print(f'\nAssignment result: {assignment}')
        
        print('\nInterpretation (M_x -> C_y):')
        for model_idx, client_idx in assignment:
            original_cost = in_layer_table[model_idx][client_idx]
            print(f'M{model_idx} in_layer -> C{client_idx} (round%3={original_cost})')
            
        # Verify all models and clients are assigned exactly once
        models_assigned = [pair[0] for pair in assignment]
        clients_assigned = [pair[1] for pair in assignment]
        
        print(f'\nVerification:')
        print(f'Models assigned: {sorted(models_assigned)} (should be [0,1,2,3,4,5])')
        print(f'Clients assigned: {sorted(clients_assigned)} (should be [0,1,2,3,4,5])')
        print(f'Perfect assignment: {len(set(models_assigned)) == 6 and len(set(clients_assigned)) == 6}')
        
        return assignment
        
    except Exception as e:
        print(f'Error: {e}')
        return None

if __name__ == "__main__":
    test_in_layer_mapping()