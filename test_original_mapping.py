#!/usr/bin/env python3
"""
Test the original mapping from generate_random_plans.py with different seeds
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def random_mapping(table, seed=None):
    """
    Exact copy from generate_random_plans.py
    """
    rng = np.random.default_rng(seed)

    # 1) Convert to a float matrix, inf = forbidden
    cost = np.asarray(table, dtype="object")
    cost = np.where(
        np.isin(cost, ["X", None]),                 # forbidden markers
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

    # return list(zip(rows, cols))
    return rows, cols

def test_original_table():
    """Test with exact table from generate_random_plans.py"""
    
    table = [
        [0, 2, 1, -1, -1, -1],
        [1, -1, -1, 2, -1, 0],
        [2, -1, -1, -1, 0, 1],
        [-1, 0, -1, 1, 2, -1],
        [-1, 1, 2, 0, -1, -1],
        [-1, -1, 0, -1, 1, 2]
    ]
    
    print("Testing original table from generate_random_plans.py:")
    print("Note: generate_random_plans.py uses -1 for forbidden, NOT 'X'")
    
    for i in range(10):
        rows, cols = random_mapping(table, seed=i)
        assignment = list(zip(rows, cols))
        assignments_str = ', '.join([f'M{m}->C{c}' for m, c in assignment])
        print(f"Seed {i}: {assignments_str}")

if __name__ == "__main__":
    test_original_table()