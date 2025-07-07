#!/usr/bin/env python3
"""
Test seeding to understand why we're not getting variation
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
    cost = np.asarray(table, dtype="object")
    cost = np.where(
        np.isin(cost, ["X", None, -1]),                 # forbidden markers
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

def test_assignments():
    """Test assignments for different seeds"""
    
    table = [
        [0, 2, 1, -1, -1, -1],  # M0: eligible for C0,C1,C2
        [1, -1, -1, 2, -1, 0],  # M1: eligible for C0,C3,C5  
        [2, -1, -1, -1, 0, 1],  # M2: eligible for C0,C4,C5
        [-1, 0, -1, 1, 2, -1],  # M3: eligible for C1,C3,C4
        [-1, 1, 2, 0, -1, -1],  # M4: eligible for C1,C2,C3
        [-1, -1, 0, -1, 1, 2]   # M5: eligible for C2,C4,C5
    ]
    
    print("Testing assignments for first few rounds:")
    
    # Test Round 1 (seeds 0-5)
    print("\nRound 1:")
    for i in range(6):
        assignment = random_mapping(table, seed=i)
        part_name = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer'][i]
        assignments_str = ', '.join([f'M{m}->C{c}' for m, c in assignment])
        print(f"  {part_name} (seed {i}): {assignments_str}")
    
    # Test Round 2 (seeds 6-11)
    print("\nRound 2:")
    for i in range(6):
        seed = 6 + i
        assignment = random_mapping(table, seed=seed)
        part_name = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer'][i]
        assignments_str = ', '.join([f'M{m}->C{c}' for m, c in assignment])
        print(f"  {part_name} (seed {seed}): {assignments_str}")
    
    # Test Round 3 (seeds 12-17)
    print("\nRound 3:")
    for i in range(6):
        seed = 12 + i
        assignment = random_mapping(table, seed=seed)
        part_name = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer'][i]
        assignments_str = ', '.join([f'M{m}->C{c}' for m, c in assignment])
        print(f"  {part_name} (seed {seed}): {assignments_str}")

if __name__ == "__main__":
    test_assignments()