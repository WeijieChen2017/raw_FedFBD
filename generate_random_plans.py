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


table = [
    [0, 2, 1, -1, -1, -1],
    [1, -1, -1, 2, -1, 0],
    [2, -1, -1, -1, 0, 1],
    [-1, 0, -1, 1, 2, -1],
    [-1, 1, 2, 0, -1, -1],
    [-1, -1, 0, -1, 1, 2]
]

for i in range(30):
    print(random_mapping(table, seed=i))

# → [(0, 0), (1, 5), (2, 4), (3, 1), (4, 3), (5, 2)]

