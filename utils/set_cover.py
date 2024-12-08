import numpy as np
import cvxpy as cp


def solve_set_cover(dist_mat: np.ndarray, radius: float) -> int:
    """
    Solves the Set CoverType problem using Integer Linear Programming with cvxpy.

    Parameters:
    universe (list): A list of elements in the universe.
    sets (list of sets): A list of subsets, where each subset is a set of elements.
    """
    cover_mat = (dist_mat <= radius).astype(float)

    # 
    # Number of sets
    num_sets = cover_mat.shape[1]

    # Binary variables indicating if a set is selected
    x = cp.Variable(num_sets, boolean=True)
    # 
    # # Create a matrix that represents the covering relation between elements and sets
    # cover_mat = np.zeros((len(universe), num_sets))
    # 
    # for i, elem in enumerate(universe):
    #     for j, subset in enumerate(sets):
    #         if elem in subset:
    #             cover_mat[i, j] = 1

    # Objective: Minimize the number of selected sets
    objective = cp.Minimize(cp.sum(x))

    # Constraints: Every element in the universe must be covered by at least one set
    constraints = [cover_mat @ x >= 1]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()
    return problem.value
