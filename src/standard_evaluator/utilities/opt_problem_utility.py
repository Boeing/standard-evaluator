# Utilities to help convert an opt_problem into a fully defined state
from typing import Tuple, Dict

from standard_evaluator.problem import OptProblem


def get_opt_problem_constant_vars(problem: OptProblem) -> Dict[int, Tuple[str, float]]:
    """Determines the constant variables of a given OptProblem.

    A variable is considered constant if its lower and upper bounds are
    identical (within tolerance 1e-12), or for categorical variables, if
    all categories are the same.

    Args:
        problem: An instance of OptProblem.

    Returns:
        A dict with constant variable indices as keys and (name, value)
        tuples as values.
    """
    constant_var_dict = {}
    variables = problem.variables  # This is a list of Variable objects

    for var_idx, var in enumerate(variables):
        # For categorical variables
        if getattr(var, "class_type", None) == "cat":
            # If all bounds are the same, variable is constant
            if len(set(var.bounds)) == 1:
                constant_var_dict[var_idx] = (var.name, var.bounds[0])
        else:
            # For float and int variables
            lb, ub = var.bounds
            if abs(ub - lb) <= 1.0e-12:
                constant_var_dict[var_idx] = (var.name, lb)

    return constant_var_dict
