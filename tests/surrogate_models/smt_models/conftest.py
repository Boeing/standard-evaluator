"""Shared fixtures for SMT surrogate model tests."""

import numpy as np
import pandas as pd
import pytest

import standard_evaluator as se
from standard_evaluator.problem import OptProblem


def make_opt_prob(n_vars: int, var_bounds: list, resp_names: list) -> OptProblem:
    """Helper to create an OptProblem with specified bounds and response names.

    Args:
        n_vars: Number of independent variables.
        var_bounds: List of (lower, upper) bound tuples for each variable.
        resp_names: List of response names.

    Returns:
        OptProblem with variables named x0..xN-1 and specified response names.
    """
    new_prob = se.utilities.create_opt_problem(
        num_independent=n_vars, num_dependent=len(resp_names)
    )
    for var, bounds in zip(new_prob.variables, var_bounds):
        var.bounds = bounds
    for resp, name in zip(new_prob.responses, resp_names):
        resp.name = name
    return new_prob


@pytest.fixture
def opt_prob() -> OptProblem:
    return make_opt_prob(2, [(-2.0, 2.0), (-2.0, 2.0)], ["f"])


@pytest.fixture
def sites() -> pd.DataFrame:
    sites = np.array(
        [
            [-2.0, -2.0],
            [-2.0, -0.66666667],
            [-2.0, 0.66666667],
            [-2.0, 2.0],
            [-0.66666667, -2.0],
            [-0.66666667, -0.66666667],
            [-0.66666667, 0.66666667],
            [-0.66666667, 2.0],
            [0.66666667, -2.0],
            [0.66666667, -0.66666667],
            [0.66666667, 0.66666667],
            [0.66666667, 2.0],
            [2.0, -2.0],
            [2.0, -0.66666667],
            [2.0, 0.66666667],
            [2.0, 2.0],
        ]
    )

    site_vals = np.array(
        [
            [45.0],
            [30.77777778],
            [20.11111111],
            [13.0],
            [8.75308642],
            [4.01234568],
            [2.82716049],
            [5.19753086],
            [6.08641975],
            [1.34567901],
            [0.16049383],
            [2.5308642],
            [37.0],
            [22.77777778],
            [12.11111111],
            [5.0],
        ]
    )

    return pd.DataFrame(data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f"])


@pytest.fixture
def multiresp_opt_prob() -> OptProblem:
    return make_opt_prob(
        4, [(-4.0, 4.0), (-2.0, 2.0), (-3.0, 3.0), (-1.0, 1.0)], ["f", "g"]
    )


@pytest.fixture
def multiresp_sites() -> pd.DataFrame:
    sites = np.array(
        [
            [-1.43390995, -1.88266419, -0.41548999, -0.88564682],
            [3.44145177, -1.80679821, -2.53763282, -0.86957506],
            [-1.1801132, -1.23600234, -1.00750128, 0.95598218],
            [2.58823043, -0.59079174, -0.34969871, 0.84639297],
            [0.65145111, -1.78654187, -1.61057601, -0.42885959],
        ]
    )

    site_vals = np.array(
        [
            [25.41931167, 1.90218082],
            [83.54405058, 48.60879966],
            [14.76794761, 5.94901524],
            [35.90873658, 0.98928299],
            [4.98129029, 19.46252132],
        ]
    )

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "x2", "x3", "f", "g"]
    )
