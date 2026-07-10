"""Testing of the ConstrainedBetts class"""

import pytest
import pandas as pd
import numpy as np

from standard_evaluator.evaluators.test import ConstrainedBetts


def test_constrained_betts():
    """Test the ConstrainedBetts test function"""
    # Instantiate the test function
    test_func = ConstrainedBetts()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    # Duplicate to make sure vectorization works
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )

    # Change variable values for the second site
    initial_guess.loc[1, "x1"] = 2.0
    initial_guess.loc[1, "x2"] = 0.0

    # Evaluate the initial guess
    test_func(initial_guess)

    # Check some specific responses
    assert test_func.inputs == ["x1", "x2"]
    assert test_func.outputs == ["f", "c1"]
    assert initial_guess.iloc[0].f == pytest.approx(-98.99)
    assert initial_guess.iloc[0].c1 == pytest.approx(-9.0)
    assert initial_guess.iloc[1].f == pytest.approx(-99.96)
    assert initial_guess.iloc[1].c1 == pytest.approx(20.0)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = ConstrainedBetts()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 2, "Expected 2 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check variable bounds and defaults
    expected_var_bounds = ((2.0, 50.0), (-50.0, 50.0))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (-1.0, -1.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the names of responses
    expected_response_names = ["f", "c1"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check response bounds
    expected_resp_bounds = ((-np.inf, np.inf), (10.0, np.inf))
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the objective function
    expected_objectives = ["f"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = ["c1"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check the description
    expected_description = r"""Linearly constrained Betts function:

$$
\begin{aligned}
    \min\quad & f(x) = 0.01 x_1^2 + x_2^2 - 100 \\[1em]
    \text{s.t.}\quad & 10 x_1 - x_2 \geq 10 \\
    & 2 \leq x_1 \leq 50 \\
    & -50 \leq x_2 \leq 50
\end{aligned}
$$

Known optimal solution:

$$
x^* = (2, 0), \quad f(x^*) = -99.96
$$

Initial point (infeasible):

$$
x_0 = (-1, -1)
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = ""
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = ConstrainedBetts()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(-98.99)
    assert initial_guess_df.iloc[0].c1 == pytest.approx(-9.0)
