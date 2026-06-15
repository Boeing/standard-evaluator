"""Testing of the ExtendedRosenbrock class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import ExtendedRosenbrock


def test_extendedrosenbrock():
    """Test the ExtendedRosenbrock test function"""
    # Instantiate the test function
    test_func = ExtendedRosenbrock()
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
    # Change the variables x0, and x1
    initial_guess.loc[1, "x0"] = -1.4
    initial_guess.loc[1, "x1"] = 1.5
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x0", "x1"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(24.199999)
    assert initial_guess.iloc[1].f == pytest.approx(26.919999)


def test_extendedrosenbrock_opt():
    "Test the optimal solution of the Extended Rosenbrock test function"
    # Instantiate the test function
    test_func = ExtendedRosenbrock()
    # Get the initial guess
    known_sol = test_func.known_solution
    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0].f == pytest.approx(0.0)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = ExtendedRosenbrock()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 responses"

    # Check the names of variables
    expected_variable_names = ["x0", "x1"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["f"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = []
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"
    # Check the initial guess
    expected_initial_guess = (-1.2, 1.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the description
    expected_description = r"""$$
The Extended Rosenbrock function.

This evaluator is an example function from
"Numerical Methods for Unconstrained Optimization and
Nonlinear Equations" by J.E. Dennis Jr. and R.B. Schnabel.

The problem has two independent variables, :math:`x_0` and :math:`x_1` and
a single response:

.. math::
    f(x_0, x_1) = \left[10*\left(x_1-x_0^2\right)\right]^2 + (1-x_0)^2        
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = "J. E. Dennis, Jr., Robert B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations', Volume 16 of Classics in Applied Mathematics, 1996"
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = ExtendedRosenbrock()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(24.199999999999996)
