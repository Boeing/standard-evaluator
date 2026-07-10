"""Testing of the TP37 class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import TP37


def test_tp37():
    """Test the TP37 test function"""
    # Instantiate the test function
    test_func = TP37()
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
    # Change x1, x2, and x3
    initial_guess.loc[1, "x1"] = 13.0
    initial_guess.loc[1, "x2"] = 12.0
    initial_guess.loc[1, "x3"] = 15.0
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x1", "x2", "x3"]
    assert test_func.outputs == ["y1", "y2", "y3"]
    assert initial_guess.iloc[0].y1 == pytest.approx(-1000.0)
    assert initial_guess.iloc[0].y2 == pytest.approx(-22.0)
    assert initial_guess.iloc[0].y3 == pytest.approx(50.0)
    assert initial_guess.iloc[1].y1 == pytest.approx(-2340.0)
    assert initial_guess.iloc[1].y2 == pytest.approx(-5.0)
    assert initial_guess.iloc[1].y3 == pytest.approx(67.0)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = TP37()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 3, "Expected 3 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2", "x3"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["y1", "y2", "y3"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["y1"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = ["y2", "y3"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds
    expected_var_bounds = ((0.0, 42.0), (0.0, 42.0), (0.0, 42.0))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (10.0, 10.0, 10.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds, scales
    expected_resp_bounds = (
        (-float("inf"), float("inf")),
        (-float("inf"), 0.0),
        (0.0, float("inf")),
    )
    expected_resp_scales = (0.0001, 0.1, 0.1)
    for resp, expected_bounds, expected_scales in zip(
        opt_problem.responses, expected_resp_bounds, expected_resp_scales
    ):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"
        assert (
            resp.scale == expected_scales
        ), f"Response bounds do not match for response {resp}: {resp.scale} != {expected_scales}"

    # Check the description
    expected_description = """Implement the Rosenbrock Post Office problem, with derivatives
    minimize y1 subject to

    0 <= x(i) <= 42, i=1,...,3

    y2 <= 0.0

    y3 >= 0.0

    x* = ( 0.24000000E+02 0.12000000E+02 0.12000000E+02 )

    at x* (y1, y2, y3) = (-3456.0, 0.0, 72.0)"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = ""
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = TP37()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].y1 == pytest.approx(-1000.0)
    assert initial_guess_df.iloc[0].y2 == pytest.approx(-22.0)
    assert initial_guess_df.iloc[0].y3 == pytest.approx(50.0)
