"""Testing of the DisconnectTF class"""

import pytest
import pandas as pd
import math
from standard_evaluator.evaluators.test import Disconnect


def test_disconnect():
    """Test the Disconnect test function"""
    # Instantiate the test function
    test_func = Disconnect()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )
    # Change x, and y
    initial_guess.loc[1, "x"] = 0.4
    initial_guess.loc[1, "y"] = 0.6

    initial_guess.loc[2, "x"] = 0.3
    initial_guess.loc[2, "y"] = 0.2

    # Evaluate the initial guess
    test_func(initial_guess)

    # Check some specific responses
    assert test_func.variables == ["x", "y"]
    assert test_func.responses == ["h", "k"]
    assert initial_guess.iloc[0].h == pytest.approx(-6.9)
    assert initial_guess.iloc[0].k == pytest.approx(4.5)
    assert initial_guess.iloc[1].h == pytest.approx(0.380014)
    assert initial_guess.iloc[1].k == pytest.approx(0.02)
    assert initial_guess.iloc[2].h == pytest.approx(0.770014)
    assert initial_guess.iloc[2].k == pytest.approx(0.13)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = Disconnect()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 2, "Expected 2 responses"

    # Check the names of variables
    expected_variable_names = ["x", "y"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["h", "k"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["x", "y"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = ["h", "k"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds and defaults
    expected_var_bounds = ((1.0e-6, 2 * math.asin(1.0)), (1.0e-6, 2 * math.asin(1.0)))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (2.0, 2.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds
    expected_resp_bounds = ((-float("inf"), 0.0), (-float("inf"), 0.5))
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the description
    expected_description = r"""$$
The SphereEvaluator class is a multiobjective test problem.  It models a hyperellipsoid in
n-dimensional space, centered at an arbitrary point and extending to the coordinate
planes on each axis, although an optional offset may be used as well.  The default bounds
on the object are the interior and surface of the hyperellipsoid, hence a multiobjective
problem would expect to find the minimum-facing surface of the object.

The default problem seeks to minimize each coordinate.  This can be easily checked by
asserting that a point x is <= center and that the constraint value is equal to 1 (within
toleance).

For example, a center of [2, 5, 1] with an offset of [1, 2, 3] would have minima at [1, 7, 4],
[3, 2, 4], and [3, 7, 1].
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = "David A. Van Veldhuizen, Multiobjective Evolutionary Algorithms: Classifications, Analyses and New Innovations, Dissertation, AFIT/DS/ENG/99-01. Appendix B (Problem Tanaka)"
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = Disconnect()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].x == pytest.approx(2.0)
    assert initial_guess_df.iloc[0].y == pytest.approx(2.0)
    assert initial_guess_df.iloc[0].h == pytest.approx(-6.9)
    assert initial_guess_df.iloc[0].k == pytest.approx(4.5)
