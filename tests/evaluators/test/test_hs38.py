"""Testing of the HS38 classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import HS38


def test_hs38():
    """Test the HS38 test function"""
    # Instantiate the test function
    test_func = HS38()
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
    initial_guess.loc[1, "x0"] = 8.0
    initial_guess.loc[1, "x1"] = -5.4
    initial_guess.loc[1, "x2"] = 3.2
    initial_guess.loc[1, "x3"] = -0.8
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x1", "x2", "x3", "x4"]
    assert test_func.responses == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(19192.0)
    assert initial_guess.iloc[1].f == pytest.approx(67680.58800)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = HS38()._create_opt_problem()
    # Check the number of variables and responses
    assert len(opt_problem.variables) == 4, "Expected 4 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 response"

    # Check the names of variables
    expected_variable_names = ["x1", "x2", "x3", "x4"]
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

    # Check the bounds and scales of the variables
    for var in opt_problem.variables:
        assert var.bounds == (
            -10.0,
            10.0,
        ), f"expected_bounds {(-10.0, 10.0)}, got {var.bounds}"

    expected_scales = [0.1, 0.1, 0.1, 0.1]
    actual_scales = [var.scale for var in opt_problem.variables]
    assert (
        actual_scales == expected_scales
    ), f"expected_scales {expected_scales}, got {actual_scales}"

    # Check the scales of the responses
    for resp in opt_problem.responses:
        resp.scale == 0.0001, f"expected_scale {0.0001}, got {resp.scale}"

    # Check the initial guess
    expected_initial_guess = (-3.0, -1.0, -3.0, -1.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the description of the problem
    expected_description = r"""$$
\min f(x) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2 + 90(x_4 - x_3^2)^2
+ (1-x_3)^2 + 10.1[(x_2-1)^2 + (x_4-1)^2] + 19.8(x_2-1)(x_4-1)
$$

x0 = (-3.0, -1.0, -3.0, -1.0)

f(x0) = 19192.0

x* = (1.0, 1.0, 1.0, 1.0)

f(x*) = 0.0
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected format"

    # Check the citation
    expected_citation = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
    assert opt_problem.cite == expected_citation, f"Citation does not match expected"

    # Test to check response values of evaluated initial guess
    test_func = HS38()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(19192.0)


def test_hs38_opt():
    "Test the optimal solution of the HS38 test function"
    # Instantiate the test function
    test_func = HS38()
    # Get the initial guess
    known_sol = test_func.known_solution

    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0].f == pytest.approx(0.0)
