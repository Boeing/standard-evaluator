"""Testing of the WrkBkPrb1 class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import WrkBkPrb1


def test_wrkbkprb1():
    """Test the WrkBkPrb1 test function"""
    # Instantiate the test function
    test_func = WrkBkPrb1()
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
    # Change x1, and x2
    initial_guess.loc[1, "x1"] = 1.5
    initial_guess.loc[1, "x2"] = 4.0
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x1", "x2"]
    assert test_func.responses == ["f","c"]
    assert initial_guess.iloc[0].c == pytest.approx(1.0)
    assert initial_guess.iloc[0].f == pytest.approx(4.25)
    assert initial_guess.iloc[1].c == pytest.approx(6.0)
    assert initial_guess.iloc[1].f == pytest.approx(20.04176)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = WrkBkPrb1()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 2, "Expected 2 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f", "c"]
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
    expected_constraints = ["c"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds and scales
    expected_var_bounds = ((1e-08, 10.0), (1e-08, 10.0))
    expected_var_scales = [0.1, 0.1]

    for var, expected_bounds, expected_scales in zip(
        opt_problem.variables, expected_var_bounds, expected_var_scales
    ):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"
        assert (
            var.scale == expected_scales
        ), f"Variable defaults do not match for variable {var}: {var.scale} != {expected_scales}"

    # Check the initial guess
    expected_initial_guess = (0.5, 2.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds, scales
    expected_resp_bounds = ((-float("inf"), float("inf")), (1.0, 100.0))
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the description
    expected_description = r"""$$
work book problem 1 from the SOCS user's guide page 23

    minimize f(x1, x2)

    subject to

    1.0e-8 <= x1, x2 <= 10.0

    1.0 <= c <= 100.0

    where

    :math:`c(x_1, x_2) = x_1x_2`

    :math:`f(x_1, x_2) = x_1^2 + x_2^2 + \ln(c)`

    x0 = (0.5, 2.0)
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = 'Beiqing Huang and Xiaoping Du, "A robust design method using variable transformation and Gauss-Hermite integration," *International Journal for Numerical Methods in Engineering*, Int. J. Numer. Meth. Engng 2006; 66:1841-1858.'
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = WrkBkPrb1()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(4.25)
    assert initial_guess_df.iloc[0].c == pytest.approx(1.0)
