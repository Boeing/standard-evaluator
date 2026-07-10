"""Testing of the ExponentialMultiFiHi class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import ExponentialMultiFiHi


def test_exponential_multi_fi_hi():
    """Test the ExponentialMultiFiHi test function"""
    # Instantiate the test function
    test_func = ExponentialMultiFiHi()
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
    # Change variable x1 & x2
    initial_guess.loc[1, "x1"] = 0.2889
    initial_guess.loc[1, "x2"] = 0.829287
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x1", "x2"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(9.410020)
    assert initial_guess.iloc[1].f == pytest.approx(6.091125888128862)


def test_exponential_multi_fi_hi_equals_zero_case():
    """Test the ExponentialMultiFiHi test function"""
    # Instantiate the test function
    test_func = ExponentialMultiFiHi()
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
    # Change variable x1 & x2
    initial_guess.loc[1, "x1"] = 0.2889
    initial_guess.loc[1, "x2"] = 0.0
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x1", "x2"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(9.4100200872)
    assert initial_guess.iloc[1].f == pytest.approx(13.4523499)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = ExponentialMultiFiHi()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2"]
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

    # Check variable bounds and defaults
    expected_var_bounds = ((0.0, 1.0), (0.0, 1.0))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (0.6759, 0.2456)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the description
    expected_description = """This function is a two-dimensional example which occurs several times in the
literature on computer experiments."""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = ""
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = ExponentialMultiFiHi()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(9.410020024031121)
