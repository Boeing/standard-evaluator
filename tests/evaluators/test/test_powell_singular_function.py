"""Testing of the Trigonometric class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import PowellSingularFunction


def test_powell_singular_function():
    """Test the PowellSingularFunction test function"""
    # Instantiate the test function
    test_func = PowellSingularFunction()
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
    initial_guess.loc[1, "x1"] = 0.4
    initial_guess.loc[1, "x2"] = 1.4
    initial_guess.loc[1, "x3"] = -2.4
    initial_guess.loc[1, "x4"] = -1.8
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x1", "x2", "x3", "x4"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(95.0)
    assert initial_guess.iloc[1].f == pytest.approx(296.00000000000006)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = PowellSingularFunction()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 4, "Expected 4 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 responses"

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

    # Check the constraints
    expected_constraints = []
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check the initial guess
    expected_initial_guess = (3.0, -1.0, 0.0, 1.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the description
    expected_description = r"""The Extended Powell Singular function.

    This function comes from
    "Numerical Methods for Unconstrained Optimization and
    Nonlinear Equations" by J.E. Dennis Jr. and R.B. Shnabel.

    .. math::
        \begin{align*}
            \min \quad& f = f_1^2 + f_2^2 + f_3^2 + f_4^2\\[.75em]
            \text{where} \quad& f_1 = x_1 + 10x_2\\
            & f_2 = \sqrt{5}\cdot(x_3 - x_4)\\
            & f_3 = x_2 - 2x_3\\
            & f_4 = \sqrt{10}\cdot(x_1 - x_4)
        \end{align*}

    The initial guess provided for this problem is

    .. math::
        f(3, -1, 0, 1) = 95"""

    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = "J.E. Dennis Jr. and R.B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations'"
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = PowellSingularFunction()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(95.0)
