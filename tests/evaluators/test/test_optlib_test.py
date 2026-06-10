"""Testing of the OptlibTest class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import OptlibTest


def test_optlibtest_initial():
    """Test the OptlibTest test function"""
    # Instantiate the test function
    test_func = OptlibTest()
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
    initial_guess.at[1, "x1"] = 2.2
    initial_guess.at[1, "x2"] = 4.5
    initial_guess.at[1, "x3"] = 3.9

    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x1", "x2", "x3"]
    assert test_func.responses == ["c", "f"]
    assert initial_guess.iloc[0].f == pytest.approx(1.0)
    assert initial_guess.iloc[0].c == pytest.approx(17.757359312880716)
    assert initial_guess.iloc[1].f == pytest.approx(6.8595999)
    assert initial_guess.iloc[1].c == pytest.approx(269.851459)


def test_optlibtest():
    """Test the OptlibTest test function"""
    # Instantiate the test function
    test_func = OptlibTest()
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
    initial_guess.at[1, "x1"] = 2.2
    initial_guess.at[1, "x2"] = 4.5
    initial_guess.at[1, "x3"] = 3.9
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x1", "x2", "x3"]
    assert test_func.responses == ["c", "f"]
    assert initial_guess.iloc[0].f == pytest.approx(1.0)
    assert initial_guess.iloc[0].c == pytest.approx(17.757359312880716)
    assert initial_guess.iloc[1].f == pytest.approx(6.8595999999999995)
    assert initial_guess.iloc[1].c == pytest.approx(269.8514593128807)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = OptlibTest()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 3, "Expected 3 variables"
    assert len(opt_problem.responses) == 2, "Expected 2 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2", "x3"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["c", "f"]
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

    # Check the initial guess
    expected_initial_guess = (2.0, 2.0, 2.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the description
    expected_description = r"""$$
The example can be found in the Optlib 6.2 manual on page 308.
The example can be found in the SOCS 7.1 manual on page 625.

.. math::
    \begin{align}
        \min\quad & f = (x_1 - 1)^2 + (x_1 - x_2)^2 + (x_2 - x_3)^4\\[1em]
        \text{s.t.}\quad & c = x_1(1 + x_2^2) + x_3^4 - 4 - 3\sqrt{2} = 0
    \end{align}

The optimal solution in the SOCS manual is

x* = [1.104859034205678, 1.196674180655277, 1.535262258200661]

with f = 0.032568200256415, c = 1.253397385880817e-010    
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
    test_func = OptlibTest()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(1.0)
    assert initial_guess_df.iloc[0].c == pytest.approx(17.757359)
