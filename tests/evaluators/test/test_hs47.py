"""Testing of the HS47 classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import HS47


def test_hs47():
    """Test the HS47 test function"""
    # Instantiate the test function
    test_func = HS47()
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
    initial_guess.loc[1, "x0"] = -20.0
    initial_guess.loc[1, "x1"] = 11.4
    initial_guess.loc[1, "x2"] = 43.2
    initial_guess.loc[1, "x3"] = 71.8
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x1", "x2", "x3", "x4", "x5"]
    assert test_func.outputs == ["f", "c1", "c2", "c3"]
    assert initial_guess.iloc[0].f == pytest.approx(12.495436)
    assert initial_guess.iloc[0].c1 == pytest.approx(0.0)
    assert initial_guess.iloc[0].c2 == pytest.approx(0.0)
    assert initial_guess.iloc[0].c3 == pytest.approx(0.0)
    assert initial_guess.iloc[1].f == pytest.approx(25721578.46691062)
    assert initial_guess.iloc[1].c1 == pytest.approx(372020.872)
    assert initial_guess.iloc[1].c2 == pytest.approx(-5112.454213562373)
    assert initial_guess.iloc[1].c3 == pytest.approx(4.7)


def test_hs47_create_opt_problem():
    # Creates an instance of the class that contains the _create_opt_problem method
    opt_problem = HS47()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 5, "Expected 5 variables"
    assert len(opt_problem.responses) == 4, "Expected 4 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2", "x3", "x4", "x5"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f", "c1", "c2", "c3"]
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
    expected_constraints = ["c1", "c2", "c3"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check the initial guess
    expected_initial_guess = (2.0, (2.0) ** 0.5, -1.0, 2.0 - (2.0) ** 0.5, 0.5)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the scales
    expected_scales = [0.1, 1.0, 1.0, 1.0]
    actual_scales = [resp.scale for resp in opt_problem.responses]
    assert (
        actual_scales == expected_scales
    ), f"Expected scales {expected_scales}, got {actual_scales}"

    # Check the description
    expected_description = r"""$$
\begin{align}
    \min\quad & f(x) = (x_1 - x_2)^2 + (x_2 - x_3)^3 +(x_3 - x_4)^4 + (x_4 - x_5)^4\\[1em]
    \text{s.t}\quad & c_1 = x_1 + x_2^2 + x_3^3 - 3 = 0\\
    & c_2 = x_2 - x_3^2 + x_4 - 1 = 0\\
    & c_3 = x_1x_5 - 1 = 0
\end{align}
$$

The initial point is given by:

$$
x_0 = (2.0,~ \sqrt{2.0},~ -1.0,~ 2.0 - \sqrt{2.0},~ 0.5)
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = HS47()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(12.495436)
    assert initial_guess_df.iloc[0].c1 == pytest.approx(0.0)
    assert initial_guess_df.iloc[0].c2 == pytest.approx(0.0)
    assert initial_guess_df.iloc[0].c3 == pytest.approx(0.0)


def test_hs47_opt():
    "Test the optimal solution of the HS38 test function"
    # Instantiate the test function
    test_func = HS47()
    # Get the initial guess
    known_sol = test_func.known_solution
    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0].f == pytest.approx(0.0)
    assert known_sol.iloc[0].c1 == pytest.approx(0.0)
    assert known_sol.iloc[0].c2 == pytest.approx(0.0)
    assert known_sol.iloc[0].c3 == pytest.approx(0.0)
