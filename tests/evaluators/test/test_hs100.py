"""Testing of the hs100 classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import HS100


def test_hs100():
    """Test the hs100 test function"""
    # Instantiate the test function
    test_func = HS100()
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
    initial_guess.loc[1, "x1"] = 11.4
    initial_guess.loc[1, "x2"] = 43.2
    initial_guess.loc[1, "x3"] = 71.8
    initial_guess.loc[1, "x4"] = -20.0
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    assert test_func.responses == ["f", "c1", "c2", "c3", "c4"]
    assert initial_guess.iloc[0].f == pytest.approx(714.0)
    assert initial_guess.iloc[0].c1 == pytest.approx(13.0)
    assert initial_guess.iloc[0].c2 == pytest.approx(265.0)
    assert initial_guess.iloc[0].c3 == pytest.approx(171.0)
    assert initial_guess.iloc[0].c4 == pytest.approx(4.0)
    assert initial_guess.iloc[1].f == pytest.approx(26584237.617599998)
    assert initial_guess.iloc[1].c1 == pytest.approx(-10450359.932800004)
    assert initial_guess.iloc[1].c2 == pytest.approx(-51459.8)
    assert initial_guess.iloc[1].c3 == pytest.approx(-1930.44000000000034)
    assert initial_guess.iloc[1].c4 == pytest.approx(-11213.119999999)


def test_hs100_create_opt_problem():
    # Creates an instance of the class that contains the _create_opt_problem method
    opt_problem = HS100()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 7, "Expected 7 variables"
    assert len(opt_problem.responses) == 5, "Expected 5 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f", "c1", "c2", "c3", "c4"]
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
    expected_constraints = ["c1", "c2", "c3", "c4"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check the initial guess
    expected_initial_guess = (1, 2, 0, 4, 0, 1, 1)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check variable bounds and scales
    expected_var_bounds = (
        (-10.0, 10.075),
        (-10.0, 10.075),
        (-10.0, 10.075),
        (-10.0, 10.075),
        (-10.0, 10.075),
        (-10.0, 10.075),
        (-10.0, 10.075),
    )
    expected_var_scales = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    for var, expected_bounds, expected_scales in zip(
        opt_problem.variables, expected_var_bounds, expected_var_scales
    ):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"
        assert (
            var.scale == expected_scales
        ), f"Variable defaults do not match for variable {var}: {var.scale} != {expected_scales}"

    # Check the description
    expected_description = r"""$$
\begin{align}
    \min\quad & (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4 + 3(x_4 - 11)^2\\
                & + 10x_5^6 + 7x_6^2 + x_7^4 - 4x_6x_7 - 10x_6 - 8x_7\\[1em]
    \text{s.t.}\quad & 2x_1^2 + 3x_2^4 + x_3 + 4x_4^2 + 5x_5 \leq 127\\
    & 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5 \leq 282\\
    & 23x_1 + x_2^2 + 6x_6 - 8x_7 \leq 196\\
    & 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7 \geq 0
\end{align}
$$

The following bounds are placed on the variables:

$$
-10 \leq x_i \leq 10.075 \qquad i = 1,...,7
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
    test_func = HS100()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(714.0)
    assert initial_guess_df.iloc[0].c1 == pytest.approx(13.0)
    assert initial_guess_df.iloc[0].c2 == pytest.approx(265.0)
    assert initial_guess_df.iloc[0].c3 == pytest.approx(171.0)
    assert initial_guess_df.iloc[0].c4 == pytest.approx(4.0)


def test_hs100_opt():
    "Test the optimal solution of the HS38 test function"
    # Instantiate the test function
    test_func = HS100()
    # Get the initial guess
    known_sol = test_func.known_solution
    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0].f == pytest.approx(680.630111240)
    assert known_sol.iloc[0].c1 == pytest.approx(4.504147689932125e-05)
    assert known_sol.iloc[0].c2 == pytest.approx(252.56172011286043)
    assert known_sol.iloc[0].c3 == pytest.approx(144.87819047865)
    assert known_sol.iloc[0].c4 == pytest.approx(6.868068080478906e-06)
