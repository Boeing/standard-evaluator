"""Testing of the HelicalValley class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import HelicalValley


def test_helicalvalley():
    """Test the HelicalValley test function"""
    # Instantiate the test function
    test_func = HelicalValley()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    # Duplicate to make sure vectorization works
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )
    # Change x1, x2, and x3
    initial_guess.loc[1, "x1"] = -1.4
    initial_guess.loc[1, "x2"] = -1.2
    initial_guess.loc[1, "x3"] = -4.2

    initial_guess.loc[2, "x1"] = 2.9
    initial_guess.loc[2, "x2"] = -1.2
    initial_guess.loc[2, "x3"] = -1.3

    initial_guess.loc[3, "x1"] = 11.5
    initial_guess.loc[3, "x2"] = 0.6
    initial_guess.loc[3, "x3"] = 1.6
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x1", "x2", "x3"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(10000.0)
    assert initial_guess.iloc[1].f == pytest.approx(2927.42)
    assert initial_guess.iloc[2].f == pytest.approx(504.635)
    assert initial_guess.iloc[3].f == pytest.approx(11290.5720799601)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = HelicalValley()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 3, "Expected 3 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 responses"

    # Check the names of variables
    expected_variable_names = ["x1", "x2", "x3"]
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
    expected_initial_guess = (-1.0, 0.0, 0.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the scales
    expected_scales = [0.0001]
    actual_scales = [resp.scale for resp in opt_problem.responses]
    assert (
        actual_scales == expected_scales
    ), f"Expected scales {expected_scales}, got {actual_scales}"

    # Check the description
    expected_description = r"""$$
\begin{aligned}
    \min\quad & f = f_1^2 + f_2^2 + x_3^2\\[1em]
    \text{where}\quad & \theta = \begin{cases}
        \frac{\arctan(x_2/x_1)}{2\pi} & x_1 \geq 0\\
        \frac{\arctan(x_2/x_1)}{2\pi} + 0.5 & x_1 < 0
    \end{cases}\\
    & f_1 = 10(x_3 - 10\theta)\\
    & f_2 = 10(\sqrt{x_1^2 + x_2^2} - 1)
\end{aligned}
$$"""

    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = "J. J. More, B. S. Garbow and K. E. Hillstrom, “Testing Unconstrained Optimization Software,” ACM Transac tions on Mathematical Software, Vol. 7, No. 1, 1981, pp. 19-31. doi:10.1145/355934.355936"
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = HelicalValley()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(10000.0)
