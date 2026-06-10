"""Testing of the SimpleMultiFiLo class"""

import pytest
import pandas as pd
import math
import numpy as np
from standard_evaluator.evaluators.test import SimpleMultiFiLo


def test_simplemultifi_lo():
    """Test the SimpleMultiFiLo test function"""
    # Instantiate the test function
    test_func = SimpleMultiFiLo()
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
    # Change x and y
    initial_guess.loc[1, "x"] = 2.889
    initial_guess.loc[1, "y"] = 1.2456
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x", "y"]
    assert test_func.responses == ["c1", "c2", "f"]
    assert initial_guess.iloc[0].f == pytest.approx(-2.4674011002723395)
    assert initial_guess.iloc[0].c1 == pytest.approx(-0.9618025729327639)
    assert initial_guess.iloc[0].c2 == pytest.approx(0.5707963267948966)
    # Check the known solution
    assert initial_guess.iloc[1].f == pytest.approx(-3.5985384)
    assert initial_guess.iloc[1].c1 == pytest.approx(-0.1558615999999997)
    assert initial_guess.iloc[1].c2 == pytest.approx(1.8889999999999998)


def test_simplemultifi_lo_opt():
    "Test the optimal solution of the SimpleMultiFiLo test function"
    # Instantiate the test function
    test_func = SimpleMultiFiLo()
    # Get the initial guess
    known_sol = test_func.known_solution
    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0].f == pytest.approx(-3.3333334)
    assert known_sol.iloc[0].c1 == pytest.approx(9.99999993922529e-08)
    assert known_sol.iloc[0].c2 == pytest.approx(1.0)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = SimpleMultiFiLo()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check the names of variables
    expected_variable_names = ["x", "y"]
    actual_variable_names = [var.name for var in opt_problem.variables]

    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["c1", "c2", "f"]
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
    expected_constraints = ["c1", "c2"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds and defaults
    expected_var_bounds = ((0.0, math.pi), (0.0, math.pi))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (math.pi / 2.0, math.pi / 2.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds
    expected_resp_bounds = ((-10.0, 0.0), (-3.0, 1.0), (-np.inf, np.inf))
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the description
    expected_description = r"""$$
Source code for a simple multifidelity problem capturing the common information.
author: Mark Abramson (Joe Simonis invented the test problem)
date Jun 12, 2015
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

    test_func = SimpleMultiFiLo()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(-2.4674011002723395)
    assert initial_guess_df.iloc[0].c1 == pytest.approx(-0.9618025729327639)
    assert initial_guess_df.iloc[0].c2 == pytest.approx(0.5707963267948966)
