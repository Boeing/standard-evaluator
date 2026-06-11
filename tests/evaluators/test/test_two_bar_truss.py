"""Testing of the TwoBarTruss class"""

import pytest
import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test import TwoBarTruss


def test_twobartruss():
    """Test the TwoBarTruss test function"""
    # Instantiate the test function
    test_func = TwoBarTruss()
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
    initial_guess.loc[1, "Q"] = 900.0
    initial_guess.loc[1, "S"] = 1500.0
    initial_guess.loc[1, "X1"] = 6.579
    initial_guess.loc[1, "X2"] = 0.457
    initial_guess.loc[1, "rho"] = 15000.0
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["Q", "S", "X1", "X2", "rho"]
    assert test_func.outputs == ["f", "g1", "g2"]
    assert initial_guess.iloc[0].f == pytest.approx(57486.67600017284)
    assert initial_guess.iloc[0].g1 == pytest.approx(-3.299482314589319e-05)
    assert initial_guess.iloc[0].g2 == pytest.approx(0.4979914049891776)
    assert initial_guess.iloc[1].f == pytest.approx(108501.85751825645)
    assert initial_guess.iloc[1].g1 == pytest.approx(0.3664403804316029)
    assert initial_guess.iloc[1].g2 == pytest.approx(0.6385879833389898)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = TwoBarTruss()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 5, "Expected 5 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check the names of variables
    expected_variable_names = ["Q", "S", "X1", "X2", "rho"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f", "g1", "g2"]
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
    expected_constraints = ["g1", "g2"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds
    expected_var_bounds = (
        (0.0, np.inf),
        (0.0, np.inf),
        (0.2, 20.0),
        (0.1, 1.6),
        (0.0, np.inf),
    )
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (800.0, 1050.0, 5.3791, 0.377, 10000.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds, scales
    expected_resp_bounds = (
        (-float("inf"), float("inf")),
        (0.0, float("inf")),
        (0.0, float("inf")),
    )

    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the description
    expected_description = r"""$$
The Two Bar Truss evaluator. This evaluator is a concrete class which inherits
    from the abstract DE::Evaluator class.  This is to be used as an example
    for deriving Design Explorer Evaluators and for testing future optimization
    methods.

    The example is a two-bar truss. See
    A robust design method using variable transformation and Gauss-Hermite
    integration by Beiqing Huang and Xiaoping Du,
    INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING
    Int. J. Numer. Meth. Engng 2006; 66:1841-1858

    The problem has five design variables with types as follows:
       - X1 - double,  cross-sectional area of the truss
       - X2 - double,  half distance between two bottom rollers.
       - rho - double, the density of the bar material
       - Q - double, magnitude of the external force Q applied on the top of the truss
       - S - double, the bar material's tensile strength

    There are three responses:
       - f, double, weight
       - g1, double, strength 1
       - g2, double, strength 2

    The objective of the design is to minimize the weight of the two-bar truss subject to
    the two strength constraints about the axial stress in each bar.
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = 'Beiqing Huang and Xiaoping Du, "A robust design method using variable transformation and Gauss-Hermite integration," *INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING*, Int. J. Numer. Meth. Engng 2006; 66:1841-1858.'
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = TwoBarTruss()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(57486.67600017284)
    assert initial_guess_df.iloc[0].g1 == pytest.approx(-3.299482314589319e-05)
    assert initial_guess_df.iloc[0].g2 == pytest.approx(0.4979914049891776)
