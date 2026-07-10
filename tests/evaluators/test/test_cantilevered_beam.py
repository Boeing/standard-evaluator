"""Testing of the CantileveredBeam classes"""

import pandas as pd
import numpy as np
import pytest
from standard_evaluator.evaluators.test import CantileveredBeam


def test_cantilevered_beam():
    """Test the Cantilevered Beam test function"""
    # Instantiate the test function
    test_func = CantileveredBeam()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )

    # Change variables hc, b1, b2 & H
    initial_guess.at[1, "x0"] = 3
    initial_guess.at[1, "b1"] = 9.9
    initial_guess.at[1, "b2"] = 1.5
    initial_guess.at[1, "H"] = 5.4
    initial_guess.at[2, "b1"] = 0.0
    initial_guess.at[2, "b2"] = 0.1e-12
    initial_guess.at[3, "x0"] = 1
    initial_guess.at[4, "x0"] = 2
    initial_guess.at[5, "x0"] = 5
    initial_guess.at[6, "x0"] = 6
    initial_guess.at[7, "x0"] = 7
    initial_guess.at[8, "x0"] = 8

    # Evaluate the initial guess
    test_func(initial_guess)

    expected = pd.DataFrame(
        columns=test_func.outputs,
        data=[
            [0.174828, 3642.250101, 672.0],
            [0.125804, 2830.599126, 838.8],
            [np.nan, np.nan, np.nan],
            [0.398198, 8295.787215, 386.4],
            [0.259611, 5408.552273, 493.5],
            [0.150624, 3137.995168, 779.1],
            [0.139379, 2903.723420, 850.5],
            [0.127053, 2646.940490, 957.6],
            [0.120949, 2519.773220, 1029.0],
        ],
    )

    pd.testing.assert_frame_equal(initial_guess[test_func.outputs], expected)
    # Check some specific responses
    assert initial_guess.x0.dtypes == pd.Int64Dtype()
    assert test_func.inputs == ["b1", "b2", "H", "x0"]
    assert test_func.outputs == ["deflection", "stress", "volume"]


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = CantileveredBeam()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 4, "Expected 4 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check the names of variables
    expected_variable_names = ["b1", "b2", "H", "x0"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["deflection", "stress", "volume"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["volume"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = ["volume", "stress", "deflection"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds and defaults
    expected_var_bounds = ((2.0, 12.0), (0.1, 2.0), (3.0, 7.0), (1, 8))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (7.0, 1.05, 5.0, 4)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check for the types of variables
    # Check the class_type of the first three variables (should be float)
    for idx in range(3):
        assert (
            opt_problem.variables[idx].class_type == "float"
        ), f"Variable x[{idx}] should be of class type 'float'"

    # Check the class_type of the last variable (should be int)
    for idx in range(3, 4):
        assert (
            opt_problem.variables[idx].class_type == "int"
        ), f"Variable x[{idx}] should be of class type 'int'"

    # Check response bounds
    expected_resp_bounds = ((0.0, 0.1), (0.0, 5000.0), (0.0, 1200.0))
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the response scales
    expected_scales = [100, 0.001, 0.01]
    actual_scales = [resp.scale for resp in opt_problem.responses]
    assert (
        actual_scales == expected_scales
    ), f"Expected scales {expected_scales}, got {actual_scales}"

    # Check the description
    expected_description = r"""Cantilevered Beam Example Evaluator

The problem has four design variables with types as follows:

- x0 - integer
- x1 - double
- x2 - double
- x3 - double

There are three responses:

- volume
- stress
- deflection

.. note:: The source for this instantiation of this evaluator is
    a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_
    called *"SHERPA - An Efficient and Robust Optimization/Search
    Algorithm"*.

::

    |<------ b1 ------>|
     __________________
    |                  |
    |                  | h1
    |_____        _____|
          |      |
          |      |
          |<-b2->|   H - 2 * h1
          |      |
     _____|      |_____
    |                  |
    |                  | h1
    |__________________|

- x0 -> h1 in [0.1, 1.0] (accessed by indicies [1, 2, 3, 4, 5, 6, 7, 8] into
  a lookup table { .1, .25, .35, .5, .65, .75, .9, 1.0 } in this domain)
- x1 -> b1 in [2.0, 12.0]
- x2 -> b2 in [0.1, 2.0]
- x3 -> H  in [3.0, 7.0]

|

- W = Load (hard coded to 1000 pounds)
- E = Modulus of Elasticity (hard coded to 1.0e7 psi)
- L = length (hard coded to 60 inches)
- Z = I / z (section modulus of the cross-section of the beam)
- z = H / 2 (Distance from neutral axis to extreme fiber (edge))
- I = Moment of Inertia (of cross section about neutral axis)

| I = (1/12) * ((H - 2 * h1) * b2 ^ 3) + (1/12) * (b1 * h1 ^ 3) + (1/12) * (b1 * h1 ^ 3) + b1 * h1 * ((H - h1) / 2 ) ^ 2 + b1 * h1 * ((H - h1) / 2) ^ 2
| = (1/12) * ((H - 2 * h1) * b2 ^ 3) + 2 * (b1 * h1 ^ 3)) + (b1 * h1 *(H - h1) ^ 2) / 2
|
| V = (2 * b1 * h1 + b2 * (H - 2 * h1)) * L
| stress(x) = W *(L - x) / Z = W * (L - x) * z / I = W * (L - x) * H / (2 * I)
| max stress = stress(0) = W * L * H / (2 * I)
| deflection = W * L ^ 3 / (3 * E * I)"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = 'The source for this instantiation of this evaluator is a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_ called *"SHERPA - An Efficient and Robust Optimization/Search Algorithm"*.'
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    test_func = CantileveredBeam()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    assert initial_guess_df.iloc[0].deflection == pytest.approx(0.17482800485633346)
    assert initial_guess_df.iloc[0].stress == pytest.approx(3642.2501011736135)
    assert initial_guess_df.iloc[0].volume == pytest.approx(672.0)
