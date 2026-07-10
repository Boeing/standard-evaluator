"""Testing of the CantileveredBeamFixedVariable class"""

import pytest
import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test import CantileveredBeamFixedVariable


def test_cantilevered_beam_fixed_variable():
    """Test the CantileveredBeamFixedVariableTF test function"""
    # Instantiate the test function
    test_func = CantileveredBeamFixedVariable()
    # Get the initial guess
    initial_guess = test_func.initial_guess()

    # Duplicate to make sure vectorization works
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )
    # Change variables hc, b1, b2 & H
    initial_guess.at[1, "b1"] = 9.9
    initial_guess.at[1, "b2"] = 1.5
    initial_guess.at[1, "H"] = 5.4
    initial_guess.at[1, "x0"] = 3
    initial_guess.at[1, "C0"] = 0.2
    initial_guess.at[1, "C1"] = 1
    initial_guess.at[2, "b1"] = 0.0
    initial_guess.at[2, "b2"] = 0.1e-12

    # Evaluate the initial guess
    test_func(initial_guess)

    # Check some specific responses
    assert test_func.inputs == ["b1", "b2", "H", "x0", "C0", "C1"]
    assert test_func.outputs == ["deflection", "stress", "volume"]

    expected = pd.DataFrame(
        columns=test_func.outputs,
        data=[
            [0.174828, 3642.250101, 672.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
    )
    pd.testing.assert_frame_equal(initial_guess[test_func.outputs], expected)

def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = CantileveredBeamFixedVariable()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 6, "Expected 6 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check the names of variables
    expected_variable_names = ["b1", "b2", "H", "x0", "C0", "C1"]
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

    expected_var_bounds = (
        (2.0, 12.0),
        (0.1, 2.0),
        (3.0, 7.0),
        (1, 8),
        (0.0, 0.0),
        (1, 1),
    )
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the class_type of the first 3 variables (should be float)
    for idx in range(3):
        assert (
            opt_problem.variables[idx].class_type == "float"
        ), f"Variable x[{idx}] should be of class type 'float'"

    # Check the class_type of the last 3 variables (should be int,float,int)
    # Assigning int class type to variables 3 and 5
    for idx in [3, 5]:
        assert (
            opt_problem.variables[idx].class_type == "int"
        ), f"Variable x[{idx}] should be of class type 'int'"

    # Check the initial guess
    expected_initial_guess = (7.0, 1.05, 5.0, 4, 0.0, 1)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the scales and response bounds
    scales = [100.0, 0.001, 0.01]
    resp_bounds = ([0.0, 0.1], [0.0, 5000.0], [0.0, 1200.0])
    # Define the bounds, scales on the responses
    for resp, local_scale, local_bounds in zip(
        opt_problem.responses, scales, resp_bounds
    ):
        resp.scale = local_scale
        resp.bounds = local_bounds

    # Check the description
    expected_description = """\
Cantilevered Beam with Fixed Variables Example Evaluator

This example initializes an optimization problem with six design variables,
three responses, and specific objectives and constraints related to the
structural analysis of a cantilevered beam. The design variables include
both float and integer types, with defined bounds and default values.

The problem has 6 design variables with types as follows:

- b1: Width of the first section of the beam (double)
- b2: Width of the second section of the beam (double)
- H: Height of the beam (double)
- x0: An integer variable
- C0: A fixed float variable
- C1: A fixed integer variable

There are three responses:

- deflection
- stress
- volume

The objectives and constraints for the optimization problem are set to
minimize the volume while ensuring that the stress and deflection
remain within specified limits.

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

    test_func = CantileveredBeamFixedVariable()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].deflection == pytest.approx(0.17482800485633346)
    assert initial_guess_df.iloc[0].stress == pytest.approx(3642.2501011736135)
    assert initial_guess_df.iloc[0].volume == pytest.approx(672.0)
