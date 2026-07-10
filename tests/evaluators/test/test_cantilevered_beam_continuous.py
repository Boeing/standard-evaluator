"""Testing of the CantileveredBeamContinuous class"""

import pytest
import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test import CantileveredBeamContinuous


def test_cantilevered_beam_continuous():
    """Test the CantileveredBeamContinuousTF test function"""
    # Instantiate the test function
    test_func = CantileveredBeamContinuous()
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
    initial_guess.at[1, "hc"] = 0.7
    initial_guess.at[1, "b1"] = 9.9
    initial_guess.at[1, "b2"] = 1.5
    initial_guess.at[1, "H"] = 5.4
    initial_guess.at[2, "b1"] = 0.0
    initial_guess.at[2, "b2"] = 0.1e-12

    # Evaluate the initial guess
    test_func(initial_guess)

    # Check some specific responses
    assert test_func.inputs == ["hc", "b1", "b2", "H"]
    assert test_func.outputs == ["deflection", "stress", "volume"]
    assert initial_guess.iloc[0].deflection == pytest.approx(0.16550077056507492)
    assert initial_guess.iloc[0].stress == pytest.approx(3447.932720105728)
    assert initial_guess.iloc[0].volume == pytest.approx(707.7)
    assert initial_guess.iloc[1].deflection == pytest.approx(0.08459859143345264)
    assert initial_guess.iloc[1].stress == pytest.approx(1903.4683072526843)
    assert initial_guess.iloc[1].volume == pytest.approx(1191.6)
    # Test that the site with low b1 and b2 returns NaN
    assert np.isnan(initial_guess.iloc[2].volume)
    assert np.isnan(initial_guess.iloc[2].stress)
    assert np.isnan(initial_guess.iloc[2].deflection)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = CantileveredBeamContinuous()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 4, "Expected 4 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check the names of variables
    expected_variable_names = ["hc", "b1", "b2", "H"]
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
    expected_var_bounds = ((0.1, 1.0), (2.0, 12.0), (0.1, 2.0), (3.0, 7.0))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (0.55, 7.0, 1.05, 5.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds & scales
    expected_resp_bounds = ((0.0, 0.1), (0.0, 5000.0), (0.0, 1200.0))
    expected_resp_scales = [100, 0.001, 0.01]
    for resp, expected_bounds, expected_scales in zip(
        opt_problem.responses, expected_resp_bounds, expected_resp_scales
    ):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"
        assert (
            resp.scale == expected_scales
        ), f"Response bounds do not match for response {resp}: {resp.scale} !=  {expected_scales}"

        # Define the description of the problem
        expected_description = """\
Cantilevered Beam Example Evaluator with only continuous variables

This example is making the first variable (x0) double instead of integer, so it
is a relaxation of the CantileveredBeam example.

The problem has four design variables with types as follows:

- hc: Height of the cantilever beam (double)
- b1: Width of the first section of the beam (double)
- b2: Width of the second section of the beam (double)
- H: Height of the beam (double)

There are three responses:

- deflection: The deflection of the beam (bounded between 0.0 and 0.1)
- stress: The stress experienced by the beam (bounded between 0.0 and 5000.0)
- volume: The volume of the beam (bounded between 0.0 and 1200.0)

Objectives:

- Minimize the volume of the beam.

Constraints:

- The volume, stress, and deflection of the beam must meet specified limits.

.. note:: The source for this instantiation of this evaluator is
    a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_
    called *"SHERPA - An Efficient and Robust Optimization/Search
    Algorithm"*."""

    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = 'The source for this instantiation of this evaluator is a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_ called *"SHERPA - An Efficient and Robust Optimization/Search Algorithm"*.'
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = CantileveredBeamContinuous()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].deflection == pytest.approx(0.16550077056507492)
    assert initial_guess_df.iloc[0].stress == pytest.approx(3447.932720105728)
    assert initial_guess_df.iloc[0].volume == pytest.approx(707.7)
