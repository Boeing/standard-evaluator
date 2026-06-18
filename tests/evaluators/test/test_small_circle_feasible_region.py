"""Testing of the SmallCircleFeasibleRegion class"""

import pytest
import pandas as pd
import numpy as np

from standard_evaluator.evaluators import TestEvaluator
from standard_evaluator.evaluators.test import SmallCircleFeasibleRegion
from standard_evaluator.evaluators.test.small_circle_feasible_region import (
    SmallCircleFeasibleRegionOptions,
)


def test_instantiation():
    """Test that SmallCircleFeasibleRegion instantiation produces a TestEvaluator instance."""
    test_func = SmallCircleFeasibleRegion()
    assert isinstance(test_func, TestEvaluator)


def test_problem_structure():
    """Test problem structure: variable names, bounds, response names, objectives, constraints."""
    opt_problem = SmallCircleFeasibleRegion()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2, "Expected 2 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 response"

    # Check variable names
    expected_variable_names = ["x1", "x2"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert actual_variable_names == expected_variable_names

    # Check variable bounds
    expected_var_bounds = ((-1.0, 1.0), (-1.0, 1.0))
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert var.bounds == expected_bounds, (
            f"Variable bounds do not match for {var.name}: "
            f"{var.bounds} != {expected_bounds}"
        )

    # Check the initial guess
    expected_initial_guess = (0.0, 0.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert actual_initial_guess == expected_initial_guess

    # Check response names
    expected_response_names = ["g1"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert actual_response_names == expected_response_names

    # Check response bounds
    expected_resp_bounds = [(-np.inf, 0.0)]
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert resp.bounds == expected_bounds, (
            f"Response bounds do not match for {resp.name}: "
            f"{resp.bounds} != {expected_bounds}"
        )

    # Check objectives (should be empty for feasibility-only problem)
    assert opt_problem.objectives == []

    # Check constraints
    assert opt_problem.constraints == ["g1"]


def test_feasible_point_at_center():
    """Test feasible point at circle center (0.5, -0.3): g1 = -r² = -0.0225."""
    test_func = SmallCircleFeasibleRegion()
    sites = pd.DataFrame({"x1": [0.5], "x2": [-0.3]})
    test_func(sites)

    # At the center, g1 = (0.5-0.5)^2 + (-0.3-(-0.3))^2 - 0.15^2 = -0.0225
    assert sites.iloc[0]["g1"] == pytest.approx(-0.0225, abs=1e-8)
    # g1 <= 0 means feasible
    assert sites.iloc[0]["g1"] <= 0


def test_infeasible_point_at_origin():
    """Test infeasible point at origin (0.0, 0.0): g1 > 0."""
    test_func = SmallCircleFeasibleRegion()
    sites = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    test_func(sites)

    # At origin: g1 = (0-0.5)^2 + (0-(-0.3))^2 - 0.15^2 = 0.25 + 0.09 - 0.0225 = 0.3175
    assert sites.iloc[0]["g1"] > 0


def test_custom_options_override():
    """Test that custom options override default values."""
    # Test _define_options returns the correct class
    assert SmallCircleFeasibleRegion._define_options() is SmallCircleFeasibleRegionOptions

    # Test default option values
    test_func = SmallCircleFeasibleRegion()
    assert test_func.lookup_option_value("a") == pytest.approx(0.5)
    assert test_func.lookup_option_value("b") == pytest.approx(-0.3)
    assert test_func.lookup_option_value("r") == pytest.approx(0.15)

    # Test custom options
    custom_opts = SmallCircleFeasibleRegionOptions(a=0.0, b=0.0, r=0.5)
    custom_func = SmallCircleFeasibleRegion(options=custom_opts)
    assert custom_func.lookup_option_value("a") == pytest.approx(0.0)
    assert custom_func.lookup_option_value("b") == pytest.approx(0.0)
    assert custom_func.lookup_option_value("r") == pytest.approx(0.5)

    # Evaluate with custom options: center at (0, 0), radius 0.5
    # Point (0, 0) should be feasible with g1 = -0.25
    sites = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    custom_func(sites)
    assert sites.iloc[0]["g1"] == pytest.approx(-0.25, abs=1e-8)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st

# Feature: benchmark-test-problems, Property 1: SmallCircleFeasibleRegion formula correctness
@given(
    x1=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    a=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    r=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_small_circle_feasible_region_formula_correctness(x1, x2, a, b, r):
    """Property 1: For any (x1, x2) and valid (a, b, r), evaluating
    SmallCircleFeasibleRegion SHALL produce g1 = (x1-a)^2 + (x2-b)^2 - r^2
    within floating-point tolerance.

    Feature: benchmark-test-problems
    Property 1: SmallCircleFeasibleRegion formula correctness

    **Validates: Requirements 1.6**
    """
    # Instantiate evaluator with custom options
    custom_opts = SmallCircleFeasibleRegionOptions(a=a, b=b, r=r)
    evaluator = SmallCircleFeasibleRegion(options=custom_opts)

    # Create input DataFrame
    sites = pd.DataFrame({"x1": [x1], "x2": [x2]})

    # Evaluate
    evaluator(sites)

    # Independently compute expected g1
    expected_g1 = (x1 - a) ** 2 + (x2 - b) ** 2 - r**2

    # Assert evaluator output matches within 1e-10 relative tolerance
    actual_g1 = sites["g1"].iloc[0]
    assert actual_g1 == pytest.approx(expected_g1, rel=1e-10, abs=1e-10), (
        f"Formula mismatch: evaluator g1={actual_g1}, expected g1={expected_g1} "
        f"for x1={x1}, x2={x2}, a={a}, b={b}, r={r}"
    )
