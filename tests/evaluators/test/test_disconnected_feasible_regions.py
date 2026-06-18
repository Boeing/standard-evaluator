"""Testing of the DisconnectedFeasibleRegions class"""

import pytest
import pandas as pd
import numpy as np

from pydantic import ValidationError

from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.evaluators import TestEvaluator
from standard_evaluator.evaluators.test import DisconnectedFeasibleRegions
from standard_evaluator.evaluators.test.disconnected_feasible_regions import (
    DisconnectedFeasibleRegionsOptions,
)


# ===========================================================================
# Unit Tests (Requirements 7.1, 7.2, 7.3, 7.4, 7.6)
# ===========================================================================


def test_instantiation():
    """Test that DisconnectedFeasibleRegions instantiation produces a TestEvaluator instance."""
    test_func = DisconnectedFeasibleRegions()
    assert isinstance(test_func, TestEvaluator)


def test_problem_structure():
    """Test problem structure: variables, responses, objectives, constraints."""
    opt_problem = DisconnectedFeasibleRegions()._create_opt_problem()

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


def test_feasible_point_at_first_island_center_nonsmooth():
    """Test feasible point at first island center (-0.5, -0.4): g1 <= 0 with smooth=False."""
    test_func = DisconnectedFeasibleRegions()
    sites = pd.DataFrame({"x1": [-0.5], "x2": [-0.4]})
    test_func(sites)

    # At the first island center, d_1 = 0 - r_1^2 = -0.15^2 = -0.0225
    # min over all islands includes d_1 = -0.0225, so g1 = -0.0225
    assert sites.iloc[0]["g1"] == pytest.approx(-0.0225, abs=1e-8)
    assert sites.iloc[0]["g1"] <= 0


def test_feasible_point_at_first_island_center_smooth():
    """Test feasible point at first island center (-0.5, -0.4): g1 <= 0 with smooth=True."""
    opts = DisconnectedFeasibleRegionsOptions(smooth=True)
    test_func = DisconnectedFeasibleRegions(options=opts)
    sites = pd.DataFrame({"x1": [-0.5], "x2": [-0.4]})
    test_func(sites)

    # With smooth formulation, g1 should still be <= 0 at the island center
    assert sites.iloc[0]["g1"] <= 0
    # The smooth approximation should be close to the nonsmooth value
    assert np.isfinite(sites.iloc[0]["g1"])


def test_infeasible_point_at_origin_nonsmooth():
    """Test infeasible point at origin (0.0, 0.0): g1 > 0 with smooth=False."""
    test_func = DisconnectedFeasibleRegions()
    sites = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    test_func(sites)

    # At origin: distances to all island centers are well outside the radii
    assert sites.iloc[0]["g1"] > 0


def test_infeasible_point_at_origin_smooth():
    """Test infeasible point at origin (0.0, 0.0): g1 > 0 with smooth=True."""
    opts = DisconnectedFeasibleRegionsOptions(smooth=True)
    test_func = DisconnectedFeasibleRegions(options=opts)
    sites = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    test_func(sites)

    # With smooth formulation, origin should still be infeasible
    assert sites.iloc[0]["g1"] > 0
    assert np.isfinite(sites.iloc[0]["g1"])


def test_pydantic_validation_mismatched_centers_radii():
    """Test Pydantic validation rejects mismatched centers/radii lengths."""
    with pytest.raises(ValidationError, match="centers and radii must have the same length"):
        DisconnectedFeasibleRegionsOptions(
            centers=[(-0.5, -0.4), (0.4, 0.3)],
            radii=[0.15, 0.12, 0.10],
        )


def test_pydantic_validation_tau_must_be_positive():
    """Test Pydantic validation rejects non-positive tau."""
    with pytest.raises(ValidationError, match="tau must be greater than 0"):
        DisconnectedFeasibleRegionsOptions(tau=0.0)

    with pytest.raises(ValidationError, match="tau must be greater than 0"):
        DisconnectedFeasibleRegionsOptions(tau=-0.5)


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

# Strategy for generating valid island configurations (1-5 islands)
@st.composite
def island_configs(draw):
    """Generate valid island configurations: centers and radii of same length (1-5 islands)."""
    num_islands = draw(st.integers(min_value=1, max_value=5))
    centers = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=num_islands,
            max_size=num_islands,
        )
    )
    radii = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=num_islands,
            max_size=num_islands,
        )
    )
    return centers, radii


# Feature: benchmark-test-problems, Property 2: DisconnectedFeasibleRegions nonsmooth formula correctness
@given(
    x1=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    config=island_configs(),
)
@settings(max_examples=100)
def test_disconnected_feasible_regions_nonsmooth_formula_correctness(x1, x2, config):
    """Property 2: For any (x1, x2) and any valid island configuration (centers, radii)
    with smooth=False, evaluating DisconnectedFeasibleRegions SHALL produce g1 equal to
    the minimum over all islands of [(x1-a_l)^2 + (x2-b_l)^2 - r_l^2],
    within floating-point tolerance.

    Feature: benchmark-test-problems
    Property 2: DisconnectedFeasibleRegions nonsmooth formula correctness

    **Validates: Requirements 2.6**
    """
    centers, radii = config

    # Instantiate evaluator with custom options (smooth=False)
    custom_opts = DisconnectedFeasibleRegionsOptions(
        centers=centers, radii=radii, smooth=False
    )
    evaluator = DisconnectedFeasibleRegions(options=custom_opts)

    # Create input DataFrame
    sites = pd.DataFrame({"x1": [x1], "x2": [x2]})

    # Evaluate
    evaluator(sites)

    # Independently compute expected g1 = min over l of [(x1-a_l)^2 + (x2-b_l)^2 - r_l^2]
    d_values = [
        (x1 - a) ** 2 + (x2 - b) ** 2 - r**2
        for (a, b), r in zip(centers, radii)
    ]
    expected_g1 = min(d_values)

    # Assert evaluator output matches within 1e-10 relative tolerance
    actual_g1 = sites["g1"].iloc[0]
    assert actual_g1 == pytest.approx(expected_g1, rel=1e-10, abs=1e-10), (
        f"Formula mismatch: evaluator g1={actual_g1}, expected g1={expected_g1} "
        f"for x1={x1}, x2={x2}, centers={centers}, radii={radii}"
    )


# Feature: benchmark-test-problems, Property 3: DisconnectedFeasibleRegions smooth formula correctness
@given(
    x1=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    config=island_configs(),
    tau=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_disconnected_feasible_regions_smooth_formula_correctness(x1, x2, config, tau):
    """Property 3: For any (x1, x2) and valid island configuration (centers, radii)
    with smooth=True and tau > 0, evaluating DisconnectedFeasibleRegions SHALL produce
    g1 = -tau * log(sum_l exp(-d_l / tau)) where d_l = (x1-a_l)^2 + (x2-b_l)^2 - r_l^2,
    within floating-point tolerance.

    Feature: benchmark-test-problems
    Property 3: DisconnectedFeasibleRegions smooth formula correctness

    **Validates: Requirements 2.7**
    """
    centers, radii = config

    # Instantiate evaluator with smooth=True and specified tau
    opts = DisconnectedFeasibleRegionsOptions(
        centers=centers, radii=radii, smooth=True, tau=tau
    )
    evaluator = DisconnectedFeasibleRegions(options=opts)

    # Create input DataFrame
    sites = pd.DataFrame({"x1": [x1], "x2": [x2]})

    # Evaluate
    evaluator(sites)

    # Independently compute expected g1 using log-sum-exp trick
    d = np.array([(x1 - a) ** 2 + (x2 - b) ** 2 - r**2 for (a, b), r in zip(centers, radii)])
    shifted = -d / tau
    max_val = np.max(shifted)
    expected_g1 = -tau * (max_val + np.log(np.sum(np.exp(shifted - max_val))))

    # Assert evaluator output matches within 1e-10 relative tolerance
    actual_g1 = sites["g1"].iloc[0]
    assert actual_g1 == pytest.approx(expected_g1, rel=1e-10, abs=1e-10), (
        f"Smooth formula mismatch: evaluator g1={actual_g1}, expected g1={expected_g1} "
        f"for x1={x1}, x2={x2}, tau={tau}, centers={centers}, radii={radii}"
    )
