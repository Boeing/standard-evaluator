"""Testing of the G6Problem class."""

import pytest
import pandas as pd
import numpy as np

from standard_evaluator.evaluators.test import G6Problem
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


def test_g6_instantiation():
    """Test that G6Problem instantiation produces a TestEvaluator instance."""
    test_func = G6Problem()
    assert isinstance(test_func, TestEvaluator)
    # Verify test_info is populated
    info = test_func.test_info
    assert info["test_goal"] == "optimization"
    assert info["n_vars"] == 2
    assert info["n_constraints"] == 2
    assert info["n_inequality_constraints"] == 2
    assert info["problem_type"] == "continuous"


def test_g6_problem_structure():
    """Test G6Problem problem structure (variables, bounds, responses, objectives, constraints)."""
    opt_problem = G6Problem()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 2
    assert len(opt_problem.responses) == 3

    # Check variable names
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert actual_variable_names == ["x1", "x2"]

    # Check variable bounds
    expected_var_bounds = ([13.0, 100.0], [0.0, 100.0])
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert var.bounds == tuple(expected_bounds), (
            f"Variable bounds do not match for {var.name}: "
            f"{var.bounds} != {tuple(expected_bounds)}"
        )

    # Check initial guess (defaults)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert actual_initial_guess == (20.0, 5.5)

    # Check response names
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert actual_response_names == ["f", "g1", "g2"]

    # Check response bounds
    expected_resp_bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
    for resp, expected_bounds in zip(opt_problem.responses, expected_resp_bounds):
        assert resp.bounds == tuple(expected_bounds), (
            f"Response bounds do not match for {resp.name}: "
            f"{resp.bounds} != {tuple(expected_bounds)}"
        )

    # Check objectives
    assert opt_problem.objectives == ["f"]

    # Check constraints
    assert opt_problem.constraints == ["g1", "g2"]


def test_g6_feasible_point():
    """Test that the known optimum (14.095, 0.84296) is feasible: g1 ≤ 0 and g2 ≤ 0.

    The known optimum coordinates are approximate (rounded), so constraint
    values may be very slightly positive due to rounding. We use a small
    tolerance (1e-4) to account for this.
    """
    test_func = G6Problem()
    sites = pd.DataFrame({"x1": [14.095], "x2": [0.84296]})
    test_func(sites)

    # Both constraints must be approximately satisfied (≤ 0 within tolerance)
    assert sites.iloc[0].g1 <= 1e-4, f"g1 = {sites.iloc[0].g1} should be ≤ 0 (within tol)"
    assert sites.iloc[0].g2 <= 1e-4, f"g2 = {sites.iloc[0].g2} should be ≤ 0 (within tol)"

    # Verify objective value at the known optimum
    expected_f = (14.095 - 10.0) ** 3 + (0.84296 - 20.0) ** 3
    assert sites.iloc[0].f == pytest.approx(expected_f, rel=1e-8)


def test_g6_infeasible_point():
    """Test that point (20.0, 50.0) is infeasible: at least one g_j > 0."""
    test_func = G6Problem()
    sites = pd.DataFrame({"x1": [20.0], "x2": [50.0]})
    test_func(sites)

    # At least one constraint must be violated (> 0)
    g_values = [sites.iloc[0].g1, sites.iloc[0].g2]
    assert any(g > 0 for g in g_values), (
        f"Expected at least one constraint > 0 at (20, 50), "
        f"got g1={g_values[0]}, g2={g_values[1]}"
    )


def test_g6_known_solution():
    """Test that known_solution property returns the correct values."""
    test_func = G6Problem()
    known_sol = test_func.known_solution

    # Should be a DataFrame with one row
    assert known_sol is not None
    assert len(known_sol) == 1

    # Check variable values at the known optimum
    assert known_sol.iloc[0].x1 == pytest.approx(14.095)
    assert known_sol.iloc[0].x2 == pytest.approx(0.84296)

    # Check that the objective value is approximately -6961.81388
    assert known_sol.iloc[0].f == pytest.approx(-6961.81388, rel=1e-4)

    # Check that constraints are approximately satisfied at the known solution
    # (the known solution coordinates are rounded, so allow small tolerance)
    assert known_sol.iloc[0].g1 <= 1e-4
    assert known_sol.iloc[0].g2 <= 1e-4



# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

from hypothesis import given, settings
from hypothesis import strategies as st

# Feature: benchmark-test-problems, Property 4: G6Problem all-responses formula correctness


class TestG6ProblemFormulaCorrectness:
    """Property 4: G6Problem all-responses formula correctness.

    For any pair (x1, x2) within [13, 100] x [0, 100], evaluating G6Problem
    SHALL produce f = (x1-10)^3 + (x2-20)^3, g1 = -(x1-5)^2 - (x2-5)^2 + 100,
    and g2 = (x1-6)^2 + (x2-5)^2 - 82.81, all within floating-point tolerance.

    **Validates: Requirements 3.4, 3.5, 3.6**
    """

    @given(
        x1=st.floats(min_value=13.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        x2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_all_responses_match_expected_formulas(self, x1, x2):
        """Evaluator output matches independently computed f, g1, g2.

        **Validates: Requirements 3.4, 3.5, 3.6**
        """
        import pandas as pd
        import numpy as np

        # Arrange: create evaluator and input DataFrame
        evaluator = G6Problem()
        sites = pd.DataFrame({"x1": [x1], "x2": [x2]})

        # Act: evaluate
        evaluator(sites)

        # Independently compute expected values
        expected_f = (x1 - 10.0) ** 3 + (x2 - 20.0) ** 3
        expected_g1 = -(x1 - 5.0) ** 2 - (x2 - 5.0) ** 2 + 100.0
        expected_g2 = (x1 - 6.0) ** 2 + (x2 - 5.0) ** 2 - 82.81

        # Assert: results match within 1e-10 relative tolerance
        np.testing.assert_allclose(sites["f"].iloc[0], expected_f, rtol=1e-10)
        np.testing.assert_allclose(sites["g1"].iloc[0], expected_g1, rtol=1e-10)
        np.testing.assert_allclose(sites["g2"].iloc[0], expected_g2, rtol=1e-10)
