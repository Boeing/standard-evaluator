"""Unit tests for the SpeedReducer evaluator.

Tests instantiation, problem structure, feasible and infeasible point evaluation.

Requirements: 7.1, 7.2, 7.3, 7.6
"""

import numpy as np
import pandas as pd
import pytest

from standard_evaluator.evaluators.test.speed_reducer import SpeedReducer
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class TestSpeedReducerInstantiation:
    """Tests for SpeedReducer instantiation (Requirement 7.1)."""

    def test_instantiation_succeeds(self):
        """SpeedReducer instantiates without error."""
        evaluator = SpeedReducer()
        assert evaluator is not None

    def test_is_test_evaluator_instance(self):
        """SpeedReducer is a TestEvaluator instance."""
        evaluator = SpeedReducer()
        assert isinstance(evaluator, TestEvaluator)

    def test_test_info_populated(self):
        """SpeedReducer test_info is a non-empty dict."""
        evaluator = SpeedReducer()
        info = evaluator.test_info
        assert isinstance(info, dict)
        assert len(info) > 0


class TestSpeedReducerProblemStructure:
    """Tests for SpeedReducer problem structure (Requirement 7.2)."""

    @pytest.fixture
    def opt_problem(self):
        """Create the SpeedReducer OptProblem."""
        return SpeedReducer()._create_opt_problem()

    def test_seven_variables(self, opt_problem):
        """Problem has exactly 7 variables."""
        assert len(opt_problem.variables) == 7

    def test_variable_names(self, opt_problem):
        """Variables are named x1 through x7."""
        expected = [f"x{i}" for i in range(1, 8)]
        actual = [var.name for var in opt_problem.variables]
        assert actual == expected

    def test_variable_bounds(self, opt_problem):
        """Variables have the correct bounds."""
        expected_bounds = [
            (2.6, 3.6),
            (0.7, 0.8),
            (17.0, 28.0),
            (7.3, 8.3),
            (7.8, 8.3),
            (2.9, 3.9),
            (5.0, 5.5),
        ]
        for var, expected in zip(opt_problem.variables, expected_bounds):
            assert var.bounds == pytest.approx(expected), (
                f"Variable {var.name} bounds {var.bounds} != {expected}"
            )

    def test_twelve_responses(self, opt_problem):
        """Problem has exactly 12 responses (f + g1-g11)."""
        assert len(opt_problem.responses) == 12

    def test_response_names(self, opt_problem):
        """Responses are named f, g1, ..., g11."""
        expected = ["f"] + [f"g{i}" for i in range(1, 12)]
        actual = [resp.name for resp in opt_problem.responses]
        assert actual == expected

    def test_objective_response_bounds(self, opt_problem):
        """Objective f has bounds (-inf, inf)."""
        f_resp = opt_problem.responses[0]
        assert f_resp.name == "f"
        assert f_resp.bounds == (-np.inf, np.inf)

    def test_constraint_response_bounds(self, opt_problem):
        """Constraints g1-g11 have bounds (-inf, 0]."""
        for resp in opt_problem.responses[1:]:
            assert resp.bounds == (-np.inf, 0.0), (
                f"Response {resp.name} bounds {resp.bounds} != (-inf, 0.0)"
            )

    def test_objectives(self, opt_problem):
        """Objectives list is ['f']."""
        assert opt_problem.objectives == ["f"]

    def test_constraints(self, opt_problem):
        """Constraints list is ['g1', ..., 'g11']."""
        expected = [f"g{i}" for i in range(1, 12)]
        assert opt_problem.constraints == expected

    def test_initial_guess_midpoints(self, opt_problem):
        """Initial guess is the midpoints of variable bounds."""
        expected_midpoints = [3.1, 0.75, 22.5, 7.8, 8.05, 3.4, 5.25]
        actual = [var.default for var in opt_problem.variables]
        assert actual == pytest.approx(expected_midpoints)


class TestSpeedReducerFeasiblePoint:
    """Tests for evaluation at a known feasible point (Requirement 7.3).

    Point: (3.5, 0.7, 17, 7.3, 7.8, 3.352, 5.287)
    All constraints g1-g11 should be <= 0.
    """

    @pytest.fixture
    def evaluated_sites(self):
        """Evaluate the SpeedReducer at the known feasible point."""
        evaluator = SpeedReducer()
        sites = pd.DataFrame(
            {
                "x1": [3.5],
                "x2": [0.7],
                "x3": [17.0],
                "x4": [7.3],
                "x5": [7.8],
                "x6": [3.352],
                "x7": [5.287],
            }
        )
        evaluator(sites)
        return sites

    def test_all_constraints_feasible(self, evaluated_sites):
        """All constraints g1-g11 are <= 0 at the feasible point."""
        for i in range(1, 12):
            col = f"g{i}"
            value = evaluated_sites[col].iloc[0]
            assert value <= 0.0 + 1e-8, (
                f"Constraint {col} = {value} > 0 at feasible point"
            )

    def test_objective_is_finite(self, evaluated_sites):
        """Objective f is finite at the feasible point."""
        f_value = evaluated_sites["f"].iloc[0]
        assert np.isfinite(f_value)

    def test_all_responses_finite(self, evaluated_sites):
        """All responses are finite at the feasible point."""
        responses = ["f"] + [f"g{i}" for i in range(1, 12)]
        for col in responses:
            value = evaluated_sites[col].iloc[0]
            assert np.isfinite(value), f"Response {col} = {value} is not finite"


class TestSpeedReducerInfeasiblePoint:
    """Tests for evaluation at a known infeasible point (Requirement 7.3).

    Point: bounds midpoints (3.1, 0.75, 22.5, 7.8, 8.05, 3.4, 5.25)
    At least one constraint g_j > 0.
    """

    @pytest.fixture
    def evaluated_sites(self):
        """Evaluate the SpeedReducer at the bounds midpoints."""
        evaluator = SpeedReducer()
        sites = pd.DataFrame(
            {
                "x1": [3.1],
                "x2": [0.75],
                "x3": [22.5],
                "x4": [7.8],
                "x5": [8.05],
                "x6": [3.4],
                "x7": [5.25],
            }
        )
        evaluator(sites)
        return sites

    def test_at_least_one_constraint_violated(self, evaluated_sites):
        """At least one constraint g_j > 0 at the infeasible point."""
        constraint_values = [
            evaluated_sites[f"g{i}"].iloc[0] for i in range(1, 12)
        ]
        assert any(g > 0 for g in constraint_values), (
            f"Expected at least one violated constraint, but all <= 0: "
            f"{constraint_values}"
        )

    def test_all_responses_finite(self, evaluated_sites):
        """All responses are finite at the infeasible point."""
        responses = ["f"] + [f"g{i}" for i in range(1, 12)]
        for col in responses:
            value = evaluated_sites[col].iloc[0]
            assert np.isfinite(value), f"Response {col} = {value} is not finite"


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

from hypothesis import given, settings
from hypothesis import strategies as st


# Feature: benchmark-test-problems, Property 6: SpeedReducer all-responses formula correctness
class TestSpeedReducerFormulaCorrectness:
    """Property 6: SpeedReducer all-responses formula correctness.

    For any 7-tuple (x1, ..., x7) within the defined variable bounds,
    evaluating SpeedReducer SHALL produce the objective f and constraints
    g1-g11 exactly as defined by the speed reducer formulas, within
    floating-point tolerance.

    **Validates: Requirements 5.4, 5.5**
    """

    @given(
        x1=st.floats(min_value=2.6, max_value=3.6, allow_nan=False, allow_infinity=False),
        x2=st.floats(min_value=0.7, max_value=0.8, allow_nan=False, allow_infinity=False),
        x3=st.floats(min_value=17.0, max_value=28.0, allow_nan=False, allow_infinity=False),
        x4=st.floats(min_value=7.3, max_value=8.3, allow_nan=False, allow_infinity=False),
        x5=st.floats(min_value=7.8, max_value=8.3, allow_nan=False, allow_infinity=False),
        x6=st.floats(min_value=2.9, max_value=3.9, allow_nan=False, allow_infinity=False),
        x7=st.floats(min_value=5.0, max_value=5.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_objective_formula(self, x1, x2, x3, x4, x5, x6, x7):
        """The objective f matches the speed reducer weight formula.

        **Validates: Requirements 5.4**
        """
        evaluator = SpeedReducer()
        sites = pd.DataFrame(
            {"x1": [x1], "x2": [x2], "x3": [x3], "x4": [x4], "x5": [x5], "x6": [x6], "x7": [x7]}
        )
        evaluator(sites)

        # Independently compute expected f
        expected_f = (
            0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )

        actual_f = sites.iloc[0]["f"]
        np.testing.assert_allclose(actual_f, expected_f, rtol=1e-10)

    @given(
        x1=st.floats(min_value=2.6, max_value=3.6, allow_nan=False, allow_infinity=False),
        x2=st.floats(min_value=0.7, max_value=0.8, allow_nan=False, allow_infinity=False),
        x3=st.floats(min_value=17.0, max_value=28.0, allow_nan=False, allow_infinity=False),
        x4=st.floats(min_value=7.3, max_value=8.3, allow_nan=False, allow_infinity=False),
        x5=st.floats(min_value=7.8, max_value=8.3, allow_nan=False, allow_infinity=False),
        x6=st.floats(min_value=2.9, max_value=3.9, allow_nan=False, allow_infinity=False),
        x7=st.floats(min_value=5.0, max_value=5.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_all_constraints_formula(self, x1, x2, x3, x4, x5, x6, x7):
        """All constraints g1-g11 match the speed reducer constraint formulas.

        **Validates: Requirements 5.5**
        """
        evaluator = SpeedReducer()
        sites = pd.DataFrame(
            {"x1": [x1], "x2": [x2], "x3": [x3], "x4": [x4], "x5": [x5], "x6": [x6], "x7": [x7]}
        )
        evaluator(sites)

        # Independently compute expected constraints
        expected_g1 = 27.0 / (x1 * x2**2 * x3) - 1.0
        expected_g2 = 397.5 / (x1 * x2**2 * x3**2) - 1.0
        expected_g3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0
        expected_g4 = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0
        expected_g5 = (
            np.sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 16.9e6)
            / (110.0 * x6**3) - 1.0
        )
        expected_g6 = (
            np.sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 157.5e6)
            / (85.0 * x7**3) - 1.0
        )
        expected_g7 = x2 * x3 / 40.0 - 1.0
        expected_g8 = 5.0 * x2 / x1 - 1.0
        expected_g9 = x1 / (12.0 * x2) - 1.0
        expected_g10 = (1.5 * x6 + 1.9) / x4 - 1.0
        expected_g11 = (1.1 * x7 + 1.9) / x5 - 1.0

        row = sites.iloc[0]
        np.testing.assert_allclose(row["g1"], expected_g1, rtol=1e-10)
        np.testing.assert_allclose(row["g2"], expected_g2, rtol=1e-10)
        np.testing.assert_allclose(row["g3"], expected_g3, rtol=1e-10)
        np.testing.assert_allclose(row["g4"], expected_g4, rtol=1e-10)
        np.testing.assert_allclose(row["g5"], expected_g5, rtol=1e-10)
        np.testing.assert_allclose(row["g6"], expected_g6, rtol=1e-10)
        np.testing.assert_allclose(row["g7"], expected_g7, rtol=1e-10)
        np.testing.assert_allclose(row["g8"], expected_g8, rtol=1e-10)
        np.testing.assert_allclose(row["g9"], expected_g9, rtol=1e-10)
        np.testing.assert_allclose(row["g10"], expected_g10, rtol=1e-10)
        np.testing.assert_allclose(row["g11"], expected_g11, rtol=1e-10)
