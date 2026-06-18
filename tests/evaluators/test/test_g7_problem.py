# Feature: benchmark-test-problems, Property 5: G7Problem all-responses formula correctness
"""Tests for the G7Problem evaluator.

Unit tests validate: Requirements 7.1, 7.2, 7.3, 7.6
Property tests validate: Requirements 4.4, 4.5
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.evaluators.test.g7_problem import G7Problem
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


# ===========================================================================
# Unit Tests (Requirements 7.1, 7.2, 7.3, 7.6)
# ===========================================================================


def test_instantiation():
    """Test that G7Problem instantiates as a TestEvaluator instance.

    Requirements: 7.1
    """
    evaluator = G7Problem()
    assert isinstance(evaluator, TestEvaluator)
    assert evaluator.test_info is not None
    assert evaluator.test_info["n_vars"] == 10
    assert evaluator.test_info["test_goal"] == "optimization"


def test_problem_structure():
    """Test G7Problem has correct structure: 10 variables, 9 responses, objectives, constraints.

    Requirements: 7.2
    """
    evaluator = G7Problem()
    opt_problem = evaluator._create_opt_problem()

    # 10 variables
    assert len(opt_problem.variables) == 10
    var_names = [var.name for var in opt_problem.variables]
    expected_var_names = [f"x{i + 1}" for i in range(10)]
    assert var_names == expected_var_names

    # All variables bounded [-10, 10]
    for var in opt_problem.variables:
        assert var.bounds == (-10.0, 10.0), (
            f"Variable {var.name} bounds {var.bounds} != (-10.0, 10.0)"
        )

    # 9 responses: f + g1-g8
    assert len(opt_problem.responses) == 9
    resp_names = [resp.name for resp in opt_problem.responses]
    expected_resp_names = ["f"] + [f"g{i + 1}" for i in range(8)]
    assert resp_names == expected_resp_names

    # Response bounds: f is (-inf, inf), g1-g8 are (-inf, 0]
    f_resp = opt_problem.responses[0]
    assert f_resp.bounds == (-np.inf, np.inf)
    for resp in opt_problem.responses[1:]:
        assert resp.bounds == (-np.inf, 0.0), (
            f"Response {resp.name} bounds {resp.bounds} != (-inf, 0.0)"
        )

    # Objectives and constraints
    assert opt_problem.objectives == ["f"]
    assert opt_problem.constraints == [f"g{i + 1}" for i in range(8)]


def test_feasible_point_from_literature():
    """Test known optimal solution from literature: all g1-g8 <= 0.

    Uses x = (2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
              1.430574, 1.321644, 9.828726, 8.280092, 8.375927)
    which is the known optimal solution from Hock & Schittkowski (1981).

    Requirements: 7.3
    """
    evaluator = G7Problem()

    # Known optimal solution from literature
    x_opt = [
        2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
        1.430574, 1.321644, 9.828726, 8.280092, 8.375927,
    ]

    sites = pd.DataFrame(
        [x_opt],
        columns=[f"x{i + 1}" for i in range(10)],
    )

    evaluator(sites)

    # All constraints g1-g8 should be <= 0 (feasible)
    # Use tolerance of 1e-4 since the known solution values are rounded
    # and active constraints may be numerically slightly positive.
    for i in range(1, 9):
        col = f"g{i}"
        assert sites.iloc[0][col] <= 1e-4, (
            f"Constraint {col} = {sites.iloc[0][col]} > 0 at known optimal solution"
        )


def test_infeasible_point_at_origin():
    """Test that the origin (0,...,0) is infeasible: at least one g_j > 0.

    Requirements: 7.6
    """
    evaluator = G7Problem()

    sites = pd.DataFrame(
        [[0.0] * 10],
        columns=[f"x{i + 1}" for i in range(10)],
    )

    evaluator(sites)

    # At least one constraint should be violated (> 0)
    constraint_values = [sites.iloc[0][f"g{i}"] for i in range(1, 9)]
    assert any(g > 0 for g in constraint_values), (
        f"Expected at least one g_j > 0 at origin, got {constraint_values}"
    )


# ===========================================================================
# Property-Based Tests (Requirements 4.4, 4.5)
# ===========================================================================

# Hypothesis Strategy: random 10-tuple within [-10, 10]^10
_bounded_float = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


class TestG7ProblemFormulaCorrectness:
    """Property 5: G7Problem all-responses formula correctness.

    For any 10-tuple (x1, ..., x10) within [-10, 10]^10, evaluating
    G7Problem SHALL produce the objective f and constraints g1-g8 exactly
    as defined by the G7 formulas, within floating-point tolerance.

    **Validates: Requirements 4.4, 4.5**
    """

    @given(
        x1=_bounded_float,
        x2=_bounded_float,
        x3=_bounded_float,
        x4=_bounded_float,
        x5=_bounded_float,
        x6=_bounded_float,
        x7=_bounded_float,
        x8=_bounded_float,
        x9=_bounded_float,
        x10=_bounded_float,
    )
    @settings(max_examples=100)
    def test_objective_formula(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        """The G7 objective f matches the independent computation.

        **Validates: Requirements 4.4**
        """
        evaluator = G7Problem()
        sites = pd.DataFrame(
            {"x1": [x1], "x2": [x2], "x3": [x3], "x4": [x4], "x5": [x5],
             "x6": [x6], "x7": [x7], "x8": [x8], "x9": [x9], "x10": [x10]}
        )
        evaluator(sites)

        # Independently compute expected f
        expected_f = (
            x1**2 + x2**2 + x1 * x2
            - 14 * x1 - 16 * x2
            + (x3 - 10) ** 2
            + 4 * (x4 - 5) ** 2
            + (x5 - 3) ** 2
            + 2 * (x6 - 1) ** 2
            + 5 * x7**2
            + 7 * (x8 - 11) ** 2
            + 2 * (x9 - 10) ** 2
            + (x10 - 7) ** 2
            + 45
        )

        np.testing.assert_allclose(sites["f"].iloc[0], expected_f, rtol=1e-10)

    @given(
        x1=_bounded_float,
        x2=_bounded_float,
        x3=_bounded_float,
        x4=_bounded_float,
        x5=_bounded_float,
        x6=_bounded_float,
        x7=_bounded_float,
        x8=_bounded_float,
        x9=_bounded_float,
        x10=_bounded_float,
    )
    @settings(max_examples=100)
    def test_constraints_formula(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        """The G7 constraints g1-g8 match the independent computation.

        **Validates: Requirements 4.5**
        """
        evaluator = G7Problem()
        sites = pd.DataFrame(
            {"x1": [x1], "x2": [x2], "x3": [x3], "x4": [x4], "x5": [x5],
             "x6": [x6], "x7": [x7], "x8": [x8], "x9": [x9], "x10": [x10]}
        )
        evaluator(sites)

        # Independently compute expected constraints
        expected_g1 = -105 + 4 * x1 + 5 * x2 - 3 * x7 + 9 * x8
        expected_g2 = 10 * x1 - 8 * x2 - 17 * x7 + 2 * x8
        expected_g3 = -8 * x1 + 2 * x2 + 5 * x9 - 2 * x10 - 12
        expected_g4 = (
            3 * (x1 - 2) ** 2 + 4 * (x2 - 3) ** 2
            + 2 * x3**2 - 7 * x4 - 120
        )
        expected_g5 = 5 * x1**2 + 8 * x2 + (x3 - 6) ** 2 - 2 * x4 - 40
        expected_g6 = (
            x1**2 + 2 * (x2 - 2) ** 2
            - 2 * x1 * x2 + 14 * x5 - 6 * x6
        )
        expected_g7 = (
            0.5 * (x1 - 8) ** 2 + 2 * (x2 - 4) ** 2
            + 3 * x5**2 - x6 - 30
        )
        expected_g8 = -3 * x1 + 6 * x2 + 12 * (x9 - 8) ** 2 - 7 * x10

        np.testing.assert_allclose(sites["g1"].iloc[0], expected_g1, rtol=1e-10)
        np.testing.assert_allclose(sites["g2"].iloc[0], expected_g2, rtol=1e-10)
        np.testing.assert_allclose(sites["g3"].iloc[0], expected_g3, rtol=1e-10)
        np.testing.assert_allclose(sites["g4"].iloc[0], expected_g4, rtol=1e-10)
        np.testing.assert_allclose(sites["g5"].iloc[0], expected_g5, rtol=1e-10)
        np.testing.assert_allclose(sites["g6"].iloc[0], expected_g6, rtol=1e-10)
        np.testing.assert_allclose(sites["g7"].iloc[0], expected_g7, rtol=1e-10)
        np.testing.assert_allclose(sites["g8"].iloc[0], expected_g8, rtol=1e-10)
