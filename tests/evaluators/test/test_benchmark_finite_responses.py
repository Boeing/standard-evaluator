# Feature: benchmark-test-problems, Property 7: All evaluators produce finite responses for any input
"""Property test verifying all benchmark evaluators produce finite responses.

**Validates: Requirements 6.3, 6.4**

For any of the 5 new evaluators and for any input sites within the defined
variable bounds, calling the evaluator SHALL not raise an exception and SHALL
produce response values that are finite real numbers (not NaN, not ±inf).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.evaluators.test.disconnected_feasible_regions import (
    DisconnectedFeasibleRegions,
)
from standard_evaluator.evaluators.test.g6_problem import G6Problem
from standard_evaluator.evaluators.test.g7_problem import G7Problem
from standard_evaluator.evaluators.test.small_circle_feasible_region import (
    SmallCircleFeasibleRegion,
)
from standard_evaluator.evaluators.test.speed_reducer import SpeedReducer

# All 5 new evaluator classes to parametrize over
EVALUATOR_CLASSES = [
    SmallCircleFeasibleRegion,
    DisconnectedFeasibleRegions,
    G6Problem,
    G7Problem,
    SpeedReducer,
]


def _get_variable_bounds(evaluator_cls):
    """Retrieve variable names and bounds from an evaluator's OptProblem.

    Returns
    -------
    list of tuple
        Each element is (var_name, lower_bound, upper_bound).
    """
    evaluator = evaluator_cls()
    opt_problem = evaluator.opt_problem
    return [(var.name, var.bounds[0], var.bounds[1]) for var in opt_problem.variables]


def _build_strategy_for_evaluator(evaluator_cls):
    """Build a Hypothesis strategy that generates a dict of variable values
    within the evaluator's defined bounds.

    Returns
    -------
    SearchStrategy[dict]
        A strategy producing dicts mapping variable names to float values.
    """
    var_info = _get_variable_bounds(evaluator_cls)
    strategies = {}
    for var_name, lo, hi in var_info:
        strategies[var_name] = st.floats(
            min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False
        )
    return st.fixed_dictionaries(strategies)


@pytest.mark.parametrize("evaluator_cls", EVALUATOR_CLASSES, ids=lambda c: c.__name__)
def test_finite_responses_for_all_evaluators(evaluator_cls):
    """Property 7: All evaluators produce finite responses for any input within bounds.

    This test dynamically generates inputs within each evaluator's variable bounds
    and verifies all response values are finite.
    """
    evaluator = evaluator_cls()
    opt_problem = evaluator.opt_problem
    var_info = [(var.name, var.bounds[0], var.bounds[1]) for var in opt_problem.variables]
    response_names = [resp.name for resp in opt_problem.responses]

    # Use Hypothesis inside the parametrized test via an inner function
    @given(data=st.data())
    @settings(max_examples=100)
    def _check(data):
        # Generate values within bounds for each variable
        values = {}
        for var_name, lo, hi in var_info:
            values[var_name] = data.draw(
                st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False),
                label=var_name,
            )

        # Build a single-row DataFrame with the generated values
        sites = pd.DataFrame({name: [val] for name, val in values.items()})

        # Call the evaluator — must not raise any exception
        evaluator(sites)

        # Assert all response values are finite (not NaN, not ±inf)
        for resp_name in response_names:
            val = sites[resp_name].iloc[0]
            assert np.isfinite(val), (
                f"Evaluator {evaluator_cls.__name__} produced non-finite value "
                f"for response '{resp_name}': {val} with inputs {values}"
            )

    _check()
