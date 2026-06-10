"""Property-based tests for PyEvaluator and constructor validation.

Property 2: PyEvaluator functional equivalence
Property 3: Evaluator constructor input validation

Validates: Requirements 3.4, 3.5, 6.5
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.strategies import composite

from standard_evaluator import FloatVariable, OptProblem
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.evaluators.evaluator import PyEvaluator
from standard_evaluator.evaluators.shift_scale_evaluator import ShiftScaleEvaluator


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@composite
def simple_problem_and_func(draw):
    """Generate a valid OptProblem with FloatVariables and a deterministic callable.

    Returns a tuple of (opt_problem, func, input_df) where:
    - opt_problem has N variables and M responses
    - func is a callable that takes a DataFrame and modifies it in-place
      by computing polynomial responses from input columns
    - input_df is a valid DataFrame with values within bounds
    """
    n_vars = draw(st.integers(min_value=1, max_value=4))
    n_responses = draw(st.integers(min_value=1, max_value=3))
    n_sites = draw(st.integers(min_value=1, max_value=5))

    # Generate variables with valid non-degenerate bounds
    variables = []
    for i in range(n_vars):
        lower = draw(st.floats(min_value=-10.0, max_value=9.0,
                               allow_nan=False, allow_infinity=False))
        upper = draw(st.floats(min_value=lower + 0.5, max_value=10.0,
                               allow_nan=False, allow_infinity=False))
        variables.append(FloatVariable(name=f"x{i}", bounds=[lower, upper]))

    # Generate responses
    responses = []
    for i in range(n_responses):
        responses.append(FloatVariable(name=f"y{i}", bounds=[-1000.0, 1000.0]))

    # Build an OptProblem
    opt_problem = OptProblem(
        name="test_eval",
        variables=variables,
        responses=responses,
        objectives=["y0"],
    )

    # Generate coefficients for a simple polynomial: y_j = sum(c_ji * x_i) + offset_j
    coefficients = []
    offsets = []
    for j in range(n_responses):
        coefs = [draw(st.floats(min_value=-5.0, max_value=5.0,
                                allow_nan=False, allow_infinity=False))
                 for _ in range(n_vars)]
        offset = draw(st.floats(min_value=-5.0, max_value=5.0,
                                allow_nan=False, allow_infinity=False))
        coefficients.append(coefs)
        offsets.append(offset)

    # Create the callable function
    var_names = [f"x{i}" for i in range(n_vars)]
    resp_names = [f"y{i}" for i in range(n_responses)]

    def eval_func(sites: pd.DataFrame) -> None:
        """Compute linear polynomial responses in-place."""
        for j, resp_name in enumerate(resp_names):
            result = np.zeros(len(sites))
            for i, var_name in enumerate(var_names):
                result += coefficients[j][i] * sites[var_name].values
            result += offsets[j]
            sites[resp_name] = result

    # Generate input DataFrame with values within bounds
    data = {}
    for i, var in enumerate(variables):
        lb, ub = var.bounds
        values = [draw(st.floats(min_value=lb, max_value=ub,
                                 allow_nan=False, allow_infinity=False))
                  for _ in range(n_sites)]
        data[f"x{i}"] = values
    input_df = pd.DataFrame(data)

    return opt_problem, eval_func, input_df


@composite
def non_callable_values(draw):
    """Generate arbitrary values that are not callable and not None."""
    strategy = st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(min_size=0, max_size=20),
        st.floats(allow_nan=False, allow_infinity=False),
        st.lists(st.integers(), min_size=0, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=5),
                        st.integers(), min_size=0, max_size=3),
        st.booleans(),
        st.binary(min_size=0, max_size=10),
    )
    value = draw(strategy)
    # Ensure the value is not callable (lists, dicts, etc. are not callable)
    assume(not callable(value))
    return value


@composite
def non_evaluator_values(draw):
    """Generate arbitrary values that are not instances of Evaluator."""
    strategy = st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(min_size=0, max_size=20),
        st.floats(allow_nan=False, allow_infinity=False),
        st.lists(st.integers(), min_size=0, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=5),
                        st.integers(), min_size=0, max_size=3),
        st.booleans(),
        st.just(lambda x: x),  # A callable but not an Evaluator
        st.just(None),
    )
    value = draw(strategy)
    assume(not isinstance(value, Evaluator))
    return value


# ---------------------------------------------------------------------------
# Property 2: PyEvaluator functional equivalence
# ---------------------------------------------------------------------------


class TestPyEvaluatorFunctionalEquivalence:
    """Property 2: PyEvaluator functional equivalence.

    For any callable func and valid problem, constructing a PyEvaluator and
    calling it produces the same result as calling func directly on the sites
    DataFrame.

    **Validates: Requirements 3.4**
    """

    @given(data=simple_problem_and_func())
    @settings(max_examples=100)
    def test_py_evaluator_matches_direct_call(self, data):
        """PyEvaluator output matches direct function invocation.

        **Validates: Requirements 3.4**
        """
        opt_problem, eval_func, input_df = data

        # Create PyEvaluator
        evaluator = PyEvaluator(func=eval_func, opt_problem=opt_problem)

        # Call the evaluator on a copy of the input
        eval_sites = input_df.copy()
        evaluator(eval_sites)

        # Call the function directly on another copy
        direct_sites = input_df.copy()
        eval_func(direct_sites)

        # The responses should be identical
        response_names = [r.name for r in opt_problem.responses]
        for resp_name in response_names:
            assert resp_name in eval_sites.columns, (
                f"PyEvaluator did not produce response column '{resp_name}'"
            )
            assert resp_name in direct_sites.columns, (
                f"Direct call did not produce response column '{resp_name}'"
            )
            np.testing.assert_array_almost_equal(
                eval_sites[resp_name].values,
                direct_sites[resp_name].values,
                decimal=12,
                err_msg=f"PyEvaluator and direct call differ for response '{resp_name}'",
            )


# ---------------------------------------------------------------------------
# Property 3: Evaluator constructor input validation
# ---------------------------------------------------------------------------


class TestEvaluatorConstructorInputValidation:
    """Property 3: Evaluator constructor input validation.

    Passing non-callable to PyEvaluator raises TypeError.
    Passing non-Evaluator to ShiftScaleEvaluator raises TypeError.

    **Validates: Requirements 3.5, 6.5**
    """

    @given(value=non_callable_values())
    @settings(max_examples=100)
    def test_py_evaluator_rejects_non_callable(self, value):
        """PyEvaluator raises TypeError when func is not callable and not None.

        **Validates: Requirements 3.5**
        """
        # Create a minimal opt_problem for the constructor
        opt_problem = OptProblem(
            name="test",
            variables=[FloatVariable(name="x0", bounds=[0.0, 1.0])],
            responses=[FloatVariable(name="y0", bounds=[-10.0, 10.0])],
            objectives=["y0"],
        )

        with pytest.raises(TypeError):
            PyEvaluator(func=value, opt_problem=opt_problem)

    @given(value=non_evaluator_values())
    @settings(max_examples=100)
    def test_shift_scale_evaluator_rejects_non_evaluator(self, value):
        """ShiftScaleEvaluator raises TypeError when evaluate is not an Evaluator instance.

        **Validates: Requirements 6.5**
        """
        # Create a minimal opt_problem for the constructor
        opt_problem = OptProblem(
            name="test",
            variables=[FloatVariable(name="x0", bounds=[0.0, 1.0])],
            responses=[FloatVariable(name="y0", bounds=[-10.0, 10.0])],
            objectives=["y0"],
        )

        with pytest.raises(TypeError):
            ShiftScaleEvaluator(evaluate=value, opt_problem=opt_problem)
