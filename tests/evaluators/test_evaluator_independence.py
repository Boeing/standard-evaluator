"""Property-based tests for test evaluator independence and NumpyEvaluator equivalence.

Property 1: Test evaluator independence and correctness
Property 4: NumpyEvaluator numerical equivalence

Validates: Requirements 2.5, 4.5, 12.5, 13.5, 16.3
"""

import inspect
import sys

import numpy as np
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.strategies import composite

from standard_evaluator.evaluators.numpy_evaluator import NumpyEvaluator
from standard_evaluator.evaluators.test import __all__ as TEST_EVALUATOR_NAMES
import standard_evaluator.evaluators.test as test_pkg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_concrete_test_evaluator_classes():
    """Retrieve only concrete (non-abstract) test evaluator classes."""
    return [
        getattr(test_pkg, name) for name in TEST_EVALUATOR_NAMES
        if not inspect.isabstract(getattr(test_pkg, name))
    ]


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@composite
def evaluator_with_random_sites(draw):
    """Generate a test evaluator instance and a valid input DataFrame.

    Draws a random concrete test evaluator class, instantiates it, then generates
    input values within the evaluator's declared variable bounds.
    """
    # Pick a random concrete test evaluator class
    cls = draw(st.sampled_from(get_concrete_test_evaluator_classes()))

    # Instantiate the evaluator
    evaluator = cls()

    # Generate random values within bounds for each variable
    data = {}
    for var in evaluator.opt_problem.variables:
        lb, ub = var.bounds
        # Handle case where bounds are arrays (ArrayVariable)
        if hasattr(lb, '__len__'):
            lb = float(np.min(lb))
            ub = float(np.max(ub))
        else:
            lb = float(lb)
            ub = float(ub)

        # If bounds are equal (fixed variable), use that value
        if lb == ub:
            values = [lb if var.class_type != 'int' else int(lb)]
        elif var.class_type == 'int':
            # Generate integer values strictly within bounds
            int_lb = int(np.ceil(lb)) if np.isfinite(lb) else -100
            int_ub = int(np.floor(ub)) if np.isfinite(ub) else 100
            # Shrink range to avoid boundary singularities if possible
            if int_ub - int_lb >= 2:
                val = draw(st.integers(min_value=int_lb + 1,
                                       max_value=int_ub - 1))
            else:
                val = draw(st.integers(min_value=int_lb,
                                       max_value=int_ub))
            values = [val]
        else:
            # Handle infinite or very large bounds
            effective_lb = lb if np.isfinite(lb) else -100.0
            effective_ub = ub if np.isfinite(ub) else 100.0
            # Generate a float value strictly within bounds (avoid boundary)
            margin = (effective_ub - effective_lb) * 0.01
            inner_lb = effective_lb + margin
            inner_ub = effective_ub - margin
            if inner_lb >= inner_ub:
                # Bounds are too tight, just use midpoint
                values = [(effective_lb + effective_ub) / 2.0]
            else:
                val = draw(st.floats(min_value=inner_lb, max_value=inner_ub,
                                     allow_nan=False, allow_infinity=False))
                values = [val]
        data[var.name] = values

    input_df = pd.DataFrame(data)
    return evaluator, input_df


@composite
def numpy_evaluator_with_sites(draw):
    """Generate a concrete NumpyEvaluator subclass instance and valid input sites.

    Since none of the 38 test evaluators inherit from NumpyEvaluator directly,
    this strategy creates a simple concrete NumpyEvaluator subclass that
    computes a sum-of-squares function, then generates valid inputs.
    """
    n_vars = draw(st.integers(min_value=1, max_value=4))
    n_sites = draw(st.integers(min_value=1, max_value=3))

    from standard_evaluator import FloatVariable, OptProblem

    # Generate variables with valid non-degenerate bounds
    variables = []
    for i in range(n_vars):
        lower = draw(st.floats(min_value=-10.0, max_value=4.0,
                               allow_nan=False, allow_infinity=False))
        upper = draw(st.floats(min_value=lower + 1.0, max_value=10.0,
                               allow_nan=False, allow_infinity=False))
        variables.append(FloatVariable(name=f"x{i}", bounds=[lower, upper],
                                       default=(lower + upper) / 2.0))

    responses = [FloatVariable(name="f", bounds=[-1e6, 1e6])]

    opt_problem = OptProblem(
        name="numpy_test",
        variables=variables,
        responses=responses,
        objectives=["f"],
    )

    # Define the eval_np function for a sum-of-squares
    local_n_vars = n_vars

    class SumOfSquaresEvaluator(NumpyEvaluator):
        """A simple NumpyEvaluator that computes sum of squares."""

        def __init__(self):
            super().__init__(
                name="sum_of_squares",
                opt_problem=opt_problem,
            )

        def eval_np(self, sites: np.ndarray, names: list = None, **kwargs) -> np.ndarray:
            """Compute sum of squares across columns."""
            # sites shape: (n_sites, n_vars)
            result = np.sum(sites ** 2, axis=1, keepdims=True)
            return result

    evaluator = SumOfSquaresEvaluator()

    # Generate input DataFrame with values within bounds
    data = {}
    for i, var in enumerate(variables):
        lb, ub = var.bounds
        vals = [draw(st.floats(min_value=lb, max_value=ub,
                               allow_nan=False, allow_infinity=False))
                for _ in range(n_sites)]
        data[f"x{i}"] = vals
    input_df = pd.DataFrame(data)

    return evaluator, input_df, local_n_vars


# ---------------------------------------------------------------------------
# Property 1: Test evaluator independence and correctness
# ---------------------------------------------------------------------------


class TestEvaluatorIndependence:
    """Property 1: Test evaluator independence and correctness.

    Each test evaluator can be instantiated independently, called with its
    default_site(), and produces numeric (non-NaN) outputs. No module from
    boeing_standard_evaluator should be present in sys.modules after execution.

    **Validates: Requirements 2.5, 12.5, 13.5, 16.3**
    """

    @given(data=st.sampled_from(get_concrete_test_evaluator_classes()))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_evaluator_instantiates_and_produces_numeric_output(self, data):
        """Each test evaluator produces numeric (non-NaN) outputs from default_site().

        **Validates: Requirements 2.5, 12.5, 13.5, 16.3**
        """
        cls = data

        # Instantiate the evaluator
        evaluator = cls()

        # Get default site
        site = evaluator.default_site()
        assert isinstance(site, pd.DataFrame), (
            f"{cls.__name__}.default_site() did not return a DataFrame"
        )
        assert len(site) > 0, (
            f"{cls.__name__}.default_site() returned empty DataFrame"
        )

        # Call the evaluator
        evaluator(site)

        # Verify outputs are present and numeric (non-NaN)
        for output_name in evaluator.outputs:
            assert output_name in site.columns, (
                f"{cls.__name__} did not produce output column '{output_name}'"
            )
            values = site[output_name].values
            assert np.all(np.isfinite(values)), (
                f"{cls.__name__} produced non-finite values for '{output_name}': {values}"
            )

        # Verify no boeing_standard_evaluator in sys.modules
        boeing_modules = [
            m for m in sys.modules
            if 'boeing_standard_evaluator' in m
        ]
        assert len(boeing_modules) == 0, (
            f"boeing_standard_evaluator found in sys.modules: {boeing_modules}"
        )

    @given(evaluator_and_sites=evaluator_with_random_sites())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_evaluator_with_random_inputs(self, evaluator_and_sites):
        """Each test evaluator produces numeric (non-NaN) outputs for random valid inputs.

        Values at boundaries may produce NaN due to mathematical singularities
        (e.g., division by zero). We exclude boundary values from the generated
        inputs to avoid these legitimate singularities.

        **Validates: Requirements 2.5, 12.5, 13.5, 16.3**
        """
        evaluator, input_df = evaluator_and_sites

        # Call the evaluator
        evaluator(input_df)

        # Verify outputs are present and numeric (non-NaN)
        for output_name in evaluator.outputs:
            assert output_name in input_df.columns, (
                f"{type(evaluator).__name__} did not produce output column '{output_name}'"
            )
            values = input_df[output_name].values
            assert all(not np.isnan(v) for v in values), (
                f"{type(evaluator).__name__} produced NaN for '{output_name}'"
            )


# ---------------------------------------------------------------------------
# Property 4: NumpyEvaluator numerical equivalence
# ---------------------------------------------------------------------------


class TestNumpyEvaluatorEquivalence:
    """Property 4: NumpyEvaluator numerical equivalence.

    For test evaluators that inherit from NumpyEvaluator, calling eval_np and
    __call__ produce equivalent results. Since no test evaluators in the current
    codebase inherit from NumpyEvaluator, this test creates concrete subclasses
    and verifies that the NumpyEvaluator._evaluate pathway (which calls eval_np
    internally) produces results equivalent to calling eval_np directly.

    **Validates: Requirements 4.5**
    """

    @given(data=numpy_evaluator_with_sites())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_eval_np_matches_call(self, data):
        """eval_np and __call__ produce numerically equivalent results.

        **Validates: Requirements 4.5**
        """
        evaluator, input_df, n_vars = data

        # Get the numpy input array
        np_input = input_df.values.astype(np.float64)

        # Call eval_np directly
        eval_np_result = evaluator.eval_np(np_input)

        # Call the evaluator via __call__ (which goes through _evaluate -> eval_np)
        call_sites = input_df.copy()
        evaluator(call_sites)

        # Extract the output from the DataFrame
        call_result = call_sites["f"].values

        # Compare results - they should be numerically identical within tolerance
        np.testing.assert_allclose(
            call_result,
            eval_np_result.flatten(),
            rtol=1e-12,
            atol=1e-12,
            err_msg="eval_np and __call__ produced different results",
        )

    @given(data=numpy_evaluator_with_sites())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_numpy_evaluator_produces_finite_output(self, data):
        """NumpyEvaluator subclasses produce finite numeric outputs.

        **Validates: Requirements 4.5**
        """
        evaluator, input_df, n_vars = data

        # Call via __call__
        evaluator(input_df)

        # Verify output is finite
        values = input_df["f"].values
        assert np.all(np.isfinite(values)), (
            f"NumpyEvaluator produced non-finite values: {values}"
        )
