"""Property test for SMT model serialization round-trip.

Feature: surrogate-model-pydantic-options, Property 9: Serialization Round-Trip

Validates that serializing a trained RadialBasisFunctionModel with _def_to_dict()
and deserializing with _def_from_dict() produces a model whose options match the
original and whose eval_np output is numerically identical.

**Validates: Requirements 15.1, 15.3**
"""

import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from standard_evaluator.surrogate_models.smt_models import (
    RadialBasisFunctionModel,
)
from standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt import (
    RadialBasisFunctionModelOptions,
)
import standard_evaluator as se


# --- Fixed test data (training is expensive, so we only vary options) ---


def _make_opt_problem():
    """Create a simple 2-variable, 1-response opt problem."""
    new_prob = se.utilities.create_opt_problem(num_independent=2, num_dependent=1)
    new_prob.variables[0].bounds = (-2.0, 2.0)
    new_prob.variables[1].bounds = (-2.0, 2.0)
    new_prob.responses[0].name = "f"
    new_prob.objectives = []
    new_prob.constraints = []
    return new_prob


def _make_sites() -> pd.DataFrame:
    """Create a fixed set of training sites (Rosenbrock-like function)."""
    sites = np.array(
        [
            [-2.0, -2.0],
            [-2.0, -0.66666667],
            [-2.0, 0.66666667],
            [-2.0, 2.0],
            [-0.66666667, -2.0],
            [-0.66666667, -0.66666667],
            [-0.66666667, 0.66666667],
            [-0.66666667, 2.0],
            [0.66666667, -2.0],
            [0.66666667, -0.66666667],
            [0.66666667, 0.66666667],
            [0.66666667, 2.0],
            [2.0, -2.0],
            [2.0, -0.66666667],
            [2.0, 0.66666667],
            [2.0, 2.0],
        ]
    )
    site_vals = np.array(
        [
            [45.0],
            [30.77777778],
            [20.11111111],
            [13.0],
            [8.75308642],
            [4.01234568],
            [2.82716049],
            [5.19753086],
            [6.08641975],
            [1.34567901],
            [0.16049383],
            [2.5308642],
            [37.0],
            [22.77777778],
            [12.11111111],
            [5.0],
        ]
    )
    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f"]
    )


# --- Hypothesis strategies ---


@composite
def rbf_options_strategy(draw):
    """Generate valid RadialBasisFunctionModelOptions with randomized fields."""
    return RadialBasisFunctionModelOptions(
        print_global=draw(st.booleans()),
        print_training=draw(st.booleans()),
        print_prediction=draw(st.booleans()),
        print_problem=draw(st.booleans()),
        print_solver=draw(st.booleans()),
        d0=draw(st.floats(min_value=0.1, max_value=10.0)),
        poly_degree=draw(st.sampled_from([-1, 0, 1])),
        reg=draw(
            st.floats(min_value=1e-15, max_value=1e-5)
        ),
        max_print_depth=draw(st.integers(min_value=0, max_value=10)),
    )


# --- Property-based test ---


class TestSmtSerializationRoundTrip:
    """Property 9: Serialization Round-Trip.

    For any SMT model constructed with valid Pydantic options and trained state,
    calling _def_from_dict on the serialized model should produce a model whose
    eval_np output is numerically identical to the original.

    Feature: surrogate-model-pydantic-options, Property 9: Serialization Round-Trip

    **Validates: Requirements 15.1, 15.3**
    """

    @settings(max_examples=100)
    @given(options=rbf_options_strategy())
    def test_serialization_round_trip(self, options):
        """Serialize and deserialize a trained RBF model, verify options and eval_np match.

        Feature: surrogate-model-pydantic-options, Property 9: Serialization Round-Trip

        **Validates: Requirements 15.1, 15.3**
        """
        opt_problem = _make_opt_problem()
        sites = _make_sites()

        # Build and train the original model
        original_model = RadialBasisFunctionModel(
            sites=sites,
            options=options,
            opt_problem=opt_problem,
        )

        # Serialize via the public to_dict (wraps _def_to_dict)
        model_dict = original_model.to_dict()

        # Deserialize via _def_from_dict
        reconstructed_model = RadialBasisFunctionModel._def_from_dict(model_dict)

        # Verify options match
        original_opts = original_model.full_options()
        reconstructed_opts = reconstructed_model.full_options()

        assert reconstructed_opts.d0 == original_opts.d0
        assert reconstructed_opts.poly_degree == original_opts.poly_degree
        assert reconstructed_opts.reg == original_opts.reg
        assert reconstructed_opts.max_print_depth == original_opts.max_print_depth
        assert reconstructed_opts.print_global == original_opts.print_global
        assert reconstructed_opts.print_training == original_opts.print_training
        assert reconstructed_opts.print_prediction == original_opts.print_prediction
        assert reconstructed_opts.print_problem == original_opts.print_problem
        assert reconstructed_opts.print_solver == original_opts.print_solver

        # Verify eval_np output matches within tolerance
        test_points = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [-1.0, 0.5],
                [0.5, -0.5],
            ]
        )

        original_output = original_model.eval_np(test_points)
        reconstructed_output = reconstructed_model.eval_np(test_points)

        np.testing.assert_allclose(
            reconstructed_output,
            original_output,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Reconstructed model eval_np does not match original",
        )
