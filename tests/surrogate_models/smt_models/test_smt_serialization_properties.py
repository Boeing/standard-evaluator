"""Property-based test for SMT model serialization round-trip.

Feature: surrogate-model-migration
Property 6: SMT model serialization round-trip

**Validates: Requirements 5.8, 17.5**
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.problem import OptProblem, FloatVariable
from standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt import (
    RadialBasisFunctionModel,
)
from standard_evaluator.surrogate_models.smt_models.idw_model_using_smt import (
    InverseDistanceWeightingModel,
)
from standard_evaluator.surrogate_models.smt_models.ls_model_using_smt import (
    LeastSquaresApproximationModel,
)
from standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt import (
    RegularizedMinimalEnergyTensorProductBSplines,
)
from standard_evaluator.surrogate_models.smt_models.sopa_model_using_smt import (
    SecondOrderPolynomialApproximationModel,
)


# --- Helpers ---


def _build_opt_problem(nind: int, nout: int):
    """Build an OptProblem with nind input variables and nout responses.

    Variables have bounds [0, 1] for simplicity.
    """
    variables = [
        FloatVariable(name=f"x{i}", bounds=(0.0, 1.0)) for i in range(nind)
    ]
    responses = [FloatVariable(name=f"y{j}") for j in range(nout)]
    return OptProblem(variables=variables, responses=responses)


def _build_sites_df(nind: int, nout: int, nsites: int, rng: np.random.Generator):
    """Build a training DataFrame with random data in [0, 1] for inputs and
    outputs computed as simple polynomials of the inputs.
    """
    x = rng.uniform(0.0, 1.0, size=(nsites, nind))
    # Generate outputs as a simple linear combination + noise to ensure
    # the model has something meaningful to fit
    y = np.zeros((nsites, nout))
    for j in range(nout):
        coeffs = rng.uniform(-2.0, 2.0, size=nind)
        y[:, j] = x @ coeffs + rng.uniform(-0.1, 0.1, size=nsites)

    columns = [f"x{i}" for i in range(nind)] + [f"y{j}" for j in range(nout)]
    return pd.DataFrame(np.hstack([x, y]), columns=columns)


def _serialization_round_trip(model_cls, sites_df, opt_problem, test_points, **kwargs):
    """Instantiate a model, serialize via to_dict(), reconstruct via from_dict(),
    and compare predictions on test_points.

    Returns (original_preds, reconstructed_preds).
    """
    model = model_cls(sites=sites_df, opt_problem=opt_problem, **kwargs)
    original_preds = model.eval_np(test_points)

    model_dict = model.to_dict()
    reconstructed = model_cls.from_dict(model_dict)
    reconstructed_preds = reconstructed.eval_np(test_points)

    return original_preds, reconstructed_preds


# --- Hypothesis strategies ---


@st.composite
def smt_training_data(draw, min_sites=20, extra_sites_for_sopa=False):
    """Generate valid training data for SMT models.

    Generates:
    - nind between 2 and 3
    - nout = 1 (all SMT models are per-response, keep simple)
    - nsites sufficient for the model type
    - A random seed for reproducibility within each test case
    """
    nind = draw(st.integers(min_value=2, max_value=3))
    nout = 1

    if extra_sites_for_sopa:
        # SOPA (QP) requires at minimum (nind+1)*(nind+2)/2 sites
        min_required = int((nind + 1) * (nind + 2) / 2)
        nsites = draw(st.integers(min_value=max(min_sites, min_required + 5),
                                  max_value=max(min_sites, min_required + 5) + 10))
    else:
        nsites = draw(st.integers(min_value=min_sites, max_value=min_sites + 10))

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    return nind, nout, nsites, seed


# --- Property tests ---


@given(data=smt_training_data())
@settings(max_examples=10, deadline=None)
def test_rbf_serialization_round_trip(data):
    """Property 6: RBF model serialization round-trip.

    Test that to_dict() followed by from_dict() produces predictions
    within tolerance 1e-10 for the RBF model.

    Feature: surrogate-model-migration
    Property 6: SMT model serialization round-trip

    **Validates: Requirements 5.8, 17.5**
    """
    nind, nout, nsites, seed = data
    rng = np.random.default_rng(seed)

    opt_problem = _build_opt_problem(nind, nout)
    sites_df = _build_sites_df(nind, nout, nsites, rng)
    test_points = rng.uniform(0.0, 1.0, size=(5, nind))

    original_preds, reconstructed_preds = _serialization_round_trip(
        RadialBasisFunctionModel, sites_df, opt_problem, test_points
    )

    np.testing.assert_allclose(
        original_preds, reconstructed_preds, atol=1e-10,
        err_msg="RBF model serialization round-trip failed: predictions differ"
    )


@given(data=smt_training_data())
@settings(max_examples=10, deadline=None)
def test_idw_serialization_round_trip(data):
    """Property 6: IDW model serialization round-trip.

    Test that to_dict() followed by from_dict() produces predictions
    within tolerance 1e-10 for the IDW model.

    Feature: surrogate-model-migration
    Property 6: SMT model serialization round-trip

    **Validates: Requirements 5.8, 17.5**
    """
    nind, nout, nsites, seed = data
    rng = np.random.default_rng(seed)

    opt_problem = _build_opt_problem(nind, nout)
    sites_df = _build_sites_df(nind, nout, nsites, rng)
    test_points = rng.uniform(0.0, 1.0, size=(5, nind))

    original_preds, reconstructed_preds = _serialization_round_trip(
        InverseDistanceWeightingModel, sites_df, opt_problem, test_points
    )

    np.testing.assert_allclose(
        original_preds, reconstructed_preds, atol=1e-10,
        err_msg="IDW model serialization round-trip failed: predictions differ"
    )


@given(data=smt_training_data())
@settings(max_examples=10, deadline=None)
def test_ls_serialization_round_trip(data):
    """Property 6: LS model serialization round-trip.

    Test that to_dict() followed by from_dict() produces predictions
    within tolerance 1e-10 for the LS model.

    Feature: surrogate-model-migration
    Property 6: SMT model serialization round-trip

    **Validates: Requirements 5.8, 17.5**
    """
    nind, nout, nsites, seed = data
    rng = np.random.default_rng(seed)

    opt_problem = _build_opt_problem(nind, nout)
    sites_df = _build_sites_df(nind, nout, nsites, rng)
    test_points = rng.uniform(0.0, 1.0, size=(5, nind))

    original_preds, reconstructed_preds = _serialization_round_trip(
        LeastSquaresApproximationModel, sites_df, opt_problem, test_points
    )

    np.testing.assert_allclose(
        original_preds, reconstructed_preds, atol=1e-10,
        err_msg="LS model serialization round-trip failed: predictions differ"
    )


@given(data=smt_training_data())
@settings(max_examples=10, deadline=None)
def test_rmts_serialization_round_trip(data):
    """Property 6: RMTS model serialization round-trip.

    Test that to_dict() followed by from_dict() produces predictions
    within tolerance 1e-10 for the RMTS model.
    Note: RMTS requires use_xlimits=True (the default for this model).

    Feature: surrogate-model-migration
    Property 6: SMT model serialization round-trip

    **Validates: Requirements 5.8, 17.5**
    """
    nind, nout, nsites, seed = data
    rng = np.random.default_rng(seed)

    opt_problem = _build_opt_problem(nind, nout)
    sites_df = _build_sites_df(nind, nout, nsites, rng)
    test_points = rng.uniform(0.0, 1.0, size=(5, nind))

    # RMTS uses use_xlimits=True by default in its options
    original_preds, reconstructed_preds = _serialization_round_trip(
        RegularizedMinimalEnergyTensorProductBSplines,
        sites_df, opt_problem, test_points
    )

    np.testing.assert_allclose(
        original_preds, reconstructed_preds, atol=1e-10,
        err_msg="RMTS model serialization round-trip failed: predictions differ"
    )


@given(data=smt_training_data(extra_sites_for_sopa=True))
@settings(max_examples=10, deadline=None)
def test_sopa_serialization_round_trip(data):
    """Property 6: SOPA model serialization round-trip.

    Test that to_dict() followed by from_dict() produces predictions
    within tolerance 1e-10 for the SOPA (QP) model.
    Note: SOPA requires at minimum (nind+1)*(nind+2)/2 sites.

    Feature: surrogate-model-migration
    Property 6: SMT model serialization round-trip

    **Validates: Requirements 5.8, 17.5**
    """
    nind, nout, nsites, seed = data
    rng = np.random.default_rng(seed)

    opt_problem = _build_opt_problem(nind, nout)
    sites_df = _build_sites_df(nind, nout, nsites, rng)
    test_points = rng.uniform(0.0, 1.0, size=(5, nind))

    original_preds, reconstructed_preds = _serialization_round_trip(
        SecondOrderPolynomialApproximationModel,
        sites_df, opt_problem, test_points
    )

    np.testing.assert_allclose(
        original_preds, reconstructed_preds, atol=1e-10,
        err_msg="SOPA model serialization round-trip failed: predictions differ"
    )
