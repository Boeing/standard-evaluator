"""Property-based test for SMT model training produces valid predictions.

Feature: surrogate-model-migration
Property 7: SMT model training produces valid predictions

**Validates: Requirements 6.5, 7.5, 9.5, 10.5, 11.5**
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from standard_evaluator.problem import OptProblem, FloatVariable
from standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt import (
    RadialBasisFunctionModel,
    RadialBasisFunctionModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.idw_model_using_smt import (
    InverseDistanceWeightingModel,
    InverseDistanceWeightingModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.ls_model_using_smt import (
    LeastSquaresApproximationModel,
    LeastSquaresApproximationModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt import (
    RegularizedMinimalEnergyTensorProductBSplines,
    RegularizedMinimalEnergyTensorProductBSplinesOptions,
)
from standard_evaluator.surrogate_models.smt_models.sopa_model_using_smt import (
    SecondOrderPolynomialApproximationModel,
    SecondOrderPolynomialApproximationModelOptions,
)


# --- Strategies ---


@st.composite
def smt_training_data(draw, min_sites=10, max_sites=30):
    """Generate valid training data for SMT models.

    Strategy:
    - Generate 1-3 independent variables with finite bounds
    - Generate 1-2 responses
    - Generate between min_sites and max_sites training points
    - Use well-spaced data within bounds to ensure numerical stability
    - Ensure response values are non-trivial (not all constant) to avoid
      degenerate training paths in some SMT models
    """
    nind = draw(st.integers(min_value=1, max_value=3))
    num_responses = draw(st.integers(min_value=1, max_value=2))
    nsites = draw(st.integers(min_value=min_sites, max_value=max_sites))

    # Generate variables with well-separated bounds
    variables = []
    for i in range(nind):
        lb = draw(st.floats(min_value=-10.0, max_value=0.0,
                            allow_nan=False, allow_infinity=False))
        ub = draw(st.floats(min_value=1.0, max_value=10.0,
                            allow_nan=False, allow_infinity=False))
        assume(ub - lb >= 0.5)
        variables.append(FloatVariable(name=f"x{i}", bounds=(lb, ub)))

    responses = [FloatVariable(name=f"y{j}") for j in range(num_responses)]
    opt_problem = OptProblem(variables=variables, responses=responses)

    # Generate input sites uniformly within bounds
    site_inputs = np.empty((nsites, nind))
    for i in range(nind):
        lb, ub = variables[i].bounds
        # Use linspace with random permutation for good space coverage
        points = np.linspace(lb + 0.01 * (ub - lb), ub - 0.01 * (ub - lb), nsites)
        perm = draw(st.permutations(list(range(nsites))))
        site_inputs[:, i] = points[list(perm)]

    # Generate response values as a non-trivial function of inputs.
    # Use at least one non-zero coefficient to ensure responses vary.
    site_outputs = np.empty((nsites, num_responses))
    for j in range(num_responses):
        # First coefficient is forced non-zero; others can be anything
        first_coeff = draw(
            st.floats(min_value=0.5, max_value=2.0,
                      allow_nan=False, allow_infinity=False)
        )
        other_coeffs = draw(
            st.lists(
                st.floats(min_value=-2.0, max_value=2.0,
                          allow_nan=False, allow_infinity=False),
                min_size=max(0, nind - 1),
                max_size=max(0, nind - 1),
            )
        )
        coeffs = [first_coeff] + other_coeffs
        site_outputs[:, j] = sum(
            c * site_inputs[:, i] for i, c in enumerate(coeffs)
        )
        # Add a small offset
        offset = draw(st.floats(min_value=-5.0, max_value=5.0,
                                allow_nan=False, allow_infinity=False))
        site_outputs[:, j] += offset

    # Ensure responses are non-constant (some SMT models fail on constant data)
    for j in range(num_responses):
        assume(np.std(site_outputs[:, j]) > 1e-10)

    # Build the DataFrame
    input_cols = [f"x{i}" for i in range(nind)]
    output_cols = [f"y{j}" for j in range(num_responses)]
    columns = input_cols + output_cols
    data = np.hstack([site_inputs, site_outputs])
    sites_df = pd.DataFrame(data, columns=columns)

    return sites_df, opt_problem


@st.composite
def sopa_training_data(draw):
    """Generate valid training data for the SOPA (QP) model.

    SOPA requires at minimum (nind+1)*(nind+2)/2 sites.
    We keep nind small (1-2) to keep the minimum manageable.
    Input points must not be collinear to avoid singular design matrices.
    We verify the QP design matrix is non-singular before proceeding.
    """
    nind = draw(st.integers(min_value=1, max_value=2))
    num_responses = draw(st.integers(min_value=1, max_value=2))

    # SOPA minimum sites: (nind+1)*(nind+2)/2
    min_sites_required = int((nind + 1) * (nind + 2) / 2)
    nsites = draw(st.integers(
        min_value=min_sites_required + 2,
        max_value=min_sites_required + 10,
    ))

    # Generate variables with well-separated bounds
    variables = []
    for i in range(nind):
        lb = draw(st.floats(min_value=-10.0, max_value=-0.5,
                            allow_nan=False, allow_infinity=False))
        ub = draw(st.floats(min_value=1.0, max_value=10.0,
                            allow_nan=False, allow_infinity=False))
        assume(ub - lb >= 1.0)
        variables.append(FloatVariable(name=f"x{i}", bounds=(lb, ub)))

    responses = [FloatVariable(name=f"y{j}") for j in range(num_responses)]
    opt_problem = OptProblem(variables=variables, responses=responses)

    # Generate input sites with independent permutations per dimension
    site_inputs = np.empty((nsites, nind))
    for i in range(nind):
        lb, ub = variables[i].bounds
        points = np.linspace(lb + 0.05 * (ub - lb), ub - 0.05 * (ub - lb), nsites)
        perm = draw(st.permutations(list(range(nsites))))
        site_inputs[:, i] = points[list(perm)]

    # For QP, build the second-order design matrix and check its rank.
    # QP uses: 1, x0, x1, ..., x0^2, x0*x1, x1^2, ...
    # We check the condition number to ensure the matrix is non-singular.
    if nind == 2:
        x0 = site_inputs[:, 0:1]
        x1 = site_inputs[:, 1:2]
        design = np.hstack([
            np.ones((nsites, 1)), x0, x1, x0**2, x0 * x1, x1**2
        ])
        cond = np.linalg.cond(design)
        assume(cond < 1e8)

    # Generate non-trivial response values
    site_outputs = np.empty((nsites, num_responses))
    for j in range(num_responses):
        first_coeff = draw(
            st.floats(min_value=0.5, max_value=2.0,
                      allow_nan=False, allow_infinity=False)
        )
        other_coeffs = draw(
            st.lists(
                st.floats(min_value=-2.0, max_value=2.0,
                          allow_nan=False, allow_infinity=False),
                min_size=max(0, nind - 1),
                max_size=max(0, nind - 1),
            )
        )
        coeffs = [first_coeff] + other_coeffs
        site_outputs[:, j] = sum(
            c * site_inputs[:, i] for i, c in enumerate(coeffs)
        )
        offset = draw(st.floats(min_value=-5.0, max_value=5.0,
                                allow_nan=False, allow_infinity=False))
        site_outputs[:, j] += offset

    # Ensure responses are non-constant
    for j in range(num_responses):
        assume(np.std(site_outputs[:, j]) > 1e-10)

    # Build the DataFrame
    input_cols = [f"x{i}" for i in range(nind)]
    output_cols = [f"y{j}" for j in range(num_responses)]
    columns = input_cols + output_cols
    data = np.hstack([site_inputs, site_outputs])
    sites_df = pd.DataFrame(data, columns=columns)

    return sites_df, opt_problem


# --- Property Tests ---


@given(data=smt_training_data())
@settings(max_examples=100, deadline=None)
def test_rbf_training_produces_valid_predictions(data):
    """Property 7: RadialBasisFunctionModel training produces valid predictions.

    Test that instantiating with valid training data completes without error
    and produces finite-valued predictions (no NaN or Inf).

    Feature: surrogate-model-migration
    Property 7: SMT model training produces valid predictions

    **Validates: Requirements 6.5**
    """
    sites_df, opt_problem = data
    nind = len(opt_problem.variables)

    options = RadialBasisFunctionModelOptions(print_global=False)
    model = RadialBasisFunctionModel(
        sites=sites_df, options=options, opt_problem=opt_problem
    )

    # Evaluate at training sites
    site_inputs = sites_df[[f"x{i}" for i in range(nind)]].values
    predictions = model.eval_np(site_inputs)

    # Check predictions are finite (no NaN or Inf)
    assert np.all(np.isfinite(predictions)), (
        f"RBF model produced non-finite predictions. "
        f"NaN count: {np.sum(np.isnan(predictions))}, "
        f"Inf count: {np.sum(np.isinf(predictions))}"
    )


@given(data=smt_training_data())
@settings(max_examples=100, deadline=None)
def test_idw_training_produces_valid_predictions(data):
    """Property 7: InverseDistanceWeightingModel training produces valid predictions.

    Test that instantiating with valid training data completes without error
    and produces finite-valued predictions (no NaN or Inf).

    Feature: surrogate-model-migration
    Property 7: SMT model training produces valid predictions

    **Validates: Requirements 7.5**
    """
    sites_df, opt_problem = data
    nind = len(opt_problem.variables)

    options = InverseDistanceWeightingModelOptions(print_global=False)
    model = InverseDistanceWeightingModel(
        sites=sites_df, options=options, opt_problem=opt_problem
    )

    # Evaluate at training sites
    site_inputs = sites_df[[f"x{i}" for i in range(nind)]].values
    predictions = model.eval_np(site_inputs)

    # Check predictions are finite (no NaN or Inf)
    assert np.all(np.isfinite(predictions)), (
        f"IDW model produced non-finite predictions. "
        f"NaN count: {np.sum(np.isnan(predictions))}, "
        f"Inf count: {np.sum(np.isinf(predictions))}"
    )


@given(data=smt_training_data())
@settings(max_examples=100, deadline=None)
def test_ls_training_produces_valid_predictions(data):
    """Property 7: LeastSquaresApproximationModel training produces valid predictions.

    Test that instantiating with valid training data completes without error
    and produces finite-valued predictions (no NaN or Inf).

    Feature: surrogate-model-migration
    Property 7: SMT model training produces valid predictions

    **Validates: Requirements 9.5**
    """
    sites_df, opt_problem = data
    nind = len(opt_problem.variables)

    options = LeastSquaresApproximationModelOptions(print_global=False)
    model = LeastSquaresApproximationModel(
        sites=sites_df, options=options, opt_problem=opt_problem
    )

    # Evaluate at training sites
    site_inputs = sites_df[[f"x{i}" for i in range(nind)]].values
    predictions = model.eval_np(site_inputs)

    # Check predictions are finite (no NaN or Inf)
    assert np.all(np.isfinite(predictions)), (
        f"LS model produced non-finite predictions. "
        f"NaN count: {np.sum(np.isnan(predictions))}, "
        f"Inf count: {np.sum(np.isinf(predictions))}"
    )


@given(data=smt_training_data())
@settings(max_examples=100, deadline=None)
def test_rmts_training_produces_valid_predictions(data):
    """Property 7: RMTS model training produces valid predictions.

    Test that instantiating with valid training data completes without error
    and produces finite-valued predictions (no NaN or Inf).
    RMTS requires use_xlimits=True (already the default for its options class).

    Feature: surrogate-model-migration
    Property 7: SMT model training produces valid predictions

    **Validates: Requirements 10.5**
    """
    sites_df, opt_problem = data
    nind = len(opt_problem.variables)

    options = RegularizedMinimalEnergyTensorProductBSplinesOptions(
        print_global=False
    )
    model = RegularizedMinimalEnergyTensorProductBSplines(
        sites=sites_df, options=options, opt_problem=opt_problem
    )

    # Evaluate at training sites
    site_inputs = sites_df[[f"x{i}" for i in range(nind)]].values
    predictions = model.eval_np(site_inputs)

    # Check predictions are finite (no NaN or Inf)
    assert np.all(np.isfinite(predictions)), (
        f"RMTS model produced non-finite predictions. "
        f"NaN count: {np.sum(np.isnan(predictions))}, "
        f"Inf count: {np.sum(np.isinf(predictions))}"
    )


@given(data=sopa_training_data())
@settings(max_examples=100, deadline=None)
def test_sopa_training_produces_valid_predictions(data):
    """Property 7: SOPA model training produces valid predictions.

    Test that instantiating with valid training data completes without error
    and produces finite-valued predictions (no NaN or Inf).
    SOPA (QP) requires at minimum (nind+1)*(nind+2)/2 sites.

    Feature: surrogate-model-migration
    Property 7: SMT model training produces valid predictions

    **Validates: Requirements 11.5**
    """
    sites_df, opt_problem = data
    nind = len(opt_problem.variables)

    options = SecondOrderPolynomialApproximationModelOptions(print_global=False)
    model = SecondOrderPolynomialApproximationModel(
        sites=sites_df, options=options, opt_problem=opt_problem
    )

    # Evaluate at training sites
    site_inputs = sites_df[[f"x{i}" for i in range(nind)]].values
    predictions = model.eval_np(site_inputs)

    # Check predictions are finite (no NaN or Inf)
    assert np.all(np.isfinite(predictions)), (
        f"SOPA model produced non-finite predictions. "
        f"NaN count: {np.sum(np.isnan(predictions))}, "
        f"Inf count: {np.sum(np.isinf(predictions))}"
    )
