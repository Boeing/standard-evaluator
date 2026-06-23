"""Property-based test for PolynomialModel serialization round-trip.

Feature: surrogate-model-migration
Property 4: PolynomialModel serialization round-trip

**Validates: Requirements 4.7, 17.3, 17.4**
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from standard_evaluator.problem import OptProblem, FloatVariable
from standard_evaluator.surrogate_models.polynomial_model import (
    CoefficientOrdering,
    PolynomialModel,
    PolynomialModelOptions,
)
from math import comb


# --- Strategies ---

# Number of independent variables (1-4)
nind_strategy = st.integers(min_value=1, max_value=4)

# Polynomial degree (1-3)
degree_strategy = st.integers(min_value=1, max_value=3)

# Coefficient ordering choices
ordering_strategy = st.sampled_from(list(CoefficientOrdering))


@st.composite
def polynomial_model_strategy(draw):
    """Generate a valid trained PolynomialModel with random data.

    Ensures the number of training sites exceeds the number of polynomial
    terms so that the least-squares fit is well-determined.
    """
    nind = draw(nind_strategy)
    degree = draw(degree_strategy)
    ordering = draw(ordering_strategy)

    # Number of responses (1-3)
    nresp = draw(st.integers(min_value=1, max_value=3))

    # Calculate minimum number of sites needed (must exceed number of terms)
    nterms = comb(nind + degree, degree)
    # Need at least nterms + 1 sites to avoid underdetermined system
    nsites = nterms + draw(st.integers(min_value=1, max_value=5))

    # Generate variable names and bounds
    var_names = [f"x{i}" for i in range(nind)]
    resp_names = [f"y{j}" for j in range(nresp)]

    # Generate random bounds (ensure lb < ub)
    variables = []
    for name in var_names:
        lb = draw(st.floats(min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False))
        ub = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
        variables.append(FloatVariable(name=name, bounds=(lb, ub)))

    responses = [FloatVariable(name=name) for name in resp_names]

    opt_problem = OptProblem(variables=variables, responses=responses)

    # Generate random site data within bounds
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31)))
    x_data = np.empty((nsites, nind))
    for i, var in enumerate(variables):
        lb, ub = var.bounds
        x_data[:, i] = rng.uniform(lb, ub, size=nsites)

    # Generate response data (use a simple polynomial relationship + noise)
    y_data = np.zeros((nsites, nresp))
    for j in range(nresp):
        # Linear combination of inputs plus small random component
        coeffs = rng.uniform(-2.0, 2.0, size=nind)
        y_data[:, j] = x_data @ coeffs + rng.uniform(-0.1, 0.1, size=nsites)

    # Build DataFrame
    data = np.column_stack([x_data, y_data])
    columns = var_names + resp_names
    sites = pd.DataFrame(data, columns=columns)

    # Build options
    options = PolynomialModelOptions(degree=degree, coefficient_ordering=ordering)

    # Create model
    model = PolynomialModel(sites=sites, opt_problem=opt_problem, options=options)

    return model


@given(model=polynomial_model_strategy())
@settings(max_examples=100, deadline=None)
def test_polynomial_model_serialization_round_trip(model: PolynomialModel):
    """Property 4: For any valid trained PolynomialModel, calling to_dict()
    followed by from_dict() produces a reconstructed model whose predictions
    at any input point are numerically identical (within tolerance 1e-12) to
    the original model's predictions.

    Feature: surrogate-model-migration
    Property 4: PolynomialModel serialization round-trip

    **Validates: Requirements 4.7, 17.3, 17.4**
    """
    # Serialize
    model_dict = model.to_dict()
    print(model_dict)

    # Verify dict has required keys
    assert "type" in model_dict
    assert "info" in model_dict
    assert "opt_problem" in model_dict
    assert "version" in model_dict
    assert "name" in model_dict
    assert model_dict["type"] == "PolynomialModel"

    # Deserialize
    reconstructed = PolynomialModel.from_dict(model_dict)

    # Generate test points within model bounds
    nind = model.nind
    rng = np.random.default_rng(42)
    n_test = 5
    test_points = np.empty((n_test, nind))
    for i in range(nind):
        lb = model.xlb[i]
        ub = model.xub[i]
        test_points[:, i] = rng.uniform(lb, ub, size=n_test)

    # Compare predictions
    pred_original = model.eval_np(test_points)
    pred_reconstructed = reconstructed.eval_np(test_points)

    # Verify numerical identity within tolerance 1e-12
    np.testing.assert_allclose(
        pred_reconstructed,
        pred_original,
        atol=1e-12,
        rtol=0,
        err_msg=(
            "Serialization round-trip produced different predictions. "
            f"Max absolute difference: {np.max(np.abs(pred_original - pred_reconstructed))}"
        ),
    )
