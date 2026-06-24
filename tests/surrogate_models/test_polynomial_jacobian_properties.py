"""Property-based tests for Polynomial Jacobian consistency with finite differences.

Property 3: Polynomial Jacobian consistency with finite differences

Validates: Requirements 4.6
"""

import numpy as np
import pandas as pd
import numdifftools as nd
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.strategies import composite

from standard_evaluator.problem import OptProblem, FloatVariable
from standard_evaluator.surrogate_models.polynomial_model import (
    PolynomialModel,
    PolynomialModelOptions,
)


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@composite
def polynomial_model_and_point(draw):
    """Generate a trained PolynomialModel and a valid input point within bounds.

    Returns a tuple of (model, x_point) where:
    - model is a fitted PolynomialModel with nind variables and ndep responses
    - x_point is a (1 x nind) numpy array within the model's variable bounds
    """
    nind = draw(st.integers(min_value=1, max_value=4))
    ndep = draw(st.integers(min_value=1, max_value=3))
    degree = draw(st.integers(min_value=1, max_value=3))

    # Number of monomials for a polynomial of given degree and nind variables
    # is C(nind + degree, degree). We need at least that many sites for fitting.
    from math import comb
    n_terms = comb(nind + degree, degree)
    # Need enough sites for least-squares to work (at least n_terms)
    n_sites = n_terms + draw(st.integers(min_value=0, max_value=5))

    # Generate variable bounds (non-degenerate)
    variables = []
    bounds_list = []
    for i in range(nind):
        lower = draw(st.floats(min_value=-5.0, max_value=4.0,
                               allow_nan=False, allow_infinity=False))
        upper = draw(st.floats(min_value=lower + 0.5, max_value=5.0,
                               allow_nan=False, allow_infinity=False))
        variables.append(FloatVariable(name=f"x{i}", bounds=[lower, upper]))
        bounds_list.append((lower, upper))

    responses = []
    for j in range(ndep):
        responses.append(FloatVariable(name=f"f{j}", bounds=[-1e6, 1e6]))

    opt_problem = OptProblem(
        name="test_poly",
        variables=variables,
        responses=responses,
        objectives=["f0"],
    )

    # Generate training site inputs within bounds
    site_inputs = np.zeros((n_sites, nind))
    for i in range(nind):
        lb, ub = bounds_list[i]
        values = draw(
            st.lists(
                st.floats(min_value=lb, max_value=ub,
                          allow_nan=False, allow_infinity=False),
                min_size=n_sites, max_size=n_sites
            )
        )
        site_inputs[:, i] = values

    # Generate response values (use a random polynomial-like function)
    site_outputs = np.zeros((n_sites, ndep))
    for j in range(ndep):
        # Simple polynomial: weighted sum of input products
        coeffs = draw(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0,
                          allow_nan=False, allow_infinity=False),
                min_size=nind, max_size=nind
            )
        )
        offset = draw(st.floats(min_value=-5.0, max_value=5.0,
                                allow_nan=False, allow_infinity=False))
        for k in range(nind):
            site_outputs[:, j] += coeffs[k] * site_inputs[:, k]
        site_outputs[:, j] += offset

    # Ensure finite values
    assume(np.all(np.isfinite(site_inputs)))
    assume(np.all(np.isfinite(site_outputs)))

    # Build DataFrame
    var_names = [f"x{i}" for i in range(nind)]
    resp_names = [f"f{j}" for j in range(ndep)]
    data = {}
    for i, name in enumerate(var_names):
        data[name] = site_inputs[:, i]
    for j, name in enumerate(resp_names):
        data[name] = site_outputs[:, j]
    sites_df = pd.DataFrame(data)

    # Create the model
    options = PolynomialModelOptions(degree=degree)
    model = PolynomialModel(
        sites=sites_df,
        options=options,
        opt_problem=opt_problem,
    )

    # Generate a test point strictly within bounds (avoid boundary issues for FD)
    x_point = np.zeros((1, nind))
    for i in range(nind):
        lb, ub = bounds_list[i]
        # Use interior point to avoid finite-difference stepping outside bounds
        margin = (ub - lb) * 0.1
        inner_lb = lb + margin
        inner_ub = ub - margin
        val = draw(st.floats(min_value=inner_lb, max_value=inner_ub,
                             allow_nan=False, allow_infinity=False))
        x_point[0, i] = val

    assume(np.all(np.isfinite(x_point)))

    return model, x_point


# ---------------------------------------------------------------------------
# Property 3: Polynomial Jacobian consistency with finite differences
# ---------------------------------------------------------------------------


class TestPolynomialJacobianConsistency:
    """Property 3: Polynomial Jacobian consistency with finite differences.

    For any trained PolynomialModel and any valid input point x within the
    model's input bounds, the analytical Jacobian returned by jacobian(x)
    agrees with the central finite-difference approximation within tolerance 1e-6.

    **Validates: Requirements 4.6**
    """

    @given(data=polynomial_model_and_point())
    @settings(max_examples=100, deadline=None)
    def test_jacobian_matches_finite_difference(self, data):
        """Analytical Jacobian matches central finite-difference approximation.

        **Validates: Requirements 4.6**
        """
        model, x_point = data

        nind = model.nind
        ndep = len(model.outputs)

        # Compute analytical Jacobian: shape (npts, ndep, nind)
        analytical_jac = model.jacobian(x_point)
        assert analytical_jac.shape == (1, ndep, nind), (
            f"Expected Jacobian shape (1, {ndep}, {nind}), got {analytical_jac.shape}"
        )

        # Compute finite-difference Jacobian using numdifftools
        # For each response, compute the gradient (partial derivatives w.r.t. inputs)
        # using nd.Gradient, which is more robust for scalar-valued functions.
        fd_jac = np.zeros((ndep, nind))
        for j in range(ndep):
            def eval_response_j(x_flat, resp_idx=j):
                """Evaluate a single response at a point."""
                x_2d = x_flat.reshape(1, -1)
                return model.eval_np(x_2d)[0, resp_idx]

            grad_func = nd.Gradient(eval_response_j, method='central')
            fd_jac[j, :] = grad_func(x_point.flatten())

        # Compare: analytical_jac[0] has shape (ndep, nind)
        # fd_jac has shape (ndep, nind)
        np.testing.assert_allclose(
            analytical_jac[0],
            fd_jac,
            atol=1e-6,
            rtol=1e-6,
            err_msg=(
                f"Analytical Jacobian does not match finite-difference approximation "
                f"for model with nind={nind}, ndep={ndep}, degree={model.degree}"
            ),
        )
