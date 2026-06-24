"""Property-based test for polynomial model least-squares fit accuracy.

Feature: surrogate-model-migration
Property 2: Polynomial model least-squares fit accuracy

**Validates: Requirements 4.5**
"""

from math import comb

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from standard_evaluator.problem import OptProblem, FloatVariable
from standard_evaluator.surrogate_models.polynomial_model import (
    PolynomialModel,
    PolynomialModelOptions,
)
from standard_evaluator.surrogate_models.polynomial_model_utils.monomial_ordering import (
    grlex_ordering_to_deg,
)


def build_vandermonde(sites_input: np.ndarray, degree: int) -> np.ndarray:
    """Build the monomial Vandermonde matrix for a given set of input sites.

    Parameters
    ----------
    sites_input : np.ndarray
        Input sites array of shape (nsites, nind).
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Vandermonde-like matrix of shape (nsites, nterms).
    """
    nind = sites_input.shape[1]
    deg_exp = np.array(grlex_ordering_to_deg(nind, degree))
    # Compute monomials: product of x_i^e_i for each monomial
    x = sites_input[:, np.newaxis, :]  # (nsites, 1, nind)
    vandermonde = np.prod(x ** deg_exp, axis=-1)  # (nsites, nterms)
    return vandermonde


@st.composite
def polynomial_model_data(draw):
    """Generate valid polynomial model training data ensuring a well-conditioned system.

    Strategy:
    - nind independent variables (1-3)
    - degree of polynomial (1-2)
    - 1 response
    - Sites with exactly nterms points
    - Use a Latin Hypercube-like approach: each dimension gets a random
      permutation of evenly-spaced values, ensuring good space coverage
    """
    nind = draw(st.integers(min_value=1, max_value=3))
    degree = draw(st.integers(min_value=1, max_value=2))

    # Number of polynomial terms for this configuration
    nterms = comb(nind + degree, degree)

    # Use exactly nterms sites so the system is exactly determined
    nsites = nterms

    # Use fixed bounds [0, 1] to keep things well-scaled
    variables = [
        FloatVariable(name=f"x{i}", bounds=(0.0, 1.0))
        for i in range(nind)
    ]
    responses = [FloatVariable(name="y0")]
    opt_problem = OptProblem(variables=variables, responses=responses)

    # Generate input sites: use independently permuted linspace values for each dim
    site_values = np.empty((nsites, nind))
    for i in range(nind):
        # Evenly spaced points in (0, 1)
        points = np.linspace(0.05, 0.95, nsites)
        perm = draw(st.permutations(list(range(nsites))))
        site_values[:, i] = points[list(perm)]

    # Compute the Vandermonde matrix and check condition number
    vandermonde = build_vandermonde(site_values, degree)
    cond_num = np.linalg.cond(vandermonde)
    # Skip ill-conditioned systems (condition number > 1e6 means
    # we can't expect 1e-10 accuracy from lstsq)
    assume(cond_num < 1e6)

    # Generate random response values
    response_values = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=nsites,
            max_size=nsites,
        ).map(lambda vals: np.array(vals).reshape(-1, 1))
    )

    # Build the DataFrame
    columns = [f"x{i}" for i in range(nind)] + ["y0"]
    data = np.hstack([site_values, response_values])
    sites_df = pd.DataFrame(data, columns=columns)

    return sites_df, opt_problem, degree


@given(data=polynomial_model_data())
@settings(max_examples=100, deadline=None)
def test_polynomial_fit_accuracy(data):
    """Property 2: For any valid set of training sites with non-constant variables
    and sufficient points for the requested degree, fitting a PolynomialModel and
    evaluating at the training sites reproduces the training responses within
    tolerance 1e-10.

    Feature: surrogate-model-migration
    Property 2: Polynomial model least-squares fit accuracy

    **Validates: Requirements 4.5**
    """
    sites_df, opt_problem, degree = data

    nind = len(opt_problem.variables)
    nterms = comb(nind + degree, degree)
    nsites = len(sites_df)

    # Get the input portion of sites
    input_cols = [f"x{i}" for i in range(nind)]
    site_inputs = sites_df[input_cols].values

    # Create and fit the polynomial model
    options = PolynomialModelOptions(degree=degree)
    model = PolynomialModel(sites=sites_df, opt_problem=opt_problem, options=options)

    # Evaluate at the training sites
    predictions = model.eval_np(site_inputs)

    # Get expected response values
    expected = sites_df[["y0"]].values

    # Check that predictions match training data within tolerance
    max_error = np.max(np.abs(predictions - expected))
    assert max_error < 1e-10, (
        f"Polynomial model fit accuracy failed: max error = {max_error} "
        f"(tolerance 1e-10) for nind={nind}, degree={degree}, nsites={nsites}, nterms={nterms}"
    )
