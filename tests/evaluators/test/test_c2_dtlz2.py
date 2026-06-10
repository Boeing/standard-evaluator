"""Testing of the C2_DTLZ2 class"""

import pytest
import pandas as pd
import numpy as np
from standard_evaluator import OptProblem
from standard_evaluator.evaluators.test import C2_DTLZ2


def test_c2_dtlz2():
    """Test the DasTruss test function"""
    # Instantiate the test function
    test_func = C2_DTLZ2()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    # Duplicate to make sure vectorization works
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )
    # Change x1, x2, and x3
    initial_guess.loc[1, "x1"] = 0.8
    initial_guess.loc[1, "x2"] = 0.2
    initial_guess.loc[1, "x3"] = 0.8
    initial_guess.loc[1, "x4"] = 0.6
    initial_guess.loc[1, "x5"] = 0.7
    initial_guess.loc[1, "x6"] = 0.5
    initial_guess.loc[1, "x7"] = 0.4

    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    assert test_func.responses == ["f1", "f2", "f3"]

    assert initial_guess.iloc[0].f1 == pytest.approx(0.5000000000000001)
    assert initial_guess.iloc[0].f2 == pytest.approx(0.5000000000000001)
    assert initial_guess.iloc[0].f3 == pytest.approx(0.7071067811865476)

    assert initial_guess.iloc[1].f1 == pytest.approx(0.337976520068172)
    assert initial_guess.iloc[1].f2 == pytest.approx(0.10981522823440522)
    assert initial_guess.iloc[1].f3 == pytest.approx(1.0937149937394264)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = C2_DTLZ2()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 7, "Expected 7 variables"
    assert len(opt_problem.responses) == 3, "Expected 3 responses"

    # Check that the returned object is an OptProblem instance
    assert isinstance(opt_problem, OptProblem)

    # Check variables
    assert isinstance(opt_problem.variables, list)

    # Check name, bounds, type for variables
    expected_variable_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    expected_var_bounds = (
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    )

    for var, name, expected_bounds in zip(
        opt_problem.variables, expected_variable_names, expected_var_bounds
    ):
        assert var.name == name, f"Variable names do not match expected values"
        assert (
            var.bounds == expected_bounds
        ), f"Bounds for {var.name} are not set correctly"
        assert var.class_type == "float"

    # Check name, bounds, type for responses
    expected_resp_names = ["f1", "f2", "f3"]
    expected_resp_bounds = ((0.0, np.inf), (0.0, np.inf), (0.0, np.inf))

    for resp, name, expected_bounds in zip(
        opt_problem.responses, expected_resp_names, expected_resp_bounds
    ):
        assert resp.name == name, f"Variable names do not match expected values"
        assert (
            resp.bounds == expected_bounds
        ), f"Bounds for {var.name} are not set correctly"
        assert resp.class_type == "float"

    assert isinstance(opt_problem.responses, list)

    # Check the description
    expected_description = r"""$$
General DTLZ2 Problem Formulation

For $M$ objectives and $n$ decision variables, where $k = n - M + 1$, the problem is:

$$
\begin{aligned}
\text{Minimize } & f_1(\mathbf{x}) = (1 + g(\mathbf{x}_M)) \prod_{i=1}^{M-1} \cos\left(x_i \frac{\pi}{2}\right) \\
\text{Minimize } & f_2(\mathbf{x}) = (1 + g(\mathbf{x}_M)) \left(\prod_{i=1}^{M-2} \cos\left(x_i \frac{\pi}{2}\right)\right) \sin\left(x_{M-1} \frac{\pi}{2}\right) \\
\text{Minimize } & f_3(\mathbf{x}) = (1 + g(\mathbf{x}_M)) \left(\prod_{i=1}^{M-3} \cos\left(x_i \frac{\pi}{2}\right)\right) \sin\left(x_{M-2} \frac{\pi}{2}\right) \\
& \vdots \\
\text{Minimize } & f_{M-1}(\mathbf{x}) = (1 + g(\mathbf{x}_M)) \cos\left(x_1 \frac{\pi}{2}\right) \sin\left(x_2 \frac{\pi}{2}\right) \\
\text{Minimize } & f_M(\mathbf{x}) = (1 + g(\mathbf{x}_M)) \sin\left(x_1 \frac{\pi}{2}\right)
\end{aligned}
$$

where

$$
g(\mathbf{x}_M) = \sum_{x_i \in \mathbf{x}_M} (x_i - 0.5)^2
$$

and

- $\mathbf{x} = (x_1, x_2, \ldots, x_n)$,
- $\mathbf{x}_M = (x_M, x_{M+1}, \ldots, x_n)$,
- $x_i \in [0,1]$ for all $i$.

The Pareto-optimal front corresponds to $g(\mathbf{x}_M) = 0$, i.e., $x_i = 0.5$ for all $i \in \{M, \ldots, n\}$, and the objective vectors lie on the unit hypersphere:

$$
\sum_{m=1}^M f_m^2 = 1
$$
$$"""

    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = "Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002). Scalable test problems for evolutionary multiobjective optimization. Evolutionary multiobjective optimization, 105-145."
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = C2_DTLZ2()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f1 == pytest.approx(0.5000)
    assert initial_guess_df.iloc[0].f2 == pytest.approx(0.5000)
    assert initial_guess_df.iloc[0].f3 == pytest.approx(0.7071067811865476)
