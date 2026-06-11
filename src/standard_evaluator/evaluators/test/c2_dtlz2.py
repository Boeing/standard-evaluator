"""A Python module to provide access to C2_DTLZ2 optimization test problems"""

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator

class C2_DTLZ2(TestEvaluator):
    """
    Compute the objective values for the DTLZ2 multi-objective optimization problem.

    The DTLZ2 problem is defined for M objectives and n decision variables, where
    n >= M and k = n - M + 1. The decision vector x is composed of n variables,
    each in the range [0, 1].

    Problem formulation:
        Minimize f_m(x) for m = 1, ..., M, where

        f_1(x) = (1 + g(x_M)) * prod_{i=1}^{M-1} cos(x_i * pi/2)
        f_2(x) = (1 + g(x_M)) * (prod_{i=1}^{M-2} cos(x_i * pi/2)) * sin(x_{M-1} * pi/2)
        ...
        f_{M-1}(x) = (1 + g(x_M)) * cos(x_1 * pi/2) * sin(x_2 * pi/2)
        f_M(x) = (1 + g(x_M)) * sin(x_1 * pi/2)

    where
        g(x_M) = sum_{x_i in x_M} (x_i - 0.5)^2,
        x_M = (x_M, x_{M+1}, ..., x_n).

    Parameters
    ----------
    x : array-like of shape (n,)
        Decision variable vector, with each element in [0, 1].
    M : int
        Number of objectives.

    Returns
    -------
    f : list of float
        Objective values [f_1, f_2, ..., f_M].

    Notes
    -----
    - The Pareto-optimal front corresponds to g(x_M) = 0, i.e., x_i = 0.5 for all i >= M.
    - On the Pareto front, the objective vectors lie on the unit hypersphere:
      sum_{m=1}^M f_m^2 = 1.
    - The parameter k = n - M + 1 controls the number of distance variables and
      typically k >= 10.
    - The function g(x_M) introduces multimodality and convergence difficulty.

    References
    ----------
    Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002).
    Scalable test problems for evolutionary multiobjective optimization.
    Evolutionary multiobjective optimization, 105-145.
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates and configures an optimization problem instance for the DTLZ2 test problem.

        This method sets up a multi-objective optimization problem with 7 decision variables 
        and 3 objectives, following the DTLZ2 formulation. It initializes variable names, bounds, 
        scales, and default values, as well as response bounds, scales, and names. The problem's 
        objectives and constraints are defined, along with a detailed LaTeX-formatted description 
        of the problem formulation and its Pareto-optimal front characteristics. A citation for 
        the original DTLZ2 problem publication is also included.

        Returns:
            OptProblem: An instance of the optimization problem configured for the DTLZ2 test case.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=7, num_dependent=3, name="c2_dtlz2"
        )

        var_names = [f"x{items + 1}" for items in range(7)]
        # Define default values
        defaults = self._def_initial_guess()
        var_bounds = (
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        )
        var_scales = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for var, local_default, local_bound, local_scale, name in zip(
            new_prob.variables, defaults, var_bounds, var_scales, var_names
        ):
            var.bounds = local_bound
            var.scale = local_scale
            var.default = local_default
            var.name = name

        # # Define the bounds on the constraints
        resp_bounds = ([0.0, np.inf], [0.0, np.inf], [0.0, np.inf])

        # Define and set the scales
        resp_scales = [1.0, 1.0, 1.0]

        resp_names = [f"f{items + 1}" for items in range(3)]
        for resp, scale, local_bounds, name in zip(
            new_prob.responses, resp_scales, resp_bounds, resp_names
        ):
            resp.scale = scale
            resp.name = name
            resp.bounds = local_bounds

        # Define objectives and constraints
        new_prob.objectives = ["f1", "f2", "f3"]
        new_prob.constraints = []

        # Define th description of the problem
        new_prob.description = r"""$$
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
        # Define the citation
        new_prob.cite = "Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002). Scalable test problems for evolutionary multiobjective optimization. Evolutionary multiobjective optimization, 105-145."
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        Here, the initial guess list with 7 variables, each set to 0.5.
        :rtype: list
        """
        return [0.5] * 7

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """
        Call to the C2DTLZ2 function
        Evaluate the DTLZ2 objectives for each row in the input DataFrame `sites`.
        Updates the DataFrame in-place by adding columns for each objective.

        :sites : pd.DataFrame
            DataFrame containing the decision variables as columns named x1, x2, ..., x7.
            The DataFrame is updated with new columns f1, f2, f3 for the objective values.
        :type sites: DataFrame
        """
        var_num = 7
        obj_num = 3
        # k is a user-defined parameter (often k=10 or more).
        k = var_num - obj_num + 1  # k = 5

        var_sites = sites[[f"x{i+1}" for i in range(var_num)]].to_numpy()

        # Compute sum_func(x_M) = sum_{i=obj_num}^{var_num} (x_i - 0.5)^2
        sum_func = np.sum((var_sites[:, obj_num - 1 :] - 0.5) ** 2, axis=1)

        obj_values = np.zeros((var_sites.shape[0], obj_num))

        # f1 = (1+sum_func) * prod_{i=1}^{obj_num-1} cos(x_i * pi/2)
        obj_values[:, 0] = (1 + sum_func) * np.prod(
            np.cos(var_sites[:, : obj_num - 1] * np.pi / 2), axis=1
        )

        # f2 = (1+sum_func) * (prod_{i=1}^{obj_num-2} cos(x_i * pi/2)) * sin(x_{obj_num-1} * pi/2)
        if obj_num > 1:
            obj_values[:, 1] = (
                (1 + sum_func)
                * np.prod(np.cos(var_sites[:, : obj_num - 2] * np.pi / 2), axis=1)
                * np.sin(var_sites[:, obj_num - 2] * np.pi / 2)
            )

        # f3 = (1+sum_func) * sin(x_1 * pi/2)
        if obj_num > 2:
            obj_values[:, 2] = (1 + sum_func) * np.sin(var_sites[:, 0] * np.pi / 2)

        for i in range(obj_num):
            sites[f"f{i+1}"] = obj_values[:, i]
