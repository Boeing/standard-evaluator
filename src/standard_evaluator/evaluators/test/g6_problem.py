"""G6 constrained optimization benchmark problem.

A 2D constrained optimization benchmark from the literature with a narrow
crescent-shaped feasible region defined by two nonlinear inequality constraints.

Reference:
    Floudas, C.A. and Pardalos, P.M., "A Collection of Test Problems for
    Constrained Global Optimization Algorithms", Springer-Verlag, 1990.
"""

# pylint: disable=W0223

import numpy as np
import pandas as pd

from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class G6Problem(TestEvaluator):
    r"""Implement the G6 constrained optimization benchmark problem.

    .. math::
        \begin{align}
            \min\quad & f(x) = (x_1 - 10)^3 + (x_2 - 20)^3 \\[1em]
            \text{s.t.}\quad & g_1(x) = -(x_1 - 5)^2 - (x_2 - 5)^2 + 100 \leq 0 \\
            & g_2(x) = (x_1 - 6)^2 + (x_2 - 5)^2 - 82.81 \leq 0
        \end{align}

    Variable bounds:

    .. math::
        13 \leq x_1 \leq 100, \quad 0 \leq x_2 \leq 100

    Known optimal solution:

    .. math::
        x^* = (14.095, 0.84296), \quad f(x^*) \approx -6961.81388

    Initial guess (feasible):

    .. math::
        x_0 = (20.0, 5.5)
    """

    def _create_opt_problem(self) -> OptProblem:
        """Create the G6 constrained optimization problem.

        Returns:
            OptProblem: Configured optimization problem instance.
        """
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=3, name="g6_problem"
        )

        # Define variables
        var_names = ["x1", "x2"]
        var_bounds = ([13.0, 100.0], [0.0, 100.0])
        defaults = self._def_initial_guess()

        for var, name, default, bounds in zip(
            new_prob.variables, var_names, defaults, var_bounds
        ):
            var.name = name
            var.default = default
            var.bounds = bounds

        # Define responses
        resp_names = ["f", "g1", "g2"]
        resp_bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]

        for resp, name, bounds in zip(new_prob.responses, resp_names, resp_bounds):
            resp.name = name
            resp.bounds = bounds

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["g1", "g2"]

        # Define the description
        new_prob.description = r"""$$
\begin{align}
    \min\quad & f(x) = (x_1 - 10)^3 + (x_2 - 20)^3 \\
    \text{s.t.}\quad & g_1(x) = -(x_1 - 5)^2 - (x_2 - 5)^2 + 100 \leq 0 \\
    & g_2(x) = (x_1 - 6)^2 + (x_2 - 5)^2 - 82.81 \leq 0
\end{align}
$$

Variable bounds:
$$
13 \leq x_1 \leq 100, \quad 0 \leq x_2 \leq 100
$$

Known optimal solution:
$$
x^* = (14.095, 0.84296), \quad f(x^*) \approx -6961.81388
$$"""

        # Define the citation
        new_prob.cite = (
            'Floudas, C.A. and Pardalos, P.M., "A Collection of Test Problems '
            'for Constrained Global Optimization Algorithms", Springer-Verlag, 1990.'
        )

        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide a feasible initial guess for an optimizer.

        Returns:
            list: The initial guess values [x1, x2].
        """
        return [20.0, 5.5]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Evaluate the G6 objective and constraint functions.

        Args:
            sites: DataFrame containing input values (x1, x2), updated
                in-place with computed responses (f, g1, g2).
        """
        sites["f"] = (sites.x1 - 10.0) ** 3 + (sites.x2 - 20.0) ** 3
        sites["g1"] = -(sites.x1 - 5.0) ** 2 - (sites.x2 - 5.0) ** 2 + 100.0
        sites["g2"] = (sites.x1 - 6.0) ** 2 + (sites.x2 - 5.0) ** 2 - 82.81

    def _def_known_solution(self) -> pd.Series:
        """Provide the known optimal solution for the G6 problem.

        Returns:
            pd.Series: Variable values at the known optimum.
        """
        return pd.Series(data={"x1": 14.095, "x2": 0.84296})
