"""A Python module to provide access to hs47 optimization test problems"""

# pylint: disable=W0223

import pandas as pd
import numpy as np
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class HS47(TestEvaluator):
    r"""Implement the Hock-Schittkowski number 47 problem.

    .. math::
        \begin{align}
            \min\quad & f(x) = (x_1 - x_2)^2 + (x_2 - x_3)^3 +(x_3 - x_4)^4 + (x_4 - x_5)^4\\[1em]
            \text{s.t}\quad & c_1 = x_1 + x_2^2 + x_3^3 - 3 = 0\\
            & c_2 = x_2 - x_3^2 + x_4 - 1 = 0\\
            & c_3 = x_1x_5 - 1 = 0
        \end{align}

    :math:`x_0 = (2.0,~ \sqrt{2.0},~ -1.0,~ 2.0 - \sqrt{2.0},~ 0.5)`

    f(x0) = 12.4954368

    x* = (1.0, 1.0, 1.0, 1.0, 1.0)

    f(x*) = 0.0
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem instance with predefined objectives, constraints, and initial values.

        This method initializes a new optimization problem with the following characteristics:
        
        - **Variables**: The problem consists of 5 variables and 4 responses.
        - **Objectives**: The objective function to minimize is defined as:
        $$
        f(x) = (x_1 - x_2)^2 + (x_2 - x_3)^3 +(x_3 - x_4)^4 + (x_4 - x_5)^4
        $$
        - **Constraints**: The problem includes three equality constraints:
        $$
        \begin{align}
            c_1 &= x_1 + x_2^2 + x_3^3 - 3 = 0\\
            c_2 &= x_2 - x_3^2 + x_4 - 1 = 0\\
            c_3 &= x_1x_5 - 1 = 0
        \end{align}
        $$
        - **Initial Guess**: The initial point for the optimization is set to:
        $$
        x_0 = (2.0,~ \sqrt{2.0},~ -1.0,~ 2.0 - \sqrt{2.0},~ 0.5)
        $$
        - **Scales**: The scales for the responses are defined as [0.1, 1.0, 1.0, 1.0].
        - **Citation**: The problem is based on the work by Hock and Schittkowski (1980), which provides test examples for nonlinear programming codes.

        Returns:
            OptProblem: An instance of the optimization problem with the defined objectives, constraints, and initial values.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(5, 4, "hs47")
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(5)]
        for var, name, local_default in zip(new_prob.variables, var_names, defaults):
            var.default = local_default
            var.name = name

        # Define the bounds on the constraints
        for resp in new_prob.responses[1:]:
            resp.bounds = (0, 0)

        # Define response names and set the scales
        resp_names = [f"f"] + [f"c{items + 1}" for items in range(3)]
        scales = [0.1, 1.0, 1.0, 1.0]
        for resp, name, scale in zip(new_prob.responses, resp_names, scales):
            resp.scale = scale
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["c1", "c2", "c3"]

        # Define th description of the problem
        new_prob.description = r"""$$
\begin{aligned}
    \min\quad & f(x) = (x_1 - x_2)^2 + (x_2 - x_3)^3 +(x_3 - x_4)^4 + (x_4 - x_5)^4\\[1em]
    \text{s.t}\quad & c_1 = x_1 + x_2^2 + x_3^3 - 3 = 0\\
    & c_2 = x_2 - x_3^2 + x_4 - 1 = 0\\
    & c_3 = x_1x_5 - 1 = 0
\end{aligned}
$$

The initial point is given by:

$$
x_0 = (2.0,~ \sqrt{2.0},~ -1.0,~ 2.0 - \sqrt{2.0},~ 0.5)
$$"""
        # Define the citation
        new_prob.cite = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        x0 = (2.0, sqrt(2.0), -1.0, 2.0 - sqrt(2.0), 0.5)
        f(x0) = 12.4954368
        """
        return [2.0, np.sqrt(2.0), -1.0, 2.0 - np.sqrt(2.0), 0.5]

    def _def_known_solution(self) -> list:
        """Provide the variable values of the known optimal solution.

        :return: List providing the variable values of the optimal solution of the problem
        :rtype: list
        x* = (1.0, 1.0, 1.0, 1.0, 1.0)
        f(x*) = 0.0
        """
        return [1.0] * 5

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the HS47 function

        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        f1 = sites.x1 - sites.x2
        f2 = sites.x2 - sites.x3
        f3 = sites.x3 - sites.x4
        f4 = sites.x4 - sites.x5
        sites["f"] = (f1 * f1) + (f2 * f2) + (f3 * f3 * f3 * f3) + (f4 * f4 * f4 * f4)

        sites["c1"] = (
            sites.x1 + sites.x2 * sites.x2 + sites.x3 * sites.x3 * sites.x3 - 3.0
        )
        sites["c2"] = sites.x2 - sites.x3 * sites.x3 + sites.x4 - 1.0
        sites["c3"] = sites.x1 * sites.x5 - 1.0
