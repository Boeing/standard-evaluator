"""A Python module to provide access to OptlibTest optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class OptlibTest(TestEvaluator):
    r"""The example can be found in the Optlib 6.2 manual on page 308.
    The example can be found in the SOCS 7.1 manual on page 625.

    .. math::
        \begin{aligned}
            \min\quad & f = (x_1 - 1)^2 + (x_1 - x_2)^2 + (x_2 - x_3)^4\\[1em]
            \text{s.t.}\quad & c = x_1(1 + x_2^2) + x_3^4 - 4 - 3\sqrt{2} = 0
        \end{aligned}

    The optimal solution in the SOCS manual is

    x* = [1.104859034205678, 1.196674180655277, 1.535262258200661]

    with f = 0.032568200256415, c = 1.253397385880817e-010
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Create an optimization problem for testing purposes.

        This method constructs an optimization problem with three independent
        variables and two dependent responses. It initializes the problem with
        default values, sets bounds for the variables and responses, and defines
        the objectives and constraints.

        The problem description includes a mathematical formulation of the
        optimization problem, referencing examples from the Optlib and SOCS manuals.

        Returns:
        --------
        OptProblem
            An instance of the OptProblem class representing the constructed
            optimization problem.

        Notes:
        ------
        The optimization problem is defined with the following characteristics:
        - Three independent variables named "x1", "x2", and "x3" with bounds
        [0.0, 5.0].
        - Two dependent responses named "c" and "f".
        - The objective function is defined as 'f'.
        - The constraint is defined as 'c'.
        - The problem description includes a mathematical representation of the
        optimization problem, which is formatted in LaTeX.

        Example:
        --------
        problem = self._create_opt_problem()
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=3, num_dependent=2, name="optlib_test"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_bounds = ([0.0, 5.0], [0.0, 5.0], [0.0, 5.0])
        var_names = [f"x{items + 1}" for items in range(3)]

        for var, local_bound, local_default, name in zip(
            new_prob.variables, var_bounds, defaults, var_names
        ):
            var.default = local_default
            var.bounds = local_bound
            var.name = name

        resp_names = ["c", "f"]
        resp_bounds = ([0.0, 0.0], [-np.inf, np.inf])
        for (
            resp,
            local_bounds,
            name,
        ) in zip(new_prob.responses, resp_bounds, resp_names):
            resp.name = name
            resp.bounds = local_bounds

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["c"]

        # Define th description of the problem
        new_prob.description = r"""The example can be found in the Optlib 6.2 manual on page 308.
The example can be found in the SOCS 7.1 manual on page 625.

.. math::
    \begin{aligned}
        \min\quad & f = (x_1 - 1)^2 + (x_1 - x_2)^2 + (x_2 - x_3)^4\\[1em]
        \text{s.t.}\quad & c = x_1(1 + x_2^2) + x_3^4 - 4 - 3\sqrt{2} = 0
    \end{aligned}

The optimal solution in the SOCS manual is

x* = [1.104859034205678, 1.196674180655277, 1.535262258200661]

with f = 0.032568200256415, c = 1.253397385880817e-010"""
        # Define the citation
        new_prob.cite = ""
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [2.0] * 3

    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.
        :return: DataFrame providing the optimal solution of the problem
        :rtype: list
        """
        return [1.104859034205678, 1.196674180655277, 1.535262258200661]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the OptlibTest function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        tmp1 = sites.x1 - 1.0
        tmp2 = sites.x1 - sites.x2
        tmp3 = sites.x2 - sites.x3

        sites["f"] = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3 * tmp3 * tmp3
        sites["c"] = (
            sites.x1 * (1.0 + sites.x2 * sites.x2)
            + (sites.x3 * sites.x3 * sites.x3 * sites.x3)
            - 4.0
            - (3.0 * np.sqrt(2.0))
        )
