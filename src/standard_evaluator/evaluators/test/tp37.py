"""A Python module to provide access to TP37_WDTF optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class TP37(TestEvaluator):
    """Implement the Rosenbrock Post Office problem, with derivatives
    minimize y1 subject to

    0 <= x(i) <= 42, i=1,...,3

    y2 <= 0.0

    y3 >= 0.0

    x* = ( 0.24000000E+02 0.12000000E+02 0.12000000E+02 )

    at x* (y1, y2, y3) = (-3456.0, 0.0, 72.0)
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for the Rosenbrock Post Office example.

        This method initializes an optimization problem with three design variables
        and three responses. The objective is to minimize the first response while
        ensuring that the second and third responses meet specified constraints.

        Design Variables:
            - x1: First design variable (bounded between 0.0 and 42.0)
            - x2: Second design variable (bounded between 0.0 and 42.0)
            - x3: Third design variable (bounded between 0.0 and 42.0)

        Responses:
            - y1: The objective function to minimize.
            - y2: A constraint that must be less than or equal to 0.0.
            - y3: A constraint that must be greater than or equal to 0.0.

        Objectives:
            - Minimize y1.

        Constraints:
            - y2 must be less than or equal to 0.0.
            - y3 must be greater than or equal to 0.0.

        Optimal Solution:
            - The optimal design variables are approximately:
            $$ x^* = (0.24 \times 10^2, 0.12 \times 10^2, 0.12 \times 10^2) $$
            - At the optimal point, the responses are:
            $$ (y1, y2, y3) = (-3456.0, 0.0, 72.0) $$

        Returns:
            OptProblem: An instance of the optimization problem configured for the
            Rosenbrock Post Office example.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=3, num_dependent=3, name="tp37"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(3)]

        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.default = local_default
            var.bounds = [0.0, 42.0]
            var.name = name

        resp_names = [f"y{items + 1}" for items in range(3)]
        resp_bounds = ([-np.inf, np.inf], [-np.inf, 0.0], [0.0, np.inf])
        resp_scales = [0.0001, 0.1, 0.1]

        for (
            resp,
            local_bounds,
            local_scale,
            name,
        ) in zip(new_prob.responses, resp_bounds, resp_scales, resp_names):
            resp.name = name
            resp.bounds = local_bounds
            resp.scale = local_scale

        # Define objectives and constraints
        new_prob.objectives = ["y1"]
        new_prob.constraints = ["y2", "y3"]

        # Define th description of the problem
        new_prob.description = """Implement the Rosenbrock Post Office problem, with derivatives
    minimize y1 subject to

    0 <= x(i) <= 42, i=1,...,3

    y2 <= 0.0

    y3 >= 0.0

    x* = ( 0.24000000E+02 0.12000000E+02 0.12000000E+02 )

    at x* (y1, y2, y3) = (-3456.0, 0.0, 72.0)"""
        # Define the citation
        new_prob.cite = ""
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [10.0, 10.0, 10.0]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the TP37_WD function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["y1"] = -sites.x1 * sites.x2 * sites.x3
        sites["y2"] = sites.x1 + 2 * sites.x2 + 2 * sites.x3 - 72
        sites["y3"] = sites.x1 + 2 * sites.x2 + 2 * sites.x3

    def __gradient__(self, sites: pd.DataFrame) -> pd.DataFrame:
        """Calculate the TP37_WD function gradient
        :param df: The dataframe that contains the input values
        :type df: DataFrame
        :return: The Gradient values
        :rtype: DataFrame
        """

        grad = pd.DataFrame()
        # Derivatives w.r.t. x1
        grad["y1_1"] = -sites.x2 * sites.x3
        grad["y2_1"] = 1.0
        grad["y3_1"] = 1.0
        # Derivatives w.r.t. x2
        grad["y1_2"] = -sites.x1 * sites.x3
        grad["y2_2"] = 2.0
        grad["y3_2"] = 2.0
        # Derivatives w.r.t. x3
        grad["y1_3"] = -sites.x1 * sites.x2
        grad["y2_3"] = 2.0
        grad["y3_3"] = 2.0

    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.
        :return: DataFrame providing the optimal solution of the problem
        :rtype: list
        """
        return [24.0, 12.0, 12.0]
