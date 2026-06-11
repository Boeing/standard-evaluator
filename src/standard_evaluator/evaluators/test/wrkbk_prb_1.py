"""A Python module to provide access to WrkBkPrb1 optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class WrkBkPrb1(TestEvaluator):
    r"""work book problem 1 from the SOCS user's guide page 23

    minimize f(x1, x2)

    subject to

    1.0e-8 <= x1, x2 <= 10.0

    1.0 <= c <= 100.0

    where

    :math:`c(x_1, x_2) = x_1x_2`

    :math:`f(x_1, x_2) = x_1^2 + x_2^2 + \ln(c)`

    x0 = (0.5, 2.0)
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for a workbook example from the SOCS user's guide.

        This method initializes an optimization problem with two independent variables and two dependent responses.
        It sets default values, bounds, scales, and names for the variables and responses. The objective function
        and constraints are defined based on the specified mathematical relationships.

        The optimization problem is defined as follows:

        - **Objective**: Minimize \( f(x_1, x_2) \)
        - **Subject to**:
            - \( 1.0e-8 \leq x_1, x_2 \leq 10.0 \)
            - \( 1.0 \leq c \leq 100.0 \)
        - **Where**:
            - \( c(x_1, x_2) = x_1 x_2 \)
            - \( f(x_1, x_2) = x_1^2 + x_2^2 + \ln(c) \)

        The initial guess for the variables is set to \( x_0 = (0.5, 2.0) \).

        Returns:
            OptProblem: An instance of the OptProblem class containing the defined optimization problem.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=2, name="wrkbk_prb_1"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(2)]
        var_bounds = ([1e-08, 10.0], [1e-08, 10.0])
        var_scales = [0.1, 0.1]

        for var, local_default, local_bound, local_scale, name in zip(
            new_prob.variables, defaults, var_bounds, var_scales, var_names
        ):
            var.default = local_default
            var.bounds = local_bound
            var.scale = local_scale
            var.name = name

        resp_names = ["f", "c"]
        resp_bounds = ([-np.inf, np.inf], [1.0, 100.0])

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
        new_prob.description = r"""$$
work book problem 1 from the SOCS user's guide page 23

    minimize f(x1, x2)

    subject to

    1.0e-8 <= x1, x2 <= 10.0

    1.0 <= c <= 100.0

    where

    :math:`c(x_1, x_2) = x_1x_2`

    :math:`f(x_1, x_2) = x_1^2 + x_2^2 + \ln(c)`

    x0 = (0.5, 2.0)
$$"""
        # Define the citation
        new_prob.cite = 'Beiqing Huang and Xiaoping Du, "A robust design method using variable transformation and Gauss-Hermite integration," *International Journal for Numerical Methods in Engineering*, Int. J. Numer. Meth. Engng 2006; 66:1841-1858.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [0.5, 2.0]

    def _evaluate(self, sites: pd.DataFrame):
        """Call to the WrkBkPrb1 function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["c"] = sites.x1 * sites.x2
        sites["f"] = sites.x1 * sites.x1 + sites.x2 * sites.x2 + np.log(sites.c)

    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.
        :return: DataFrame providing the optimal solution of the problem
        :rtype: list
        """
        return [1.0, 1.0]
