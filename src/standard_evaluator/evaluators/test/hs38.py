"""A Python module to provide access to hs38 optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class HS38(TestEvaluator):
    r"""Implement the Hock-Schittkowski number 38 problem.

    .. math::
        \min f(x) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2 + 90(x_4 - x_3^2)^2
        + (1-x_3)^2 + 10.1[(x_2-1)^2 + (x_4-1)^2] + 19.8(x_2-1)(x_4-1)

    x0 = (-3.0, -1.0, -3.0, -1.0)

    f(x0) = 19192.0

    x* = (1.0, 1.0, 1.0, 1.0)

    f(x*) = 0.0
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem instance with predefined settings.

        This method initializes a new optimization problem with 4 variables and 1 objective function,
        using the 'hs38' test problem as a basis. It sets default values, bounds, and scales for the
        variables and responses, and defines the objective function and its description.

        The optimization problem is defined as follows:

        $$
        \min f(x) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2 + 90(x_4 - x_3^2)^2
        + (1-x_3)^2 + 10.1[(x_2-1)^2 + (x_4-1)^2] + 19.8(x_2-1)(x_4-1)
        $$

        The initial guess for the variables is set to:
        $$
        x0 = (-3.0, -1.0, -3.0, -1.0)
        $$

        The function value at the initial guess is:
        $$
        f(x0) = 19192.0
        $$

        The optimal solution is:
        $$
        x^* = (1.0, 1.0, 1.0, 1.0)
        $$

        The function value at the optimal solution is:
        $$
        f(x^*) = 0.0
        $$

        The problem is cited from:
        Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes."
        Journal of optimization theory and applications 30 (1980): 127-129.

        Returns:
            OptProblem: An instance of the optimization problem with defined variables, objectives,
            constraints, and other settings.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(4, 1, "hs38")
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(7)]
        for var, name, local_default in zip(new_prob.variables, var_names, defaults):
            var.bounds = [-10.0, 10.0]
            var.scale = 0.1
            var.default = local_default
            var.name = name

        resp_names = [f"f"]
        # Define and set the scales
        for resp, name in zip(new_prob.responses, resp_names):
            resp.scale = 0.0001
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f"]

        # Define th description of the problem
        new_prob.description = r"""$$
\min f(x) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2 + 90(x_4 - x_3^2)^2
+ (1-x_3)^2 + 10.1[(x_2-1)^2 + (x_4-1)^2] + 19.8(x_2-1)(x_4-1)
$$

x0 = (-3.0, -1.0, -3.0, -1.0)

f(x0) = 19192.0

x* = (1.0, 1.0, 1.0, 1.0)

f(x*) = 0.0"""
        # Define the citation
        new_prob.cite = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [-3.0, -1.0, -3.0, -1.0]

    def _def_known_solution(self) -> list:
        """Provide the variable values of the known optimal solution.

        :return: List providing the variable values of the optimal solution of the problem
        :rtype: list
        x* = (1.0, 1.0, 1.0, 1.0)
        f(x*) = 0.0
        """
        return [1.0] * 4

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the HS38 function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["f"] = (
            100.0 * (sites.x2 - sites.x1 * sites.x1) * (sites.x2 - sites.x1 * sites.x1)
            + (1.0 - sites.x1) * (1.0 - sites.x1)
            + 90.0 * (sites.x4 - sites.x3 * sites.x3) * (sites.x4 - sites.x3 * sites.x3)
            + (1.0 - sites.x3) * (1.0 - sites.x3)
            + 10.1
            * (
                (sites.x2 - 1.0) * (sites.x2 - 1.0)
                + (sites.x4 - 1.0) * (sites.x4 - 1.0)
            )
            + 19.8 * (sites.x2 - 1.0) * (sites.x4 - 1.0)
        )
