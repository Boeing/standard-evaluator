"""A Python module to provide access to HelicalValley optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class HelicalValley(TestEvaluator):
    r"""Small dimensional example function

    .. math::
        \begin{align}
            \min\quad & f = f_1^2 + f_2^2 + x_3^2\\[1em]
            \text{where}\quad & \theta = \begin{cases}
                \frac{\arctan(x_2/x_1)}{2\pi} & x_1 \geq 0\\
                \frac{\arctan(x_2/x_1)}{2\pi} + 0.5 & x_1 < 0
            \end{cases}\\
            & f_1 = 10(x_3 - 10\theta)\\
            & f_2 = 10(\sqrt{x_1^2 + x_2^2} - 1)
        \end{align}
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for the helical valley function.

        This method initializes a new optimization problem with three independent variables,
        sets default values for these variables, defines the scaling for the responses,
        and specifies the objectives and constraints of the problem. The objective function
        is defined as the sum of squares of two functions and the square of a third variable,
        with additional constraints based on the arctangent function.

        Returns:
            OptProblem: An instance of the OptProblem class containing the defined optimization
            problem, including objectives, constraints, variable defaults, and a description
            of the problem.

        Citation:
            J. J. More, B. S. Garbow and K. E. Hillstrom, “Testing Unconstrained Optimization
            Software,” ACM Transactions on Mathematical Software, Vol. 7, No. 1, 1981,
            pp. 19-31. doi:10.1145/355934.355936
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=3, num_dependent=1, name="helical_valley"
        )

        var_names = [f"x{items + 1}" for items in range(3)]
        # Define default values
        defaults = self._def_initial_guess()

        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.default = local_default
            var.name = name

        resp_names = f"f"
        # Define the bounds on the constraints
        for resp in new_prob.responses:
            resp.scale = 0.0001
            resp.name = resp_names

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = []

        # Define th description of the problem
        new_prob.description = r"""$$
\begin{aligned}
    \min\quad & f = f_1^2 + f_2^2 + x_3^2\\[1em]
    \text{where}\quad & \theta = \begin{cases}
        \frac{\arctan(x_2/x_1)}{2\pi} & x_1 \geq 0\\
        \frac{\arctan(x_2/x_1)}{2\pi} + 0.5 & x_1 < 0
    \end{cases}\\
    & f_1 = 10(x_3 - 10\theta)\\
    & f_2 = 10(\sqrt{x_1^2 + x_2^2} - 1)
\end{aligned}
$$"""
        # Define the citation
        new_prob.cite = "J. J. More, B. S. Garbow and K. E. Hillstrom, “Testing Unconstrained Optimization Software,” ACM Transac tions on Mathematical Software, Vol. 7, No. 1, 1981, pp. 19-31. doi:10.1145/355934.355936"
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list"""
        return [-1.0, 0.0, 0.0]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the HelicalValley function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        theta = (1.0 / (2.0 * np.pi)) * np.arctan2(sites.x2, sites.x1)
        theta[sites.x1 < 0.0] += 0.5
        f_1 = 10.0 * (sites.x3 - 10.0 * theta)
        f_2 = 10.0 * (np.sqrt(sites.x1 * sites.x1 + sites.x2 * sites.x2) - 1.0)
        sites["f"] = f_1 * f_1 + f_2 * f_2 + sites.x3 * sites.x3
