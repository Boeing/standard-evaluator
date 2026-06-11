"""A Python module to provide access to hs118 optimization test problems"""

import numpy as np
import pandas as pd

from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class PowellSingularFunction(TestEvaluator):
    r"""The Extended Powell Singular function.

    This function comes from
    "Numerical Methods for Unconstrained Optimization and
    Nonlinear Equations" by J.E. Dennis Jr. and R.B. Shnabel.

    .. math::
        \begin{align*}
            \min \quad& f = f_1^2 + f_2^2 + f_3^2 + f_4^2\\[.75em]
            \text{where} \quad& f_1 = x_1 + 10x_2\\
            & f_2 = \sqrt{5}\cdot(x_3 - x_4)\\
            & f_3 = x_2 - 2x_3\\
            & f_4 = \sqrt{10}\cdot(x_1 - x_4)
        \end{align*}

    The initial guess provided for this problem is

    .. math::
        f(3, -1, 0, 1) = 95
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem instance for the Extended Powell Singular function.

        This method initializes an optimization problem with the following characteristics:
        - Number of independent variables: 4
        - Number of dependent variables: 1
        - Problem name: "powell_singular_function"

        The default initial guesses for the independent variables are set using the
        `_def_initial_guess` method. The variable names are assigned as `x1`, `x2`, `x3`, and `x4`.

        The objective function is defined as the sum of the squares of four functions:
        $$
        \min \quad f = f_1^2 + f_2^2 + f_3^2 + f_4^2
        $$
        where:
        - $$ f_1 = x_1 + 10x_2 $$
        - $$ f_2 = \sqrt{5} \cdot (x_3 - x_4) $$
        - $$ f_3 = x_2 - 2x_3 $$
        - $$ f_4 = \sqrt{10} \cdot (x_1 - x_4) $$

        The initial guess for the function evaluation is:
        $$
        f(3, -1, 0, 1) = 95
        $$

        The problem is cited from the following reference:
        "J.E. Dennis Jr. and R.B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations'".

        Returns:
            OptProblem: An instance of the optimization problem configured for the Extended Powell Singular function.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=4, num_dependent=1, name="powell_singular_function"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(4)]
        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.default = local_default
            var.name = name

        resp_names = f"f"
        for resp in new_prob.responses:
            resp.name = resp_names

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = []

        # Define th description of the problem
        new_prob.description = r"""$$
The Extended Powell Singular function.

    This function comes from
    "Numerical Methods for Unconstrained Optimization and
    Nonlinear Equations" by J.E. Dennis Jr. and R.B. Shnabel.

    .. math::
        \begin{align*}
            \min \quad& f = f_1^2 + f_2^2 + f_3^2 + f_4^2\\[.75em]
            \text{where} \quad& f_1 = x_1 + 10x_2\\
            & f_2 = \sqrt{5}\cdot(x_3 - x_4)\\
            & f_3 = x_2 - 2x_3\\
            & f_4 = \sqrt{10}\cdot(x_1 - x_4)
        \end{align*}

    The initial guess provided for this problem is

    .. math::
        f(3, -1, 0, 1) = 95
$$"""
        # Define the citation
        new_prob.cite = "J.E. Dennis Jr. and R.B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations'"
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [3.0, -1.0, 0.0, 1.0]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the PowellSingularFunction function

        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        # Create some temporary columns
        f_1 = sites.x1 + 10.0 * sites.x2
        f_2 = 5**0.5 * (sites.x3 - sites.x4)
        f_3 = sites.x2 - 2.0 * sites.x3
        f_4 = 10**0.5 * (sites.x1 - sites.x4)

        sites["f"] = f_1 * f_1 + f_2 * f_2 + f_3 * f_3 + f_4 * f_4
