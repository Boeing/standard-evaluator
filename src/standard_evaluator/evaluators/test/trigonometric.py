"""A Python module to provide access to TrigonometricTF optimization test problems"""

# pylint: disable=W0223

import pandas as pd
import numpy as np
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class Trigonometric(TestEvaluator):
    r"""Trigonometric function
    "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"
    by J.E. Dennis Jr. and R.B. Shnabel.

    minimize f where

    .. math::
        f = f_1^2 + f_2^2

    .. math::
        f_1 = 1 - [\cos(x_1) + 2(1 - \cos(x_1)) - \sin(x_1)]
            - [\cos(x_2) + 2(1 - \cos(x_1)) - \sin(x_1)]

    .. math::
        f_2 = 1 - [\cos(x_1) + 2(1 - \cos(x_2)) - \sin(x_2)]
            - [\cos(x_2) + 2(1 - \cos(x_2)) - \sin(x_2)]
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for minimizing a trigonometric function.

        This method initializes a new optimization problem with two independent variables
        and one dependent variable. It sets default values for the variables, names them,
        and defines the objective function and constraints. The objective function is
        based on a trigonometric formulation as described in the reference.

        The optimization problem is defined as follows:

        - **Objective Function**: Minimize \( f \) where
        $$ f = f_1^2 + f_2^2 $$

        with
        $$ f_1 = 1 - [\cos(x_1) + 2(1 - \cos(x_1)) - \sin(x_1)] - [\cos(x_2) + 2(1 - \cos(x_1)) - \sin(x_1)] $$

        and
        $$ f_2 = 1 - [\cos(x_1) + 2(1 - \cos(x_2)) - \sin(x_2)] - [\cos(x_2) + 2(1 - \cos(x_2)) - \sin(x_2)] $$

        - **Variables**:
        - \( x_1 \): First independent variable
        - \( x_2 \): Second independent variable

        - **Responses**:
        - The response is named \( f \).

        - **Constraints**:
        - There are no constraints defined for this optimization problem.

        - **Description**:
        The problem is based on the work "Numerical Methods for Unconstrained Optimization and Nonlinear Equations" by J.E. Dennis Jr. and R.B. Shnabel.

        - **Citation**:
        J.E. Dennis Jr. and R.B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations'.

        Returns:
            OptProblem: An instance of the optimization problem configured with the defined objectives, variables, and description.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=1, name="trigonometric"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = ["x1", "x2"]
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
        new_prob.description = r"""Trigonometric function
    "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"
    by J.E. Dennis Jr. and R.B. Schnabel.

    minimize f where

    .. math::
        f = f_1^2 + f_2^2

    .. math::
        f_1 = 1 - [\cos(x_1) + 2(1 - \cos(x_1)) - \sin(x_1)]
            - [\cos(x_2) + 2(1 - \cos(x_1)) - \sin(x_1)]

    .. math::
        f_2 = 1 - [\cos(x_1) + 2(1 - \cos(x_2)) - \sin(x_2)]
            - [\cos(x_2) + 2(1 - \cos(x_2)) - \sin(x_2)]"""
        # Define the citation
        new_prob.cite = "J.E. Dennis Jr. and R.B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations'"
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [0.5, 0.5]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the Trigonometric function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        f_1 = (
            1.0
            - (np.cos(sites.x1) + 2.0 * (1.0 - np.cos(sites.x1)) - np.sin(sites.x1))
            - (np.cos(sites.x2) + 2.0 * (1.0 - np.cos(sites.x1)) - np.sin(sites.x1))
        )
        f_2 = (
            1.0
            - (np.cos(sites.x1) + 2.0 * (1.0 - np.cos(sites.x2)) - np.sin(sites.x2))
            - (np.cos(sites.x2) + 2.0 * (1.0 - np.cos(sites.x2)) - np.sin(sites.x2))
        )
        sites["f"] = f_1 * f_1 + f_2 * f_2
