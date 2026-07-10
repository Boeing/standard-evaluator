"""A Python module to provide access to ExtendedRosenbrock optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class ExtendedRosenbrock(TestEvaluator):
    r"""The Extended Rosenbrock function.

     This evaluator is an example function from
     "Numerical Methods for Unconstrained Optimization and
     Nonlinear Equations" by J.E. Dennis Jr. and R.B. Schnabel.

    The problem has two independent variables, :math:`x_0` and :math:`x_1` and
    a single response:

    .. math::
        f(x_0, x_1) = \left[10*\left(x_1-x_0^2\right)\right]^2 + (1-x_0)^2
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem instance for the Extended Rosenbrock function.

        This method initializes a new optimization problem with two independent variables
        and one objective function. It sets default values for the variables, defines the
        objective and constraints, and provides a description of the problem.

        The Extended Rosenbrock function is a well-known test problem for optimization
        algorithms, characterized by its non-convex nature and a global minimum located
        within a narrow, curved valley.

        Returns:
            OptProblem: An instance of the OptProblem class containing the defined
            variables, objectives, constraints, and description for the Extended Rosenbrock
            optimization problem.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=1, name="extended_rosenbrock"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = ["x0", "x1"]
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
        new_prob.description = r"""The Extended Rosenbrock function.

This evaluator is an example function from
"Numerical Methods for Unconstrained Optimization and
Nonlinear Equations" by J.E. Dennis Jr. and R.B. Schnabel.

The problem has two independent variables, :math:`x_0` and :math:`x_1` and
a single response:

.. math::
    f(x_0, x_1) = \left[10*\left(x_1-x_0^2\right)\right]^2 + (1-x_0)^2"""
        # Define the citation
        new_prob.cite = "J. E. Dennis, Jr., Robert B. Schnabel, 'Numerical Methods for Unconstrained Optimization and Nonlinear Equations', Volume 16 of Classics in Applied Mathematics, 1996"
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns:
            list: The initial guess values.
        """
        return [-1.2, 1.0]

    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.

        Returns:
            list: The variable values of the optimal solution.

        Note:
            x* = (1.0, 1.0), f(x*) = 0.0
        """
        return [
            1.0,
            1.0,
        ]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the ExtendedRosenbrock function.

        Args:
            sites: The dataframe containing input values, updated with responses.
        """
        f_0 = 10.0 * (sites.x1 - sites.x0 * sites.x0)
        f_1 = 1.0 - sites.x0
        sites["f"] = f_0 * f_0 + f_1 * f_1
