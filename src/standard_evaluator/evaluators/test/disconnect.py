"""A Python module to provide access to Disconnect optimization test problems"""

# pylint: disable=W0223

import math
import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class Disconnect(TestEvaluator):
    """The SphereEvaluator class is a multiobjective test problem.  It models a hyperellipsoid in
    n-dimensional space, centered at an arbitrary point and extending to the coordinate
    planes on each axis, although an optional offset may be used as well.  The default bounds
    on the object are the interior and surface of the hyperellipsoid, hence a multiobjective
    problem would expect to find the minimum-facing surface of the object.

    The default problem seeks to minimize each coordinate.  This can be easily checked by
    asserting that a point x is <= center and that the constraint value is equal to 1 (within
    toleance).

    For example, a center of [2, 5, 1] with an offset of [1, 2, 3] would have minima at [1, 7, 4],
    [3, 2, 4], and [3, 7, 1].

    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Create a multiobjective optimization problem using the SphereEvaluator class.
        This method initializes a new optimization problem with the following characteristics:

        - The problem is defined in a 2-dimensional space with 2 variables.
        - The variables are bounded between a small positive value (1.0e-6) and the maximum value derived from the arcsine function.
        - Default values for the variables are set based on the initial guess defined in the `_def_initial_guess` method.
        - The responses of the problem are constrained to be less than or equal to 0.0 and 0.5, respectively.
        - The objectives of the problem are set to minimize the variables "x" and "y".
        - The constraints of the problem are defined as "h" and "k".
        - A detailed description of the problem is provided, explaining the nature of the SphereEvaluator class and its expected behavior in terms of finding minima.

        Returns:
            new_prob: An instance of the optimization problem configured with the specified objectives, constraints, bounds, and description.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(2, 2, "disconnect")

        var_names = ["x", "y"]

        # Define default values
        defaults = self._def_initial_guess()

        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.bounds = [1.0e-6, 2 * math.asin(1.0)]
            var.default = local_default
            var.name = name

        # Define the bounds on the constraints
        resp_bounds = ([-np.inf, 0], [-np.inf, 0.5])
        resp_names = ["h", "k"]
        for resp, local_bound, name in zip(new_prob.responses, resp_bounds, resp_names):
            resp.bounds = local_bound
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["x", "y"]
        new_prob.constraints = ["h", "k"]

        # Define th description of the problem
        new_prob.description = """\
The SphereEvaluator class is a multiobjective test problem.  It models a hyperellipsoid in
n-dimensional space, centered at an arbitrary point and extending to the coordinate
planes on each axis, although an optional offset may be used as well.  The default bounds
on the object are the interior and surface of the hyperellipsoid, hence a multiobjective
problem would expect to find the minimum-facing surface of the object.

The default problem seeks to minimize each coordinate.  This can be easily checked by
asserting that a point x is <= center and that the constraint value is equal to 1 (within
toleance).

For example, a center of [2, 5, 1] with an offset of [1, 2, 3] would have minima at [1, 7, 4],
[3, 2, 4], and [3, 7, 1]."""
        # Define the citation
        new_prob.cite = "David A. Van Veldhuizen, Multiobjective Evolutionary Algorithms: Classifications, Analyses and New Innovations, Dissertation, AFIT/DS/ENG/99-01. Appendix B (Problem Tanaka)"
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [2.0, 2.0]

    def _evaluate(self, sites: pd.DataFrame):
        """Call to the Disconnect function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["h"] = (
            -sites.x * sites.x
            - sites.y * sites.y
            + 1.0
            + 0.1 * np.cos(16 * np.arctan(sites.x / sites.y))
        )
        sites["k"] = (sites.x - 0.5) * (sites.x - 0.5) + (sites.y - 0.5) * (
            sites.y - 0.5
        )
