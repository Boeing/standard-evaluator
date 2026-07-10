"""A Python module to define the base class for ExponentialMultiFiBase optimization test problem
classes"""

# pylint: disable=W0223

from abc import abstractmethod
import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class ExponentialMultiFiBase(TestEvaluator):
    """This function is a two-dimensional example which occurs several times in the
    literature on computer experiments."""

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates and initializes an optimization problem for a two-dimensional example.

        This method sets up a basic optimization problem with two independent variables
        and one dependent variable. It defines the variable bounds, default values,
        objectives, and constraints for the problem. The problem is described as a
        two-dimensional example that is frequently referenced in the literature on
        computer experiments.

        Returns:
            OptProblem: An instance of the OptProblem class containing the configured
            optimization problem, including objectives, constraints, variable bounds,
            and a description.

        Example:
            >>> problem = self._create_opt_problem()
            >>> print(problem.description)
            This function is a two-dimensional example which occurs several times in the
            literature on computer experiments.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=1, name="exponential_multi_fi_base"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(2)]
        # Define the default values and bounds on the variables
        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.bounds = (0.0, 1.0)
            var.default = local_default
            var.name = name

        resp_names = f"f"
        for resp in new_prob.responses:
            resp.name = resp_names

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = []

        # Define th description of the problem
        new_prob.description = """This function is a two-dimensional example which occurs several times in the
literature on computer experiments."""
        # Define the citation
        new_prob.cite = ""
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [0.6759, 0.2456]

    @abstractmethod
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the ExponentialMultiFi function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """

    def _baseline(self, sites: pd.DataFrame) -> None:
        """baseline func used for both low & high fidelity implementation
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        # Create a filter of rows that have x2 = 0
        exp_part_filter = sites.x2 == 0
        # Create a new series of right length
        exp = sites.x2.copy()
        # Sets the rows of exp vector to 1 where sites.x2 = 0
        exp.loc[exp_part_filter] = 1.0
        # Sets the rows of exp vector to calculated expression value where sites.x2 not equals to 0
        # Note that we have to get the sub vector of x2 where x2 is not zero
        exp.loc[~exp_part_filter] = 1.0 - np.exp(
            -1.0 / (2.0 * sites.x2[~exp_part_filter])
        )

        num_part = (
            2300.0 * sites.x1**3 + 1900.0 * sites.x1**2 + 2092.0 * sites.x1 + 60.0
        )
        den_part = 100.0 * sites.x1**3 + 500.0 * sites.x1**2 + 4.0 * sites.x1 + 20.0
        sites["f"] = exp * (num_part / den_part)
