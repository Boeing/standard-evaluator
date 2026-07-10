"""A Python module to define the base class for SimpleMultiFi optimization test problem classes"""

# pylint: disable=W0223

import math
from abc import abstractmethod
import pandas as pd
import numpy as np
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class SimpleMultiFiBase(TestEvaluator):
    """Source code for a simple multifidelity problem capturing the common information."""

    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """
        Creates a simple multifidelity optimization problem.

        This method initializes an optimization problem with two independent variables and three dependent responses.
        It sets default values, bounds, and names for the variables and responses. The objective function and constraints
        are defined based on the specified relationships.

        The optimization problem is defined as follows:

        - **Objective**: Minimize \( f(x, y) \)
        - **Subject to**:
            - \( c_1 \) and \( c_2 \) as constraints
        - **Variable Bounds**:
            - \( 0.0 \leq x, y \leq \pi \)
        - **Response Bounds**:
            - \( -10.0 \leq c_1 \leq 0.0 \)
            - \( -3.0 \leq c_2 \leq 1.0 \)
            - \( c_f \) is unbounded

        The initial guesses for the variables are set using the `_def_initial_guess` method.

        Args:
            **kwargs: Additional keyword arguments that may be used for further customization (not currently implemented).

        Returns:
            OptProblem: An instance of the OptProblem class containing the defined optimization problem.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=3, name="simple_multi_fi_base"
        )
        # Define default values
        defaults = self._def_initial_guess()
        var_bounds = ([0.0, math.pi], [0.0, math.pi])
        var_names = ["x", "y"]

        # Define the default values and bounds on the variables
        for var, local_default, local_bounds, name in zip(
            new_prob.variables, defaults, var_bounds, var_names
        ):
            var.bounds = local_bounds
            var.default = local_default
            var.name = name

        resp_bounds = ([-10.0, 0.0], [-3.0, 1.0], [-np.inf, np.inf])
        resp_names = ["c1", "c2", "f"]
        for resp, local_resp_bounds, name in zip(
            new_prob.responses, resp_bounds, resp_names
        ):
            resp.bounds = local_resp_bounds
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["c1", "c2"]

        # Define th description of the problem
        new_prob.description = """Source code for a simple multifidelity problem capturing the common information.
author: Mark Abramson (Joe Simonis invented the test problem)
date Jun 12, 2015"""
        # Define the citation
        new_prob.cite = ""

        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        (math.pi/2, math.pi/2)
        """
        return [math.pi / 2.0, math.pi / 2.0]

    @abstractmethod
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the OptlibTest function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
