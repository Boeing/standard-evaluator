"""A module to define the base class for ForresterMultiFi optimization test problem classes"""

# pylint: disable=W0223

from abc import abstractmethod
from typing import Type
import numpy as np
import pandas as pd
from pydantic import BaseModel
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class ForresterMultiFiOptions(BaseModel):
    """Pydantic model defining options for Forrester multi-fidelity evaluators."""
    
    A: float = 0.2  # Parameter A for the optimization problem
    B: float = 5.0  # Parameter B for the optimization problem  
    C: float = -2.0  # Parameter C for the optimization problem


class ForresterMultiFiBase(TestEvaluator):
    """Multifidelity analysis occurs when different analysis or simulation codes predict the same
    response. Often in engineering different analysis codes are distinguished by their
    computational complexity and accuracy, for example, a quick calculation may be done using
    empirical equations while an expensive calculation may be cone using finite element analysis.
    Mathematically, we have a greater quanitity of low-fidelity data  𝑋𝑙,𝑦𝑙  and a more accurate
    but lower quantity of high-fidelity data  𝑋ℎ,𝑦ℎ . We would like to build a model that
    leverages all collected data.

    One method for accomplishing is to construct a correction model of the form

    𝑦ℎ=𝑍𝜌𝑦𝑙+𝑍𝑑"""

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this evaluator.
        
        Returns
        -------
        Type[BaseModel]
            The ForresterMultiFiOptions class defining A, B, and C parameters.
        """
        return ForresterMultiFiOptions

    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """
        Create an optimization problem for multifidelity analysis.

        This method constructs a basic optimization problem that leverages
        multifidelity data to predict responses. It initializes the problem
        with one independent variable and one dependent response, sets
        default values and bounds for the variables, and defines the
        objectives and constraints.

        The problem description includes a mathematical formulation of
        multifidelity analysis, highlighting the relationship between
        low-fidelity and high-fidelity data.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments (unused in this implementation).

        Returns
        -------
        OptProblem
            An instance of the OptProblem class representing the constructed
            optimization problem.

        Notes
        -----
        The optimization problem is defined with the following characteristics:
        - One independent variable named "x" with bounds [0.0, 1.0].
        - One dependent response named "f".
        - No constraints are defined for this problem.
        - The problem description includes a mathematical representation of
          multifidelity analysis, which is formatted in LaTeX.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=1, num_dependent=1, name="forrester_multi_fi_base"
        )
        # Define default values
        defaults = self._def_initial_guess()
        var_names = ["x"]

        # Define the default values and bounds on the variables
        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.bounds = [0.0, 1.0]
            var.default = local_default
            var.name = name

        for resp in new_prob.responses:
            resp.name = "f"

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = []

        # Define the description of the problem
        new_prob.description = r"""$$
Multifidelity analysis occurs when different analysis or simulation codes predict the same
response. Often in engineering different analysis codes are distinguished by their
computational complexity and accuracy, for example, a quick calculation may be done using
empirical equations while an expensive calculation may be cone using finite element analysis.
Mathematically, we have a greater quanitity of low-fidelity data  𝑋𝑙,𝑦𝑙  and a more accurate
but lower quantity of high-fidelity data  𝑋ℎ,𝑦ℎ . We would like to build a model that
leverages all collected data.

One method for accomplishing is to construct a correction model of the form

𝑦ℎ=𝑍𝜌𝑦𝑙+𝑍𝑑
$$"""
        # Define the citation
        new_prob.cite = ""
        
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.
        
        Returns
        -------
        list
            List providing the initial guess to use with an optimizer.
        """
        return [0.75936838]

    @abstractmethod
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the Forrester function.
        
        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe that contains the input values, and is updated with the responses.
        """

    def _baseline(self, sites: pd.DataFrame) -> None:
        """Baseline function used for both low & high fidelity implementation.
        
        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe that contains the input values, and is updated with the responses.
        """
        sites["f"] = (6.0 * sites.x - 2.0) ** 2 * np.sin(12.0 * sites.x - 4.0)
