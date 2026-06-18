from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel

import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem


class SmallCircleFeasibleRegionOptions(BaseModel):
    """Pydantic model defining options for the SmallCircleFeasibleRegion evaluator."""

    a: float = 0.5
    b: float = -0.3
    r: float = 0.15


class SmallCircleFeasibleRegion(TestEvaluator):
    """A feasibility-only evaluator with a single circular constraint.

    Points inside the circle centered at (a, b) with radius r are feasible.
    The constraint is defined as:

        g1 = (x1 - a)^2 + (x2 - b)^2 - r^2

    A point is feasible when g1 <= 0.
    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this evaluator.

        Returns
        -------
        Type[BaseModel]
            The SmallCircleFeasibleRegionOptions class.
        """
        return SmallCircleFeasibleRegionOptions

    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """Create an optimization problem for the small circle feasible region.

        Defines 2 variables x1, x2 in [-1, 1] and 1 response g1 with bounds
        (-inf, 0]. This is a feasibility-only problem (no objectives).

        Returns
        -------
        OptProblem
            An instance of the OptProblem class representing the problem.
        """
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=1, name="small_circle_feasible_region"
        )

        # Define variables
        var_names = ["x1", "x2"]
        defaults = self._def_initial_guess()

        for var, local_default, name in zip(new_prob.variables, defaults, var_names):
            var.name = name
            var.bounds = [-1.0, 1.0]
            var.default = local_default

        # Define responses
        resp_names = ["g1"]
        resp_bounds = [[-np.inf, 0.0]]

        for resp, local_bounds, name in zip(
            new_prob.responses, resp_bounds, resp_names
        ):
            resp.name = name
            resp.bounds = local_bounds

        # Define objectives and constraints
        new_prob.objectives = []
        new_prob.constraints = ["g1"]

        # Define the description of the problem
        new_prob.description = (
            "A feasibility-only problem with a single small circular feasible "
            "region. The constraint g1 = (x1-a)^2 + (x2-b)^2 - r^2 <= 0 "
            "defines a circle centered at (a, b) with radius r."
        )

        new_prob.cite = ""

        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns
        -------
        list
            The initial guess values [0.0, 0.0].
        """
        return [0.0, 0.0]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Evaluate the circular constraint for each site.

        Computes g1 = (x1 - a)^2 + (x2 - b)^2 - r^2 for each row.

        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe containing input values, updated with response g1.
        """
        a = self.lookup_option_value("a")
        b = self.lookup_option_value("b")
        r = self.lookup_option_value("r")

        sites["g1"] = (sites["x1"] - a) ** 2 + (sites["x2"] - b) ** 2 - r**2
