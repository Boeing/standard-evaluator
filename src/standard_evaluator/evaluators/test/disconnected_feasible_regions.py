from typing import Tuple, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem


class DisconnectedFeasibleRegionsOptions(BaseModel):
    """Pydantic model defining options for the DisconnectedFeasibleRegions evaluator."""

    centers: list[Tuple[float, float]] = [(-0.5, -0.4), (0.4, 0.3), (-0.2, 0.6)]
    radii: list[float] = [0.15, 0.12, 0.10]
    smooth: bool = False
    tau: float = 0.01

    @field_validator("tau")
    @classmethod
    def tau_must_be_positive(cls, v: float) -> float:
        """Validate that tau is greater than 0."""
        if v <= 0:
            raise ValueError("tau must be greater than 0")
        return v

    @model_validator(mode="after")
    def centers_radii_same_length(self) -> "DisconnectedFeasibleRegionsOptions":
        """Validate that centers and radii have the same length."""
        if len(self.centers) != len(self.radii):
            raise ValueError(
                f"centers and radii must have the same length, "
                f"got {len(self.centers)} centers and {len(self.radii)} radii"
            )
        return self


class DisconnectedFeasibleRegions(TestEvaluator):
    """A feasibility-only evaluator with multiple disconnected circular feasible islands.

    Supports both a nonsmooth (exact min) and smooth (soft-min via log-sum-exp)
    formulation.

    Nonsmooth (smooth=False):
        g1 = min over ℓ of [(x1 - a_ℓ)² + (x2 - b_ℓ)² - r_ℓ²]

    Smooth (smooth=True):
        d_ℓ = (x1 - a_ℓ)² + (x2 - b_ℓ)² - r_ℓ²
        g1 = -τ · log(Σ_ℓ exp(-d_ℓ / τ))

    A point is feasible when g1 <= 0.
    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this evaluator.

        Returns
        -------
        Type[BaseModel]
            The DisconnectedFeasibleRegionsOptions class.
        """
        return DisconnectedFeasibleRegionsOptions

    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """Create an optimization problem for disconnected feasible regions.

        Defines 2 variables x1, x2 in [-1, 1] and 1 response g1 with bounds
        (-inf, 0]. This is a feasibility-only problem (no objectives).

        Returns
        -------
        OptProblem
            An instance of the OptProblem class representing the problem.
        """
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=1, name="disconnected_feasible_regions"
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
            "A feasibility-only problem with multiple disconnected circular "
            "feasible islands. Supports both nonsmooth (exact min) and smooth "
            "(soft-min via log-sum-exp) formulations."
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
        """Evaluate the disconnected feasible regions constraint for each site.

        When smooth=False, computes:
            g1 = min over ℓ of [(x1 - a_ℓ)² + (x2 - b_ℓ)² - r_ℓ²]

        When smooth=True, computes:
            d_ℓ = (x1 - a_ℓ)² + (x2 - b_ℓ)² - r_ℓ²
            g1 = -τ · log(Σ_ℓ exp(-d_ℓ / τ))

        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe containing input values, updated with response g1.
        """
        centers = self.lookup_option_value("centers")
        radii = self.lookup_option_value("radii")
        smooth = self.lookup_option_value("smooth")
        tau = self.lookup_option_value("tau")

        x1 = sites["x1"].values
        x2 = sites["x2"].values

        # Compute d_ℓ for each island: shape (num_islands, num_sites)
        d = np.array(
            [
                (x1 - a) ** 2 + (x2 - b) ** 2 - r**2
                for (a, b), r in zip(centers, radii)
            ]
        )

        if not smooth:
            # Nonsmooth: g1 = min over islands
            g1 = np.min(d, axis=0)
        else:
            # Smooth: g1 = -tau * log(sum_ℓ exp(-d_ℓ / tau))
            # Use log-sum-exp trick for numerical stability
            shifted = -d / tau  # shape (num_islands, num_sites)
            max_val = np.max(shifted, axis=0)  # shape (num_sites,)
            g1 = -tau * (max_val + np.log(np.sum(np.exp(shifted - max_val), axis=0)))

        sites["g1"] = g1
