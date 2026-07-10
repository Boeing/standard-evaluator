from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel

import standard_evaluator as se
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class HyperbolicTangentTensorProductOptions(BaseModel):
    """Pydantic model defining options for the HyperbolicTangentTensorProduct evaluator."""

    parameter_a: float = 1.0


class HyperbolicTangentTensorProduct(TestEvaluator):
    r"""The hyperbolic tangent tensor product function approximates a step function
    which causes oscillation with some surrogate models. It is given by:

    .. math::
        f(\mathbf{x}) = \prod_{i=1}^n \tanh(a \cdot x_i)

    with :math:`-1 \leq x_i \leq 1` for :math:`i = 1,\ldots,n`.
    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this evaluator.

        Returns
        -------
        Type[BaseModel]
            The HyperbolicTangentTensorProductOptions class.
        """
        return HyperbolicTangentTensorProductOptions

    def __init__(
            self,
            num_independent: int = 1,
            num_dependent: int = 1,
            name: str = None,
            comp_cost: float = 100,
            **kwargs
        ) -> None:
        """
        Constructor method for the hyperbolic tangent tensor product function.

        Parameters
        ----------
        num_independent : int, optional
            Number of independent variables. Must be >= 1. Defaults to 1.
        num_dependent : int, optional
            Number of dependent variables. Must be >= 1. Defaults to 1.
        name : str, optional
            Name to give the evaluator.
        comp_cost : float, optional
            Computational cost of evaluating. Defaults to 100.
        **kwargs : dict
            Additional keyword arguments, including ``options`` for a
            ``HyperbolicTangentTensorProductOptions`` instance.
        """
        self._num_independent = max(1, num_independent)
        self._num_dependent = max(1, num_dependent)
        super().__init__(
            name=name,
            comp_cost=comp_cost,
            **kwargs
        )

    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """
        Creates an optimization problem for the hyperbolic tangent tensor product test function.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments (unused; dimensions come from __init__).

        Returns
        -------
        OptProblem
            An instance of the OptProblem class representing the constructed
            optimization problem.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=self._num_independent,
            num_dependent=self._num_dependent,
            name="hyperbolic_tanget_tensor_product"
        )

        # Define default values
        defaults = self._def_initial_guess()

        # Define the default values and bounds on the variables
        for var, local_default in zip(new_prob.variables, defaults):
            var.bounds = [-1.0, 1.0]
            var.default = local_default

        for resp in new_prob.responses:
            resp.name = "f"

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = []

        # Define the description of the problem
        new_prob.description = r"""The hyperbolic tangent tensor product function approximates a step function which causes oscillation with some surrogate models. It is given by:

$$
f(\mathbf{x}) = \prod_{i=1}^n \tanh(a \cdot x_i)
$$

| -1 <= x_i <= 1 for i = 1,...,n."""

        # Define the citation
        new_prob.cite = "Mohamed Amine Bouhlel, John T. Hwang, Nathalie Bartoli, Rémi Lafage, Joseph Morlier and Joaquim R. R. A. Martins, A Python surrogate modeling framework with derivatives, Advances in Engineering Software (2019)."

        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns
        -------
        list
            List providing the initial guess to use with an optimizer.
        """
        return [0.0] * self._num_independent

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the hyperbolic tangent function.

        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe that contains the input values, and is updated with the responses.
        """
        n = self._num_independent
        a = self.lookup_option_value("parameter_a")
        sites["f"] = 1.0
        for ind in range(n):
            sites["f"] = sites["f"] * np.tanh(a * sites[f"x{ind}"])
