import numpy as np
import pandas as pd

import standard_evaluator as se

from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.evaluators.abstract_evaluator import ValidInputs


class ExponentialTensorProduct(TestEvaluator):
    r"""The exponential tensor product function approximates a step function which
    causes oscillation with some surrogate models. It is given by:

    .. math::
        f(\mathbf{x}) = \prod_{i=1}^n \exp(a \cdot x_i)

    with :math:`-1 \leq x_i \leq 1` for :math:`i = 1,\ldots,n`.
    """


    def __init__(
            self,
            name: str = None,
            comp_cost: float = 100,
            num_independent: int = None,
            num_dependent: int = None,
            **kwargs
        ) -> None:
        """
        Constructor method for the exponential tensor product function.
        """

        # Validate inputs

        inputs = ValidInputs(
            num_independent=num_independent,
            num_dependent=num_dependent
        )

        # Initialize

        super().__init__(
            name=name,
            comp_cost=comp_cost,
            num_independent=inputs.num_independent,
            num_dependent=inputs.num_dependent,
            **kwargs
        )

    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """
        Creates an optimization problem for the exponential tensor product test function.        

        Parameters:
        -----------
        **kwargs : dict
            Optional keyword arguments to set parameters for the test evaluator.
            - "num_independent" (int): The number of independent variables. Default is 1.
            - "parameter_a" (float): The parameter to scale the exponential input by. Default is 1.0

        Returns:
        --------
        OptProblem
            An instance of the OptProblem class representing the constructed
            optimization problem.
        """

        # Pass along number of independents

        if "num_independent" in kwargs:
            self._num_independent = kwargs["num_independent"]
        else:
            raise ValueError(f'Expected "num_independent" in kwargs')
        if "num_dependent" in kwargs:
            self._num_dependent = kwargs["num_dependent"]
        else:
            raise ValueError(f'Expected "num_dependent" in kwargs')
        if "parameter_a" in kwargs:
            self.parameter_a = kwargs["parameter_a"]
        else:
            self.parameter_a = 1.0
        
        # Create the basic problem

        new_prob = se.utilities.create_opt_problem(
            num_independent=self._num_independent,
            num_dependent=self._num_dependent,
            name="exponential_tensor_product" 
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

        new_prob.description = r"""The exponential tensor product function approximates a step function which causes oscillation with some surrogate models. It is given by:

$$
f(\mathbf{x}) = \prod_{i=1}^n \exp(a \cdot x_i)
$$

| -1 <= x_i <= 1 for i = 1,...,n."""

        # Define the citation

        new_prob.cite = "Mohamed Amine Bouhlel, John T. Hwang, Nathalie Bartoli, Rémi Lafage, Joseph Morlier and Joaquim R. R. A. Martins, A Python surrogate modeling framework with derivatives, Advances in Engineering Software (2019)."

        return new_prob
    
    def _def_initial_guess(self) -> list:
        """
        Provide an initial guess for an optimizer

        Returns:
            list: List providing the initial guess to use with an optimizer
        """

        return [0.0] * self._num_independent
    
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """
        Call to the exponential function

        Args:
            sites (pd.DataFrame): The dataframe that contains the input values, and is updated with the responses
        """

        sites["f"] = 1.0
        for ind in range(self._num_independent):
            sites["f"] = sites["f"] * np.exp(self.parameter_a * sites[f"x{ind}"])
    
