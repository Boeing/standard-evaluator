import pandas as pd

import standard_evaluator as se

from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.evaluators.abstract_evaluator import ValidInputs


class Rosenbrock(TestEvaluator):
    """
    An example test that represents the Rosenbrock function.
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
        Constructor method for the Rosenbrock function.
        """

        # Validate inputs

        inputs = ValidInputs(
            num_independent=num_independent,
            num_dependent=num_dependent
        )
        if inputs.num_independent == 1:
            inputs.num_independent = 2

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
        Creates an optimization problem for the Rosenbrock test function.        

        Parameters:
        -----------
        **kwargs : dict
            Optional keyword arguments to set parameters for the test evaluator.
            - "num_independent" (int): The number of independent variables. Default is 2.

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

        # Create the basic problem

        new_prob = se.utilities.create_opt_problem(
            num_independent=self._num_independent,
            num_dependent=self._num_dependent,
            name="rosenbrock" 
        )

        # Define default values

        defaults = self._def_initial_guess()

        # Define the default values and bounds on the variables
        
        for var, local_default in zip(new_prob.variables, defaults):
            var.bounds = [-2.0, 2.0]
            var.default = local_default

        for resp in new_prob.responses:
            resp.name = "f"

        # Define objectives and constraints

        new_prob.objectives = ["f"]
        new_prob.constraints = []

        # Define the description of the problem

        new_prob.description = \
        """
        The Rosenbrock function is a continuous, nonlinear, and non-convex function used in optimization.
        It is given by
                        sum_{i=1}^{n-1} ((x_{i+1}-x_i^2)^2 + (x_i-1)^2)
                        -2 <= x_i <= 2 for i = 1,...,n.

        """

        # Define the citation

        new_prob.cite = \
        """
        Mohamed Amine Bouhlel, John T. Hwang, Nathalie Bartoli, Rémi Lafage, Joseph 
        Morlier and Joaquim R. R. A. Martins, A Python surrogate modeling 
        framework with derivatives, Advances in Engineering Software (2019).
        """

        return new_prob
    
    def _def_initial_guess(self) -> list:
        """
        Provide an initial guess for an optimizer

        Returns:
            list: List providing the initial guess to use with an optimizer
        """

        return [1.0] * self._num_independent
    
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """
        Call to the Rosenbrock function

        Args:
            sites (pd.DataFrame): The dataframe that contains the input values, and is updated with the responses
        """

        sites["f"] = sum([
            (sites[f"x{ind+1}"] - sites[f"x{ind}"]**2)**2 + (sites[f"x{ind}"] - 1.0)**2 \
                for ind in range(self._num_independent-1)
        ])
    
