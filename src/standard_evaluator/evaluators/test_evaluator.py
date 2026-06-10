"""
Created Aug. 12, 2022

@author Mikel Woo
"""

import copy
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

import standard_evaluator.utilities as utils

from standard_evaluator.evaluators.abstract_evaluator import Evaluator
import standard_evaluator as se
from standard_evaluator.problem import OptProblem

class TestEvaluator(Evaluator):
    """Abstract test function

    This defines the abstract test function"""

    # Prevent pytest from discovering this class as a test
    __test__ = False

    def __init__(self, name: str = None, comp_cost: float = 100, **kwargs):
        """Initialize the problem and set the inputs and outputs

        :param name: Name to give the evaluator. Defaults to evaluator type
            name. If a value is provided, this will be appended to the default.
        :type name: str, optional
        :param comp_cost: Computational cost of evaluating with this evaluator.
            Defaults to 100.
        :type comp_cost: float
        :param kwargs: Arguments sent to :py:meth:`_create_opt_problem`
        :type kwargs: dict
        """
        # Get the optimization problem
        opt_problem = self._create_opt_problem(**kwargs)
        # Re-validate the optimization problem
        opt_problem = OptProblem(**opt_problem.model_dump())
        # Pass opt_problem to super().__init__
        super().__init__(name, comp_cost, opt_problem=opt_problem, **kwargs)

        # Save solution
        self._solution = self._calculate_known_solution()

    def _calculate_known_solution(self) -> pd.DataFrame:
        """Calculate the known solution of the problem. Provides the site(s)
         and output values for the problem.

        :return: DataFrame providing the optimal solution of the problem
        :rtype: pd.DataFrame
        """
        # Need to set solution. First get it's definition. Which may include
        # just inputs or inputs and outputs
        sol = self._def_known_solution()

        # Convert solution to DataFrame if necessary
        if sol is not None:
            if not isinstance(sol, pd.DataFrame):
                # Series have nice functions for converting to DataFrames
                if isinstance(sol, pd.Series):
                    sol = sol.to_frame().transpose()
                # Convert lists and numpy arrays. Also catch type errors
                else:
                    sol = utils.create_df_from_problem(
                        self._problem, data=sol, names=self.inputs
                    )

            # Make sure no inputs are missing
            if not set(self.inputs).issubset(sol.columns):
                raise ValueError(
                    f"{type(self).__name__} Not all "
                    + "inputs are defined in known solution! Missing: "
                    + ", ".join(set(self.inputs).difference(sol.columns))
                )

            # Compute outputs if not given
            if not set(self.outputs).issubset(sol.columns):
                self(sol)

        # Return solution
        return sol

    @property
    def test_info(self) -> dict:
        """
        Provides a dictionary of information on the test evaluator.

        :return: A dictionary containing data describing the evaluator.
        :rtype: dict
        """

        opt = self.opt_problem

        # Get problem type from objectives
        n_objs = len(opt.objectives)
        if n_objs == 0:
            test_goal = "feasibility"
        elif n_objs == 1:
            test_goal = "optimization"
        else:
            test_goal = "multiple_objective_optimization"

        # Get constraints
        response_map = {r.name: r for r in opt.responses}
        n_cons = len(opt.constraints)
        n_eq_cons = 0
        n_ineq_cons = 0
        for con_name in opt.constraints:
            con_bounds = response_map[con_name].bounds
            if con_bounds[0] == con_bounds[1]:
                n_eq_cons += 1
            else:
                n_ineq_cons += 1

        # Checking inputs
        n_vars = self.nind
        n_continuous = 0
        n_discrete = 0
        bounded_variables = True
        for var in opt.variables:
            if var.class_type == "float":
                n_continuous += 1
                if var.bounds[1] - var.bounds[0] > 1e8:
                    bounded_variables = False
            else:
                n_discrete += 1

        if n_continuous == n_vars:
            problem_type = "continuous"
        elif 0 < n_continuous < n_vars:
            problem_type = "mixed"
        else:
            problem_type = "discrete"

        return {
            "test_goal": test_goal,
            "problem_type": problem_type,
            "n_vars": n_vars,
            "n_continuous": n_continuous,
            "n_discrete": n_discrete,
            "n_constraints": n_cons,
            "n_equality_constraints": n_eq_cons,
            "n_inequality_constraints": n_ineq_cons,
            "bounded_variables": bounded_variables,
        }

    @property
    def known_solution(self) -> pd.DataFrame:
        """The known solution of the problem. Provides the site(s) and output
        values for the problem.

        :return: DataFrame providing the optimal solution of the problem
        :rtype: pd.DataFrame
        """
        # Return solution
        return self._solution

    def _def_known_solution(self) -> Union[list, np.ndarray, pd.Series, pd.DataFrame]:
        """Provide the input values of the known optimal solution. By default this is empty

        :return: List providing the input values of the optimal solution of the problem
        :rtype: list
        """
        return None
    
    @abstractmethod
    def _create_opt_problem(self, **kwargs) -> OptProblem:
        """
        Define and construct the optimization problem.

        This method should specify the problem's decision inputs, 
        outputs, objectives, and any constraints that apply.

        Subclasses must implement this method to return an instance of 
        `OptProblem` that fully describes the optimization problem.

        :param kwargs: Additional keyword arguments to customize problem creation.
        :return: An `OptProblem` instance representing the optimization problem.
        :rtype: OptProblem
        """      
