"""
Created Feb. 13, 2023

@author Joerg Gablonsky
"""

import numpy as np
import pandas as pd
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.utilities.shift_scale import ShiftAndScale
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator.converters import evaluator_info_to_opt_problem


class ShiftScaleEvaluator(Evaluator):
    """Evaluator that applies mapping between shifted / scaled
    space and the original space for another evaluator.
    """

    def __init__(self, evaluate: Evaluator, interface: EvaluatorInfo = None, opt_problem: OptProblem = None) -> None:
        if not isinstance(evaluate, Evaluator):
            raise TypeError(
                f"{type(self).__name__}: eval must be an Evaluator descendant, "
                f"Received a {type(evaluate).__name__} instead."
            )

        # Determines which problem representation to use
        if opt_problem is not None:
            # Use opt_problem directly
            final_opt_problem = opt_problem

        elif interface is not None:
            # Convert interface to opt_problem
            final_opt_problem = evaluator_info_to_opt_problem(interface)

        else:
            raise ValueError(
                "One of 'opt_problem' or 'interface' must be provided."
            )

        # Save a pointer to the evaluator. This way if the evaluator is updated
        # we continue to support this without the need to create a new instance
        self._eval = evaluate

        # Create a ShiftAndScale class based on the problem that is passed in
        self._shift_scale = ShiftAndScale(opt_problem=final_opt_problem)

        # Pass in the scaled optimization problem, not the original
        scaled_problem = self._shift_scale.to_optimizer_problem(final_opt_problem)

        # Model options in function signature
        super().__init__(opt_problem=scaled_problem, name = evaluate.name)

    def optimizer_to_design_space(self, sites: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame from optimizer space to design space.

        Args:
            sites: DataFrame with points in the optimizer space.

        Returns:
            pd.DataFrame: DataFrame with the points in the design space.
        """
        return self._shift_scale.optimizer_to_design_space(sites)

    def design_to_optimizer_space(self, sites: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame from design space to optimizer space.

        Args:
            sites: DataFrame with points in the design space.

        Returns:
            pd.DataFrame: DataFrame with the points in the optimizer space.
        """
        return self._shift_scale.design_to_optimizer_space(sites)

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Perform the response calculation. Modifies the dataframe in place.

        Args:
            sites: The dataframe containing input values to be modified.
        """
        # Unscale and unshift the variable values
        unscaled_sites = self._shift_scale.optimizer_to_design_space(sites, self.inputs)
        # Evaluate the original function
        self._eval(unscaled_sites)
        # Scale the responses and return those
        sites[self.outputs] = self._shift_scale.design_to_optimizer_space(
            unscaled_sites[self.outputs], self.outputs
        )

    def _def_initial_guess(self) -> pd.Series:
        """Provide an initial guess for an optimizer.

        Uses the initial guess from the wrapped evaluator, shifted and scaled.

        Returns:
            list: The initial guess values in optimizer space.
        """
        initial = self._eval.initial_guess()
        inputs = self._eval.inputs
        scaled_initial = self._shift_scale.design_to_optimizer_space(initial, inputs)
        return scaled_initial.values[0]

    def _shift_scale_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute the shifted and scaled jacobian.

        Args:
            x: The input sites in the shifted/scaled space.

        Returns:
            np.ndarray: The jacobian values with shift/scale accounted for.
        """

        input_shift, input_scale, _, output_scale = self._shift_scale.get_arrays(
            self.inputs, self.outputs
        )

        # Push sites back to design space

        design_space_x = x / input_scale - input_shift

        # Evaluate jacobian on design space x

        design_space_jac = self._eval.jacobian(design_space_x)

        # Account for scale on input and output

        # Note that the jacobian returns an array (n_sites x n_dep x n_ind) so the
        # scaling operations need to happen on the corresponding axes

        design_space_jac *= 1.0 / input_scale[np.newaxis, np.newaxis, :]
        design_space_jac *= output_scale[np.newaxis, :, np.newaxis]

        # Return the adjusted jacobian

        return design_space_jac

    def __getattr__(self, attr):

        if attr == "jacobian":

            # Check if wrapped evaluator has jacobian method, if so apply shift / scale to it

            if hasattr(self._eval, attr):
                return self._shift_scale_jacobian
            raise AttributeError(f"{type(self._eval)} has no attribute 'jacobian'")

        # Otherwise this class does not have the attribute (__getattr__ is last resort)

        raise AttributeError
