"""Shift and scale utilities for normalizing optimization problem variables."""

from typing import Tuple, List
from copy import deepcopy

import numpy as np
import pandas as pd

from standard_evaluator.problem import OptProblem, Variable
from standard_evaluator.utilities.utility import (
    get_shift_scale_value,
    update_bounds_to_optimizer_space,
    update_bounds_to_design_space,
)


class ShiftAndScale:
    """Normalizes variable and response values to similar orders of magnitude.

    Used to get variable and response values into the same order of
    magnitude to yield better optimization results.
    """

    def __init__(self, opt_problem: OptProblem) -> None:
        """Initializes shift and scale values from the given optimization problem.

        Args:
            opt_problem: The optimization problem containing variable and response
                definitions with shift and scale information.

        Raises:
            ValueError: If no variables are defined in the problem.
        """
        # Get shift and scale values for variables and responses
        var_shift, var_scale = self._vars_to_series(opt_problem.variables)
        res_shift, res_scale = self._vars_to_series(opt_problem.responses)
        if len(var_scale) == 0:
            raise ValueError("No variables defined.")

        # Store values into series to make arithmetic easier
        self.shift = pd.concat((var_shift, res_shift))
        self.scale = pd.concat((var_scale, res_scale))
        self._all = list(self.shift.index)

    def design_to_optimizer_space(
        self, points: pd.DataFrame, cols: List[str] = None, suffix: str = None
    ) -> pd.DataFrame:
        """Shifts and scales specified columns from design space to optimizer space.

        Args:
            points: Data points containing the values to shift and scale.
            cols: List of columns to perform the rescale on. If None, all columns.
            suffix: Value to append at end of column names.

        Returns:
            A DataFrame containing the shifted and scaled values.
        """
        if cols is None:
            cols = self._all

        # Get requested columns and perform shift and scale
        ret = points.loc[:, points.columns.isin(cols)]
        ret += self.shift.loc[self.shift.index.isin(cols)]
        ret *= self.scale.loc[self.scale.index.isin(cols)]

        # Append suffix if given
        if suffix:
            ret.columns += suffix

        return ret

    def optimizer_to_design_space(
        self, points: pd.DataFrame, cols: List[str] = None
    ) -> pd.DataFrame:
        """Transforms points from optimizer space back to design space.

        Args:
            points: Data points in optimizer space to unshift and unscale.
            cols: List of columns to perform the transformation on. If None, all columns.

        Returns:
            A DataFrame containing the unshifted and unscaled columns.
        """
        if cols is None:
            cols = self._all
        # Get requested columns and perform shift and scale
        ret = points.loc[:, points.columns.isin(cols)]
        ret /= self.scale.loc[self.scale.index.isin(cols)]
        ret -= self.shift.loc[self.shift.index.isin(cols)]

        return ret

    def to_optimizer_problem(self, opt_problem: OptProblem) -> OptProblem:
        """Returns a transformed optimization problem in optimizer space.

        Applies shift and scale to all variable and response bounds and defaults,
        then resets shift/scale to 0/1 in the transformed problem.

        Args:
            opt_problem: The optimization problem in design space.

        Returns:
            A new OptProblem with variables and responses in optimizer space.
        """
        optimizer_problem = deepcopy(opt_problem)

        for var in optimizer_problem.variables:
            shift_val, scale_val = get_shift_scale_value(var)
            update_bounds_to_optimizer_space(var, shift_val, scale_val)

        for resp in optimizer_problem.responses:
            shift_val, scale_val = get_shift_scale_value(resp)
            update_bounds_to_optimizer_space(resp, shift_val, scale_val)

        return optimizer_problem

    def to_design_space_problem(self, opt_problem: OptProblem) -> OptProblem:
        """Returns the problem transformed back to design space.

        Reverses the shift and scale operations on an optimizer-space problem.

        Args:
            opt_problem: The problem in optimizer space.

        Returns:
            A new OptProblem in design space.
        """
        design_space_problem = deepcopy(opt_problem)

        for var in design_space_problem.variables:
            shift_val = self.shift[var.name]
            scale_val = self.scale[var.name]
            update_bounds_to_design_space(var, shift_val, scale_val)

        for resp in design_space_problem.responses:
            shift_val = self.shift[resp.name]
            scale_val = self.scale[resp.name]
            update_bounds_to_design_space(resp, shift_val, scale_val)

        return design_space_problem

    @staticmethod
    def _vars_to_series(var_list: List[Variable]) -> Tuple[pd.Series, pd.Series]:
        """Converts shift and scale values from a list of Variables to pandas Series.

        Args:
            var_list: List of Variable objects with shift and scale attributes.

        Returns:
            A tuple of (shift_series, scale_series).

        Raises:
            TypeError: If shift or scale values are not numeric.
            ValueError: If a scale value is zero.
        """
        shift = {}
        scale = {}
        for var in var_list:
            s, sc = get_shift_scale_value(var)
            shift[var.name] = s
            scale[var.name] = sc
        return pd.Series(shift), pd.Series(scale)

    def get_arrays(
        self, variables: List[str], responses: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns shift and scale as numpy arrays for the given variable/response names.

        Args:
            variables: List of variable names (inputs).
            responses: List of response names (outputs).

        Returns:
            A tuple of (input_shift, input_scale, output_shift, output_scale),
            each a 1-D numpy array.
        """
        input_shift = np.array([self.shift[n] for n in variables], dtype=np.float64)
        input_scale = np.array([self.scale[n] for n in variables], dtype=np.float64)
        output_shift = np.array([self.shift[n] for n in responses], dtype=np.float64)
        output_scale = np.array([self.scale[n] for n in responses], dtype=np.float64)
        return input_shift, input_scale, output_shift, output_scale
