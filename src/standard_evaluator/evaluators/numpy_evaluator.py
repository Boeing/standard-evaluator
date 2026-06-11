""" " Defining the Numpy Evaluator"""

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.utilities import unroll_data_frame_numpy
from standard_evaluator.utilities import unroll_names_using_variables
from standard_evaluator.utilities import roll_data_frame_using_variables


class NumpyEvaluator(Evaluator):
    """An abstract class to allow us to define evaluators that implement the
    calling sequence with numpy
    """

    def _evaluate(self, sites: pd.DataFrame, **kwargs) -> None:
        """Evaluate the function using numpy arrays.

        Args:
            sites: The dataframe containing input values, updated with responses.
            **kwargs: Additional keyword arguments.
        """
        # Get the variable values from the dataframe and call the numpy
        # evaluation routine
        unrolled_results_np = self.eval_np(
            self.dataframe_to_float_ndarray(sites[[var.name for var in self.interface.inputs]]), **kwargs
        )

        # The data has been unrolled and in a np.array. We need to put into 
        # DataFrame with unrolled names
        unrolled_names = unroll_names_using_variables(self.interface.outputs)
        unrolled_results_df = pd.DataFrame(unrolled_results_np, columns=unrolled_names)
        rolled_results_df = roll_data_frame_using_variables(unrolled_results_df, self.interface.outputs)
        # Store the results in the dataframe
        if "names" in kwargs:
            if kwargs["names"] is not None:
                sites[kwargs["names"]] = rolled_results_df
            else:
                sites[self.outputs] = rolled_results_df
        else:
            sites[self.outputs] = rolled_results_df

    @abstractmethod
    def eval_np(
        self, sites: NDArray[np.float64], names: list = None, **kwargs
    ) -> NDArray[np.float64]:
        """Evaluate the function with a numpy array.

        Args:
            sites: The sites to be evaluated.
            names: Which responses are computed. Defaults to None (all).
            **kwargs: Additional keyword arguments.

        Returns:
            NDArray[np.float64]: 2D array with response values for all sites.
        """

    def dataframe_to_float_ndarray(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """Convert a DataFrame to a float ndarray.

        Args:
            df: The DataFrame to be converted.

        Returns:
            NDArray[np.float64]: Array representing the DataFrame values as floats.
        """
        return unroll_data_frame_numpy(df)
