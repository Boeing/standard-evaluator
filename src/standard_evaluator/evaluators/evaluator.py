"""
Created Aug. 24, 2022

@author Mikel Woo
"""

from typing import Callable, Optional, List

import pandas as pd

from standard_evaluator.evaluators.abstract_evaluator import Evaluator


class PyEvaluator(Evaluator):
    """Runs a python function to evaluate the true function values. The
    ``sites`` dataframe passed to ``__call__`` is expected be **modified in
    place**. Thus the evaluation function must place the results back into this
    dataframe.

    .. note::
        Your function should modify the passed dataframe directly not overwrite
        it! That is, you shouldn't do anything like below.

        .. code-block:: python

            # Assign any new values/objects to the passed in argument
            sites = ...

            # Pandas functions without an ``inplace`` parameter or setting ``inplace=False``
            # will not modify the dataframe, but create a new one
            sites = sites.join(response_df)

        Doing this will appear to Design Explorer that your evaluator has done
        nothing. You may receive an error saying that all responses are missing.
        Instead you can use any of the following options which modify the
        existing dataframe.

        .. code-block:: python

            # Create new column and assign values all at once
            sites['response_name'] = ...

            # Assign values to column one at a time or in batch
            # Will also create column if it doesn't exist
            sites.loc[ind, 'response_name'] = ...

        Either of those methods are highly flexible and will ensure that your
        true response values are captured.
    """

    def __init__(
        self,
        func: Callable[[pd.DataFrame], None],
        name: str = None,
        comp_cost: float = 100,
        **kwargs,
    ) -> None:
        """Initialize the PyEvaluator.

        Args:
            func: Function that performs the evaluation. Must modify the
                incoming sites DataFrame in place.
            name: Name for identifying evaluator.
            comp_cost: Cost of running this evaluator. Defaults to 100.
            **kwargs: Parameters sent to problem definition.
        """
        super().__init__(name=name, comp_cost=comp_cost, **kwargs)

        self._check_callable(func)
        self._func = func

    def _check_callable(self, func):
        # If a function is passed, ensure that it is callable
        if (not func is None) and (not callable(func)):
            raise TypeError(
                'Evaluator: "fun" must be a callable! Received a '
                + f"{type(func).__name__} instead."
            )

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call the provided function.

        Args:
            sites: Sites to get true response values of.
        """
        self._func(sites)
