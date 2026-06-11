from typing import Callable

import pandas as pd


class site_logger:
    """Decorator that keeps track of evaluated sites in the order they were
    evaluated. Logged sites can be accessed using the :meth:`get_log` method.
    Note that this will not check for duplicate sites.
    """

    def __init__(self, func: Callable) -> None:
        """Initialize the site logger.

        Args:
            func: Function to log evaluated sites of.
        """
        self._func = func
        self._sites = pd.DataFrame()
        self._n_calls = 0

    def __call__(self, sites: pd.DataFrame, **kwargs) -> None:
        """Evaluate responses and log sites.

        Args:
            sites: Sites to be evaluated.
            **kwargs: Additional keyword arguments.
        """
        # Evaluate sites
        self._func(sites, **kwargs)

        # Increment number of evaluations and save current length for indexing
        self._n_calls += 1
        ind = len(self._sites)

        # Log sites that were just evaluated and store the call number
        self._sites = pd.concat((self._sites, sites), ignore_index=True)
        self._sites.loc[ind:, "call_num"] = self._n_calls

    def get_log(self) -> pd.DataFrame:
        """Get all the sites that have been logged.

        Returns:
            pd.DataFrame: Copy of the logged sites.
        """
        return self._sites.copy()
