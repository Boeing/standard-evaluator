"""A Python module to provide access to ExponentialMultiFiHi optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test.exponential_multi_fi_base import (
    ExponentialMultiFiBase,
)


class ExponentialMultiFiHi(ExponentialMultiFiBase):
    """Hi-fidelity class for a exponential multifidelity problem"""

    # ToDo: Implement gradient information pylint: disable=fixme
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the exponential multifidelity high function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        self._baseline(sites)
