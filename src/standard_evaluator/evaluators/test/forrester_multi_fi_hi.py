"""A Python module to provide access to ForresterMultiFi_Hi optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test.forrester_multi_fi_base import (
    ForresterMultiFiBase,
)


class ForresterMultiFiHi(ForresterMultiFiBase):
    """Hi-fidelity class for a forrester multifidelity problem"""

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the forrester multifidelity high function.
        
        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe that contains the input values, and is updated with the responses.
        """
        self._baseline(sites)
