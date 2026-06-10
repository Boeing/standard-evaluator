"""A Python module to provide access to ForresterMultiFiLo optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test.forrester_multi_fi_base import (
    ForresterMultiFiBase,
)


class ForresterMultiFiLo(ForresterMultiFiBase):
    """Lo-fidelity class for a forrester multifidelity problem. User can set values for A,B & C
    while defining the problem that affects the low-fidelity evaluator value
    as the values of A, B & C are then multiplied with high-fidelity evaluator value :
    A*f_high(x) + B*(x-0.5)**2 + C
    End user can modify values of A, B & C by passing them as functional argumnets while
    instantiating ForresterMultiFiLo()
    e.g. ForresterMultiFiLo(A = 0.6, B = 3.0, C = -1.2)
    """

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the forrester multifidelity low function.
        
        Parameters
        ----------
        sites : pd.DataFrame
            The dataframe that contains the input values, and is updated with the responses.
        """
        self._baseline(sites)
        parameter_a = self.lookup_option_value("A")
        parameter_b = self.lookup_option_value("B")
        parameter_c = self.lookup_option_value("C")
        sites["f"] = (
            parameter_a * sites.f
            + parameter_b * (sites.x - 0.5) ** 2
            + parameter_c
        )
