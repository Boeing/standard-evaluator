"""A Python module to provide access to ExponentialMultiFiLo optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test.exponential_multi_fi_base import (
    ExponentialMultiFiBase,
)


class ExponentialMultiFiLo(ExponentialMultiFiBase):
    """Lo-fidelity class for a exponential multifidelity problem.
    upper_right = input_vec + np.array([0.05, 0.05])
    lower_right = input_vec + np.array([0.05, -0.05])
    lower_right[1] = np.max([0.0, lower_right[1]])
    upper_left = input_vec + np.array([-0.05, 0.05])
    lower_left = input_vec + np.array([-0.05, -0.05])
    lower_left[1] = np.max([0.0, lower_left[1]])
    # Average outputs
    return 0.25 * (exp_high(upper_right) + exp_high(lower_right) + exp_high(upper_left) + exp_high(lower_left))
    """

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the forrester multifidelity low function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        upper_right = sites + {"x1": 0.05, "x2": 0.05}
        self._baseline(upper_right)
        lower_right = sites + {"x1": 0.05, "x2": -0.05}
        lower_right.loc[lower_right["x2"] < 0, "x2"] = 0
        self._baseline(lower_right)
        upper_left = sites + {"x1": -0.05, "x2": 0.05}
        self._baseline(upper_left)
        lower_left = sites + {"x1": -0.05, "x2": -0.05}
        lower_left.loc[lower_left["x2"] < 0, "x2"] = 0
        self._baseline(lower_left)

        sites["f"] = 0.25 * (
            upper_right.f + lower_right.f + upper_left.f + lower_left.f
        )
