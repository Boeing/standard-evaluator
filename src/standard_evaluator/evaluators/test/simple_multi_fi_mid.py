"""A Python module to provide access to SimpleMultiFi_Mid optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test.simple_multi_fi_base import (
    SimpleMultiFiBase,
)


class SimpleMultiFiMid(SimpleMultiFiBase):
    """Source code for a simple multifidelity problem, which was invented by Joe Simonis
    Mid-fidelity class for a simple multifidelity problem
    This class represents the mid-fidelity version of a simple multifidelity optimization problem
    with two independent variables and three dependent responses. The mid-fidelity model provides
    a balance between computational cost and accuracy, serving as an intermediate approximation
    between the low- and high-fidelity models.

    Problem Definition:
    -------------------
    Variables:
        - x: Independent variable bounded between 0 and π (inherited from base)
        - y: Independent variable bounded between 0 and π (inherited from base)

    Responses:
        - f: Objective function to be minimized, approximated as
             \( f(x, y) = -\sin(x) \sin(y) \) using Taylor series expansions of sine functions
        - c1: Constraint 1, approximated as
              \( c_1(x, y) = e^x y - 5 \leq 0 \) using a truncated Taylor series expansion for \( e^x \)
        - c2: Constraint 2, approximated as
              \( c_2(x, y) = \sin(x) \cos(y) - 1 \leq 0 \) using Taylor series expansions for sine and cosine

    Objective:
        Minimize the function \( f(x, y) \).

    Constraints:
        - \( c_1(x, y) \leq 0 \)
        - \( c_2(x, y) \leq 0 \)

    Bounds:
        - \( 0 \leq x \leq \pi \)
        - \( 0 \leq y \leq \pi \)

    Notes:
    ------
    - The mid-fidelity model uses Taylor series expansions to approximate nonlinear functions,
      reducing computational complexity compared to the high-fidelity model.
    - This model is intended to provide a compromise between accuracy and computational cost,
      useful in multifidelity optimization frameworks.
    - The known optimal solution is approximately (x, y) = (1.33475, 1.38051).

    References:
    -----------
    - Problem inspired by Joe Simonis and adapted by Mark Abramson.
    - Date of creation: June 12, 2015.

    """

    # ToDo: Implement gradient information pylint: disable=fixme
    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.
        :return: DataFrame providing the optimal solution of the problem
        :rtype: list
        """
        return [1.3347504, 1.3805090]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the SimpleMultiFi_Mid function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sinx = (
            sites.x
            - sites.x * sites.x * sites.x / 6.0
            + sites.x * sites.x * sites.x * sites.x * sites.x / 120.0
        )
        siny = (
            sites.y
            - sites.y * sites.y * sites.y / 6.0
            + sites.y * sites.y * sites.y * sites.y * sites.y / 120.0
        )
        expx = 1 + sites.x + sites.x * sites.x / 2.0 + sites.x * sites.x * sites.x / 6.0
        cosy = (
            1 - sites.y * sites.y / 2.0 + sites.y * sites.y * sites.y * sites.y / 24.0
        )
        sites["f"] = -sinx * siny
        sites["c1"] = expx * sites.y - 5.0
        sites["c2"] = sinx * cosy - 1.0
