"""A Python module to provide access to SimpleMultiFiLoTF optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.evaluators.test.simple_multi_fi_base import (
    SimpleMultiFiBase,
)


class SimpleMultiFiLo(SimpleMultiFiBase):
    """Source code for a simple multifidelity problem, invented by Joe Simonis, at lowest fidelity.
    Low-fidelity class for a simple multifidelity problem
    This class represents the low-fidelity version of a simple multifidelity optimization problem
    with two independent variables and three dependent responses. The low-fidelity model provides
    a simplified and computationally cheaper approximation of the high-fidelity problem, useful
    for multifidelity optimization strategies.

    Problem Definition:
    -------------------
    Variables:
        - x: Independent variable bounded between 0 and π (inherited from base)
        - y: Independent variable bounded between 0 and π (inherited from base)

    Responses:
        - f: Objective function to be minimized, defined as f(x, y) = -x * y
        - c1: Constraint 1, defined as c1(x, y) = (1 + x) * y - 5 ≤ 0
        - c2: Constraint 2, defined as c2(x) = x - 1 ≤ 0

    Objective:
        Minimize the function f(x, y).

    Constraints:
        - c1(x, y) ≤ 0
        - c2(x) ≤ 0

    Bounds:
        - 0 ≤ x ≤ π
        - 0 ≤ y ≤ π

    Notes:
    ------
    - The objective and constraints are linear or simpler compared to the high-fidelity model,
      making this model computationally cheaper but less accurate.
    - This low-fidelity model is intended to be used in conjunction with higher-fidelity models
      in multifidelity optimization frameworks.
    - The known optimal solution is approximately (x, y) = (2.0, 1.6667).

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
        return [2.0, 1.6666667]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the SimpleMultiFi_LoTF function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["f"] = -sites.x * sites.y
        sites["c1"] = (1.0 + sites.x) * sites.y - 5.0
        sites["c2"] = sites.x - 1.0
