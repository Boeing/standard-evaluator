"""A Python module to provide access to SimpleMultiFi_Hi optimization test problems"""

# pylint: disable=W0223

import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test.simple_multi_fi_base import (
    SimpleMultiFiBase,
)


class SimpleMultiFiHi(SimpleMultiFiBase):
    """Hi-fidelity class for a simple multifidelity problem

    This class represents the high-fidelity version of a simple multifidelity optimization problem
    with two independent variables and three dependent responses. The problem is designed to test
    optimization algorithms in a multifidelity context, where the high-fidelity model provides
    more accurate but computationally expensive evaluations.

    Problem Definition:
    -------------------
    Variables:
        - x: Independent variable bounded between 0 and π
        - y: Independent variable bounded between 0 and π

    Responses:
        - f: Objective function to be minimized, defined as f(x, y) = -sin(x) * sin(y)
        - c1: Constraint 1, defined as c1(x, y) = exp(x) * y - 5 ≤ 0
        - c2: Constraint 2, defined as c2(x, y) = sin(x) * cos(y) - 1 ≤ 0

    Objective:
        Minimize the function f(x, y).

    Constraints:
        - c1(x, y) ≤ 0
        - c2(x, y) ≤ 0

    Bounds:
        - 0 ≤ x ≤ π
        - 0 ≤ y ≤ π

    Notes:
    ------
    - The objective function is nonlinear and smooth.
    - The constraints are nonlinear and define feasible regions in the design space.
    - This high-fidelity model is intended to be more accurate than lower-fidelity approximations,
      and thus is typically more computationally expensive to evaluate.
    - The known optimal solution is approximately (x, y) = (1.2963, 1.3677).

    References:
    -----------
    - Original problem inspired by Joe Simonis and adapted by Mark Abramson.
    - Date of creation: June 12, 2015.

    """

    # ToDo: Implement gradient information pylint: disable=fixme
    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.
        :return: DataFrame providing the optimal solution of the problem
        :rtype: list
        """
        return [1.2962851, 1.3677304]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the OptlibTest function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sinx = np.sin(sites.x)
        siny = np.sin(sites.y)
        cosy = np.cos(sites.y)
        expx = np.exp(sites.x)
        sites["f"] = -sinx * siny
        sites["c1"] = expx * sites.y - 5.0
        sites["c2"] = sinx * cosy - 1.0
