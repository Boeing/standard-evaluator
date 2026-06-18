"""Speed reducer (gear box) design optimization test problem."""

# pylint: disable=W0223

import numpy as np
import pandas as pd

from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class SpeedReducer(TestEvaluator):
    r"""Speed reducer (gear box) design optimization problem.

    Minimizes the weight of a speed reducer subject to 11 constraints on
    bending stress, surface stress, transverse deflection of shafts, and
    geometric restrictions.

    .. math::
        \min\quad f(\mathbf{x}) = 0.7854\,x_1 x_2^2 (3.3333\,x_3^2
            + 14.9334\,x_3 - 43.0934)
            - 1.508\,x_1 (x_6^2 + x_7^2)
            + 7.4777\,(x_6^3 + x_7^3)
            + 0.7854\,(x_4 x_6^2 + x_5 x_7^2)

    subject to 11 inequality constraints g_1, \ldots, g_{11} \leq 0.

    Variables and bounds:

    ======  ====================  =====  =====
    Var     Description           Lower  Upper
    ======  ====================  =====  =====
    x1      Face width            2.6    3.6
    x2      Module of teeth       0.7    0.8
    x3      Number of teeth       17     28
    x4      Length of shaft 1     7.3    8.3
    x5      Length of shaft 2     7.8    8.3
    x6      Diameter of shaft 1   2.9    3.9
    x7      Diameter of shaft 2   5.0    5.5
    ======  ====================  =====  =====

    Reference:
        Golinski, J., "An Adaptive Optimization System Applied to Machine
        Synthesis", Mechanism and Machine Theory, 8(4), pp. 419-436, 1973.
    """

    def _create_opt_problem(self) -> OptProblem:
        """Create the speed reducer optimization problem.

        Returns:
            OptProblem: Configured optimization problem with 7 variables,
                1 objective, and 11 constraints.
        """
        # Create the basic problem: 7 variables, 12 responses (f + g1-g11)
        new_prob = se.utilities.create_opt_problem(
            num_independent=7, num_dependent=12, name="speed_reducer"
        )

        # Variable names, bounds, and defaults (midpoints)
        var_names = [f"x{i + 1}" for i in range(7)]
        var_bounds = [
            [2.6, 3.6],
            [0.7, 0.8],
            [17.0, 28.0],
            [7.3, 8.3],
            [7.8, 8.3],
            [2.9, 3.9],
            [5.0, 5.5],
        ]
        defaults = self._def_initial_guess()

        for var, name, bounds, default in zip(
            new_prob.variables, var_names, var_bounds, defaults
        ):
            var.name = name
            var.bounds = bounds
            var.default = default

        # Response names and bounds
        resp_names = ["f"] + [f"g{i + 1}" for i in range(11)]
        resp_bounds = [[-np.inf, np.inf]] + [[-np.inf, 0.0]] * 11

        for resp, name, bounds in zip(new_prob.responses, resp_names, resp_bounds):
            resp.name = name
            resp.bounds = bounds

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = [f"g{i + 1}" for i in range(11)]

        # Description and citation
        new_prob.description = (
            "Speed reducer (gear box) weight minimization with 11 constraints "
            "on stress, deflection, and geometry."
        )
        new_prob.cite = (
            'Golinski, J., "An Adaptive Optimization System Applied to Machine '
            'Synthesis", Mechanism and Machine Theory, 8(4), pp. 419-436, 1973.'
        )
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide the midpoints of variable bounds as initial guess.

        Returns:
            list: Midpoint values for each variable.
        """
        return [
            (2.6 + 3.6) / 2.0,   # x1: 3.1
            (0.7 + 0.8) / 2.0,   # x2: 0.75
            (17.0 + 28.0) / 2.0, # x3: 22.5
            (7.3 + 8.3) / 2.0,   # x4: 7.8
            (7.8 + 8.3) / 2.0,   # x5: 8.05
            (2.9 + 3.9) / 2.0,   # x6: 3.4
            (5.0 + 5.5) / 2.0,   # x7: 5.25
        ]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Evaluate the speed reducer objective and constraints.

        Args:
            sites: DataFrame containing input variable columns (x1-x7),
                updated in-place with response columns (f, g1-g11).
        """
        x1 = sites["x1"]
        x2 = sites["x2"]
        x3 = sites["x3"]
        x4 = sites["x4"]
        x5 = sites["x5"]
        x6 = sites["x6"]
        x7 = sites["x7"]

        # Objective: weight of the speed reducer
        sites["f"] = (
            0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )

        # Constraints g1-g11 (all formulated as g <= 0)
        sites["g1"] = 27.0 / (x1 * x2**2 * x3) - 1.0
        sites["g2"] = 397.5 / (x1 * x2**2 * x3**2) - 1.0
        sites["g3"] = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0
        sites["g4"] = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0
        sites["g5"] = (
            np.sqrt((745.0 * x4 / (x2 * x3))**2 + 16.9e6)
            / (110.0 * x6**3) - 1.0
        )
        sites["g6"] = (
            np.sqrt((745.0 * x5 / (x2 * x3))**2 + 157.5e6)
            / (85.0 * x7**3) - 1.0
        )
        sites["g7"] = x2 * x3 / 40.0 - 1.0
        sites["g8"] = 5.0 * x2 / x1 - 1.0
        sites["g9"] = x1 / (12.0 * x2) - 1.0
        sites["g10"] = (1.5 * x6 + 1.9) / x4 - 1.0
        sites["g11"] = (1.1 * x7 + 1.9) / x5 - 1.0
