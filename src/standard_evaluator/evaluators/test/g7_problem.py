"""G7 benchmark problem evaluator.

Implements the G7 constrained optimization benchmark — a 10D problem with
8 nonlinear inequality constraints from Hock & Schittkowski (1981).
"""

import numpy as np
import pandas as pd

from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class G7Problem(TestEvaluator):
    """G7 constrained optimization benchmark (10D, 8 constraints).

    Minimise:
        f = x1² + x2² + x1·x2 − 14·x1 − 16·x2 + (x3−10)²
            + 4·(x4−5)² + (x5−3)² + 2·(x6−1)² + 5·x7²
            + 7·(x8−11)² + 2·(x9−10)² + (x10−7)² + 45

    Subject to g1–g8 ≤ 0:
        g1 = −105 + 4·x1 + 5·x2 − 3·x7 + 9·x8
        g2 = 10·x1 − 8·x2 − 17·x7 + 2·x8
        g3 = −8·x1 + 2·x2 + 5·x9 − 2·x10 − 12
        g4 = 3·(x1−2)² + 4·(x2−3)² + 2·x3² − 7·x4 − 120
        g5 = 5·x1² + 8·x2 + (x3−6)² − 2·x4 − 40
        g6 = x1² + 2·(x2−2)² − 2·x1·x2 + 14·x5 − 6·x6
        g7 = 0.5·(x1−8)² + 2·(x2−4)² + 3·x5² − x6 − 30
        g8 = −3·x1 + 6·x2 + 12·(x9−8)² − 7·x10

    with x_i ∈ [-10, 10] for i = 1, ..., 10.

    Citation:
        Hock, W. and Schittkowski, K., "Test Examples for Nonlinear
        Programming Codes", Springer-Verlag, 1981.
    """

    def _create_opt_problem(self) -> OptProblem:
        """Create the G7 optimization problem definition.

        Returns:
            OptProblem: Problem with 10 variables, 9 responses (f + g1–g8).
        """
        n_vars = 10
        n_responses = 9  # f + g1–g8

        new_prob = se.utilities.create_opt_problem(
            num_independent=n_vars,
            num_dependent=n_responses,
            name="g7_problem",
        )

        # Define variables: x1–x10, each ∈ [-10, 10]
        var_names = [f"x{i + 1}" for i in range(n_vars)]
        defaults = self._def_initial_guess()

        for var, default, name in zip(
            new_prob.variables, defaults, var_names
        ):
            var.name = name
            var.bounds = [-10.0, 10.0]
            var.default = default

        # Define responses: f and g1–g8
        resp_names = ["f"] + [f"g{i + 1}" for i in range(8)]
        resp_bounds = [[-np.inf, np.inf]] + [[-np.inf, 0.0]] * 8

        for resp, bounds, name in zip(
            new_prob.responses, resp_bounds, resp_names
        ):
            resp.name = name
            resp.bounds = bounds

        # Objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = [f"g{i + 1}" for i in range(8)]

        # Metadata
        new_prob.description = """\
G7 constrained optimization benchmark (10D, 8 constraints).
Hock & Schittkowski, 1981.

Minimise:

$$f = x_1^2 + x_2^2 + x_1 x_2 - 14 x_1 - 16 x_2 + (x_3 - 10)^2 + 4(x_4 - 5)^2 + (x_5 - 3)^2 + 2(x_6 - 1)^2 + 5 x_7^2 + 7(x_8 - 11)^2 + 2(x_9 - 10)^2 + (x_{10} - 7)^2 + 45$$

Subject to:

| g1 = -105 + 4*x1 + 5*x2 - 3*x7 + 9*x8 <= 0
| g2 = 10*x1 - 8*x2 - 17*x7 + 2*x8 <= 0
| g3 = -8*x1 + 2*x2 + 5*x9 - 2*x10 - 12 <= 0
| g4 = 3*(x1-2)^2 + 4*(x2-3)^2 + 2*x3^2 - 7*x4 - 120 <= 0
| g5 = 5*x1^2 + 8*x2 + (x3-6)^2 - 2*x4 - 40 <= 0
| g6 = x1^2 + 2*(x2-2)^2 - 2*x1*x2 + 14*x5 - 6*x6 <= 0
| g7 = 0.5*(x1-8)^2 + 2*(x2-4)^2 + 3*x5^2 - x6 - 30 <= 0
| g8 = -3*x1 + 6*x2 + 12*(x9-8)^2 - 7*x10 <= 0

with x_i in [-10, 10] for i = 1, ..., 10."""
        new_prob.cite = (
            "Hock, W. and Schittkowski, K., "
            '"Test Examples for Nonlinear Programming Codes", '
            "Springer-Verlag, 1981."
        )

        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns:
            list: The initial guess values [0.0] * 10.
        """
        return [0.0] * 10

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Evaluate the G7 objective and constraints.

        Args:
            sites: DataFrame containing input columns x1–x10,
                updated in-place with response columns f, g1–g8.
        """
        x1 = sites["x1"]
        x2 = sites["x2"]
        x3 = sites["x3"]
        x4 = sites["x4"]
        x5 = sites["x5"]
        x6 = sites["x6"]
        x7 = sites["x7"]
        x8 = sites["x8"]
        x9 = sites["x9"]
        x10 = sites["x10"]

        # Objective
        sites["f"] = (
            x1**2 + x2**2 + x1 * x2
            - 14 * x1 - 16 * x2
            + (x3 - 10) ** 2
            + 4 * (x4 - 5) ** 2
            + (x5 - 3) ** 2
            + 2 * (x6 - 1) ** 2
            + 5 * x7**2
            + 7 * (x8 - 11) ** 2
            + 2 * (x9 - 10) ** 2
            + (x10 - 7) ** 2
            + 45
        )

        # Constraints g1–g8
        sites["g1"] = -105 + 4 * x1 + 5 * x2 - 3 * x7 + 9 * x8
        sites["g2"] = 10 * x1 - 8 * x2 - 17 * x7 + 2 * x8
        sites["g3"] = -8 * x1 + 2 * x2 + 5 * x9 - 2 * x10 - 12
        sites["g4"] = (
            3 * (x1 - 2) ** 2 + 4 * (x2 - 3) ** 2
            + 2 * x3**2 - 7 * x4 - 120
        )
        sites["g5"] = (
            5 * x1**2 + 8 * x2 + (x3 - 6) ** 2 - 2 * x4 - 40
        )
        sites["g6"] = (
            x1**2 + 2 * (x2 - 2) ** 2
            - 2 * x1 * x2 + 14 * x5 - 6 * x6
        )
        sites["g7"] = (
            0.5 * (x1 - 8) ** 2 + 2 * (x2 - 4) ** 2
            + 3 * x5**2 - x6 - 30
        )
        sites["g8"] = (
            -3 * x1 + 6 * x2 + 12 * (x9 - 8) ** 2 - 7 * x10
        )
