"""A Python module to provide access to hs100 optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd

from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class HS100(TestEvaluator):
    r""".. math::
        \begin{align}
            \min\quad & (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4 + 3(x_4 - 11)^2\\
                        & + 10x_5^6 + 7x_6^2 + x_7^4 - 4x_6x_7 - 10x_6 - 8x_7\\[1em]
            \text{s.t.}\quad & 2x_1^2 + 3x_2^4 + x_3 + 4x_4^2 + 5x_5 \leq 127\\
            & 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5 \leq 282\\
            & 23x_1 + x_2^2 + 6x_6 - 8x_7 \leq 196\\
            & 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7 \geq 0
        \end{align}

    The following bounds are placed on the variables:

    .. math::
        -10 \leq x_i \leq 10.075 \qquad i = 1,...,7

    This has a known solution of

    .. math::
        f(2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227) = 680.6300573

    An initial guess is also defined for this class as

    .. math::
        f(1, 2, 0, 4, 0, 1, 1) = 714
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Create and configure the HS100 nonlinear programming test problem with named variables and responses.

        This method initializes an optimization problem with seven independent variables and five dependent responses.
        Each variable is named sequentially from x1 to x7, bounded between -10.0 and 10.075, scaled by 0.1, and initialized 
        with default values provided by self._def_initial_guess().

        The problem has one objective function "f" to minimize and four inequality constraints "c1", "c2", "c3", and "c4".
        Constraint bounds are set to be non-negative (i.e., greater than or equal to zero).

        The objective and constraints correspond to the HS100 test problem from Hock and Schittkowski (1980), defined as:

        $$
        \begin{align}
            \min\quad & (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4 + 3(x_4 - 11)^2\\
                        & + 10x_5^6 + 7x_6^2 + x_7^4 - 4x_6x_7 - 10x_6 - 8x_7\\[1em]
            \text{s.t.}\quad & 2x_1^2 + 3x_2^4 + x_3 + 4x_4^2 + 5x_5 \leq 127\\
            & 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5 \leq 282\\
            & 23x_1 + x_2^2 + 6x_6 - 8x_7 \leq 196\\
            & 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7 \geq 0
        \end{align}
        $$

        Variable bounds:
        $$
        -10 \leq x_i \leq 10.075 \quad \text{for } i=1,\ldots,7
        $$

        Returns:
            OptProblem: Configured optimization problem instance with named variables, objectives,
                        constraints, description, and citation.

        Reference:
            Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes."
            Journal of optimization theory and applications 30 (1980): 127-129.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=7, num_dependent=5, name="hs100"
        )
        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(7)]

        for var, name, default in zip(new_prob.variables, var_names, defaults):
            # Set the bounds, scale, and default values
            var.bounds = [-10.0, 10.075]
            var.scale = 0.1
            var.default = default
            var.name = name

        resp_names = [f"f"] + [f"c{items + 1}" for items in range(4)]
        # Define the bounds on the responses
        for resp in new_prob.responses[1:]:
            resp.bounds = (0, np.inf)

        # Define and set the scales
        scales = [1.0e-3, 1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2]
        for resp, name, scale in zip(new_prob.responses, resp_names, scales):
            resp.name = name
            resp.scale = scale

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["c1", "c2", "c3", "c4"]

        # Define th description of the problem
        new_prob.description = r"""$$
\begin{align}
    \min\quad & (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4 + 3(x_4 - 11)^2\\
                & + 10x_5^6 + 7x_6^2 + x_7^4 - 4x_6x_7 - 10x_6 - 8x_7\\[1em]
    \text{s.t.}\quad & 2x_1^2 + 3x_2^4 + x_3 + 4x_4^2 + 5x_5 \leq 127\\
    & 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5 \leq 282\\
    & 23x_1 + x_2^2 + 6x_6 - 8x_7 \leq 196\\
    & 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7 \geq 0
\end{align}
$$

The following bounds are placed on the variables:

$$
-10 \leq x_i \leq 10.075 \qquad i = 1,...,7
$$"""
        # Define the citation
        new_prob.cite = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
        return new_prob

    def _def_known_solution(self) -> pd.Series:
        return pd.Series(
            data={
                "x1": 2.330499,
                "x2": 1.951372,
                "x3": -0.4775414,
                "x4": 4.365726,
                "x5": -0.6244870,
                "x6": 1.038131,
                "x7": 1.594227,
            }
        )

    def _def_initial_guess(self) -> pd.Series:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [1, 2, 0, 4, 0, 1, 1]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the HS100 function

        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["f"] = (
            (sites.x1 - 10.0) * (sites.x1 - 10.0)
            + 5.0 * (sites.x2 - 12.0) * (sites.x2 - 12.0)
            + sites.x3 * sites.x3 * sites.x3 * sites.x3
            + 3.0 * (sites.x4 - 11.0) * (sites.x4 - 11.0)
            + 10.0 * sites.x5 * sites.x5 * sites.x5 * sites.x5 * sites.x5 * sites.x5
            + 7.0 * sites.x6 * sites.x6
            + sites.x7 * sites.x7 * sites.x7 * sites.x7
            - 4.0 * sites.x6 * sites.x7
            - 10.0 * sites.x6
            - 8.0 * sites.x7
        )
        sites["c1"] = (
            127.0
            - 2.0 * sites.x1 * sites.x1
            - 3.0 * sites.x2 * sites.x2 * sites.x2 * sites.x2
            - sites.x3
            - 4.0 * sites.x4 * sites.x4
            - 5.0 * sites.x5
        )
        sites["c2"] = (
            282.0
            - 7.0 * sites.x1
            - 3.0 * sites.x2
            - 10.0 * sites.x3 * sites.x3
            - sites.x4
            + sites.x5
        )
        sites["c3"] = (
            196.0
            - 23.0 * sites.x1
            - sites.x2 * sites.x2
            - 6.0 * sites.x6 * sites.x6
            + 8.0 * sites.x7
        )
        sites["c4"] = (
            -4.0 * sites.x1 * sites.x1
            - sites.x2 * sites.x2
            + 3.0 * sites.x1 * sites.x2
            - 2.0 * sites.x3 * sites.x3
            - 5.0 * sites.x6
            + 11.0 * sites.x7
        )
