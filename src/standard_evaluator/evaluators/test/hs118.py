"""A Python module to provide access to hs118 optimization test problems"""

# pylint: disable=W0223

import pandas as pd
import numpy as np
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class HS118(TestEvaluator):
    """Implement the Hock-Schittkowski number 118 problem

    x0 = ( 20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0,
    20.0 )

    f(x0) = 942.7162499999998

    x* = ( 8.0, 49.0, 3.0, 1.0, 56.0, 0.0, 1.0, 63.0, 6.0, 3.0, 70.0, 12.0, 5.0, 77.0, 18.0 )

    f(x*) = 664.82045000
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem instance for a specific test case.

        This method initializes an optimization problem with predefined variables,
        constraints, objectives, and their respective bounds and scales. The problem
        is based on the "hs118" test case, which is commonly used for benchmarking
        optimization algorithms.

        The method performs the following steps:
        1. Creates a new optimization problem with 15 variables and 18 responses.
        2. Defines default initial guesses for the variables.
        3. Sets the bounds for each variable and assigns default values.
        4. Defines the bounds for the responses.
        5. Sets the scaling factors for the responses.
        6. Specifies the objectives and constraints of the optimization problem.
        7. Provides a mathematical description of the problem.
        8. Includes a citation for the source of the test case.

        Returns:
            OptProblem: An instance of the optimization problem configured with
            the specified parameters.

        Citation:
            Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear
            programming codes." Journal of optimization theory and applications
            30 (1980): 127-129.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=15, num_dependent=18, name="hs118"
        )

        # Define default values
        defaults = self._def_initial_guess()

        var_names = [f"x{items + 1}" for items in range(15)]
        var_bounds = (
            [8.0, 21.0],
            [43.0, 57.0],
            [3.0, 16.0],
            [0.0, 90.0],
            [0.0, 120.0],
            [0.0, 60.0],
            [0.0, 90.0],
            [0.0, 120.0],
            [0.0, 60.0],
            [0.0, 90.0],
            [0.0, 120.0],
            [0.0, 60.0],
            [0.0, 90.0],
            [0.0, 120.0],
            [0.0, 60.0],
        )
        for var, local_default, name, local_bound in zip(
            new_prob.variables, defaults, var_names, var_bounds
        ):
            var.bounds = local_bound  # the default value +- infinity
            var.scale = 0.01  # the default value 1.0
            var.default = local_default
            var.name = name

        resp_names = [f"f"] + [f"c{items + 1}" for items in range(17)]
        # Define the bounds on the constraints
        resp_bounds = (
            [0.0, 13.0],
            [0.0, 13.0],
            [0.0, 14.0],
            [0.0, 13.0],
            [0.0, 13.0],
            [0.0, 14.0],
            [0.0, 13.0],
            [0.0, 13.0],
            [0.0, 14.0],
            [0.0, 13.0],
            [0.0, 13.0],
            [0.0, 14.0],
            [0.0, np.inf],
            [0.0, np.inf],
            [0.0, np.inf],
            [0.0, np.inf],
            [0.0, np.inf],
        )
        for resp, local_bound in zip(new_prob.responses[1:], resp_bounds):
            resp.bounds = local_bound

        # Define and set the scales
        scales = [
            0.001,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
        for resp, name, scale in zip(new_prob.responses, resp_names, scales):
            resp.scale = scale
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = [
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c6",
            "c7",
            "c8",
            "c9",
            "c10",
            "c11",
            "c12",
            "c13",
            "c14",
            "c15",
            "c16",
            "c17",
        ]

        # Define th description of the problem
        new_prob.description = r"""$$
\begin{align}
    x0 &= ( 20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0 ) \\
    f(x0) &= 942.7162499999998 \\
    x^* &= ( 8.0, 49.0, 3.0, 1.0, 56.0, 0.0, 1.0, 63.0, 6.0, 3.0, 70.0, 12.0, 5.0, 77.0, 18.0 ) \\
    f(x^*) &= 664.82045000
\end{align}
$$"""
        # Define the citation
        new_prob.cite = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        x0 = ( 20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0,
               60.0, 20.0 )
        f(x0) = 942.7162499999998
        """
        return [
            20.0,
            55.0,
            15.0,
            20.0,
            60.0,
            20.0,
            20.0,
            60.0,
            20.0,
            20.0,
            60.0,
            20.0,
            20.0,
            60.0,
            20.0,
        ]

    def _def_known_solution(self) -> list:
        """
        Provide the known optimal solution.
        :return: List providing the variable values of the optimal solution of the problem
        :rtype: list
        x* = (8.0, 49.0, 3.0, 1.0, 56.0, 0.0, 1.0, 63.0, 6.0, 3.0, 70.0, 12.0, 5.0, 77.0, 18.0)
        f(x*) = 664.82045000
        """
        return [
            8.0,
            49.0,
            3.0,
            1.0,
            56.0,
            0.0,
            1.0,
            63.0,
            6.0,
            3.0,
            70.0,
            12.0,
            5.0,
            77.0,
            18.0,
        ]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the HS118 function
        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["f"] = (
            2.3 * sites.x1
            + 1.0e-4 * sites.x1 * sites.x1
            + 1.7 * sites.x2
            + 1.0e-4 * sites.x2 * sites.x2
            + 2.2 * sites.x3
            + 1.5e-4 * sites.x3 * sites.x3
            + 2.3 * sites.x4
            + 1.0e-4 * sites.x4 * sites.x4
            + 1.7 * sites.x5
            + 1.0e-4 * sites.x5 * sites.x5
            + 2.2 * sites.x6
            + 1.5e-4 * sites.x6 * sites.x6
            + 2.3 * sites.x7
            + 1.0e-4 * sites.x7 * sites.x7
            + 1.7 * sites.x8
            + 1.0e-4 * sites.x8 * sites.x8
            + 2.2 * sites.x9
            + 1.5e-4 * sites.x9 * sites.x9
            + 2.3 * sites.x10
            + 1.0e-4 * sites.x10 * sites.x10
            + 1.7 * sites.x11
            + 1.0e-4 * sites.x11 * sites.x11
            + 2.2 * sites.x12
            + 1.5e-4 * sites.x12 * sites.x12
            + 2.3 * sites.x13
            + 1.0e-4 * sites.x13 * sites.x13
            + 1.7 * sites.x14
            + 1.0e-4 * sites.x14 * sites.x14
            + 2.2 * sites.x15
            + 1.5e-4 * sites.x15 * sites.x15
        )
        sites["c1"] = sites.x4 - sites.x1 + 7
        sites["c2"] = sites.x6 - sites.x3 + 7
        sites["c3"] = sites.x5 - sites.x2 + 7
        sites["c4"] = sites.x7 - sites.x4 + 7
        sites["c5"] = sites.x9 - sites.x6 + 7
        sites["c6"] = sites.x8 - sites.x5 + 7
        sites["c7"] = sites.x10 - sites.x7 + 7
        sites["c8"] = sites.x12 - sites.x9 + 7
        sites["c9"] = sites.x11 - sites.x8 + 7
        sites["c10"] = sites.x13 - sites.x10 + 7
        sites["c11"] = sites.x15 - sites.x12 + 7
        sites["c12"] = sites.x14 - sites.x11 + 7

        sites["c13"] = sites.x1 + sites.x2 + sites.x3 - 60.0
        sites["c14"] = sites.x4 + sites.x5 + sites.x6 - 50.0
        sites["c15"] = sites.x7 + sites.x8 + sites.x9 - 70.0
        sites["c16"] = sites.x10 + sites.x11 + sites.x12 - 85.0
        sites["c17"] = sites.x13 + sites.x14 + sites.x15 - 100.0
