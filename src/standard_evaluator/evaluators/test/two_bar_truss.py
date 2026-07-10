"""A Python module to provide access to TwoBarTruss optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class TwoBarTruss(TestEvaluator):
    """The Two Bar Truss evaluator. This evaluator is a concrete class which inherits
    from the abstract DE::Evaluator class.  This is to be used as an example
    for deriving Design Explorer Evaluators and for testing future optimization
    methods.

    The example is a two-bar truss. See
    A robust design method using variable transformation and Gauss-Hermite
    integration by Beiqing Huang and Xiaoping Du,
    INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING
    Int. J. Numer. Meth. Engng 2006; 66:1841-1858

    The problem has five design variables with types as follows:
       - X1 - double,  cross-sectional area of the truss
       - X2 - double,  half distance between two bottom rollers.
       - rho - double, the density of the bar material
       - Q - double, magnitude of the external force Q applied on the top of the truss
       - S - double, the bar material's tensile strength

    There are three responses:
       - f, double, weight
       - g1, double, strength 1
       - g2, double, strength 2

    The objective of the design is to minimize the weight of the two-bar truss subject to
    the two strength constraints about the axial stress in each bar.
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem instance for the Two Bar Truss evaluator.

        This method initializes an optimization problem with the following characteristics:
        - Number of independent variables: 5
        - Number of dependent variables: 3
        - Problem name: "two_bar_truss"

        The design variables are defined as follows:
        - **Q**: double, magnitude of the external force Q applied on the top of the truss
        - **S**: double, the bar material's tensile strength
        - **X1**: double, cross-sectional area of the truss
        - **X2**: double, half distance between two bottom rollers
        - **rho**: double, the density of the bar material

        The responses of the problem are:
        - **f**: double, weight of the truss
        - **g1**: double, strength constraint 1
        - **g2**: double, strength constraint 2

        The objective of the design is to minimize the weight of the two-bar truss while satisfying the strength constraints related to the axial stress in each bar.

        The problem is cited from the following reference:
        Beiqing Huang and Xiaoping Du, "A robust design method using variable transformation and Gauss-Hermite integration," *INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING*, Int. J. Numer. Meth. Engng 2006; 66:1841-1858.

        Returns:
            OptProblem: An instance of the optimization problem configured for the Two Bar Truss evaluator.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=5, num_dependent=3, name="two_bar_truss"
        )

        var_names = ["Q", "S", "X1", "X2", "rho"]
        var_bounds = (
            [0.0, np.inf],
            [0.0, np.inf],
            [0.2, 20.0],
            [0.1, 1.6],
            [0.0, np.inf],
        )
        # Define default values
        defaults = self._def_initial_guess()

        for var, local_default, local_bound, name in zip(
            new_prob.variables, defaults, var_bounds, var_names
        ):
            var.bounds = local_bound
            var.default = local_default
            var.name = name

        resp_names = [f"f"] + [f"g{items + 1}" for items in range(2)]
        resp_bounds = ([-np.inf, np.inf], [0.0, np.inf], [0.0, np.inf])
        # Define the bounds on the constraints
        for resp, local_bounds, name in zip(
            new_prob.responses, resp_bounds, resp_names
        ):
            resp.bounds = local_bounds
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["g1", "g2"]

        # Define th description of the problem
        new_prob.description = """The Two Bar Truss evaluator. This evaluator is a concrete class which inherits
    from the abstract DE::Evaluator class.  This is to be used as an example
    for deriving Design Explorer Evaluators and for testing future optimization
    methods.

    The example is a two-bar truss. See
    A robust design method using variable transformation and Gauss-Hermite
    integration by Beiqing Huang and Xiaoping Du,
    INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING
    Int. J. Numer. Meth. Engng 2006; 66:1841-1858

    The problem has five design variables with types as follows:
       - X1 - double,  cross-sectional area of the truss
       - X2 - double,  half distance between two bottom rollers.
       - rho - double, the density of the bar material
       - Q - double, magnitude of the external force Q applied on the top of the truss
       - S - double, the bar material's tensile strength

    There are three responses:
       - f, double, weight
       - g1, double, strength 1
       - g2, double, strength 2

    The objective of the design is to minimize the weight of the two-bar truss subject to
    the two strength constraints about the axial stress in each bar."""
        # Define the citation
        new_prob.cite = 'Beiqing Huang and Xiaoping Du, "A robust design method using variable transformation and Gauss-Hermite integration," *INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING*, Int. J. Numer. Meth. Engng 2006; 66:1841-1858.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [800.0, 1050.0, 5.3791, 0.377, 10000.0]

    def _evaluate(self, sites: pd.DataFrame):
        """Call to the Trigonometric function

        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["f"] = sites.rho * sites.X1 * np.sqrt(1.0 + sites.X2 * sites.X2)
        sites["g1"] = 1.0 - 5.0 * sites.Q * np.sqrt(1.0 + sites.X2 * sites.X2) * (
            8.0 / sites.X1 + 1.0 / (sites.X1 * sites.X2)
        ) / (np.sqrt(65.0) * sites.S)
        sites["g2"] = 1.0 - 5.0 * sites.Q * np.sqrt(1.0 + sites.X2 * sites.X2) * (
            8.0 / sites.X1 - 1.0 / (sites.X1 * sites.X2)
        ) / (np.sqrt(65.0) * sites.S)
