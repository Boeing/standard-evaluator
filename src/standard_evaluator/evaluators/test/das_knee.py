"""A Python module to provide access to DasKnee optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class DasKnee(TestEvaluator):
    r"""This is a truss optimization problem.  As a more realistic mid-sized problem
    the objectives can be combined in various ways to explore problems of different
    dimension.  Any stress response which is not included as an objective should
    be included in a maximum stress constraint (g4-g6).

    Problem Description:

        A beam of length D has three bars suspended via pin joints (free-rotating).

        The middle bar is placed at position x.  It is vertical and of length L.

        The left and right bars are placed at the beginning and end of the beam,
        and adjusted in length and angle to join the middle bar at point P.
        The angles are t0 and t1 for the left and right bars, respectively.

        A horizontal wind load W0 and a suspended load W1 are applied at point P.

        The placement of the middle bar, x, and the cross sectional areas (a0, a1,
        and a2) are the design variables.

        The objectives include total displacement, total volume, and maximum stress
        in each of the bars.

        The constraints are geometric consistency and maximum allowable stress.

    Objectives:
        - **f0**: Squared Total Displacement
        - **f1**: Total Volume of Beams
        - **f2, f3, f4**: Absolute Stress of Left, Middle, and Right Bars (respectively)

    Design Variables
        - **x**: Placement of middle bar along beam
        - **a0, a1, a2**: Cross-sectional areas of left, middle, and right bars (resp.)

    Constraints:
        - **g0, g1, g2, g3**: Geometric Consistency Constraints*
        - **g4, g5, g6**: Maximum Stress Constraints

    State Variables:
        - **t0, t1**: Angles of left and right bars
        - **u0, u1**: Total Displacement (horizontal, vertical)
        - **d0, d1, d2**: Elongation of left, middle, and right bars
        - **K**: Structure stiffness matrix
        - **A**: Matrix relating bar forces to load forces

    Givens:
        - Modulus of Elasticity (steel) = EM = 29,000,000.0 lbf / in^2
        - Length of middle bar, fixed = L = 720 in
        - Length of beam = D = 1440 in
        - Wind load (horizontal) = W0 = 100,000 lbf
        - Suspended load (vertical) = W1 = 1,000,000 lbf
        - Maximum stress allowable for bars = Smax = 550,000 lbf / in^2

    .. note::
        g1 = -g0, and g3 = -g2, because these are originally equality
        constraints that have been transformed to inequality constraints.

    - Number of Objectives:               2-5     Medium
    - Number of Design Variables:         4       Low-Medium
    - Number of Constraints:              4-7     Low-Medium

    minimize
        - :math:`f_0(X) = u_0^2 + u_1^2`
        - :math:`f_1(X) = a_0\frac{L}{\sin(t_0)} + a_1L + a_2\frac{L}{\sin(t_1)}`
        - :math:`f_2(X) = \frac{EM}{L} * |d_0| * \sin(t_0)`
        - :math:`f_3(X) = \frac{EM}{L} * |d_1|`
        - :math:`f_4(X) = \frac{EM}{L} * |d_2| * \sin(t_1)`

    over

        - X = [x, a0, a1, a2]

    such that
        - :math:`g_0(X) = x - L\cot(t_0) \leq 0`
        - :math:`g_1(X) = L\cot(t_0) - x \leq 0`
        - :math:`g_2(X) = (D-x) - L\cot(t_1) \leq 0`
        - :math:`g_3(X) = L\cot(t_1) - (D-x) \leq 0`
        - :math:`g_4(X) = |f_2| - S_\text{max} \leq 0`        // g4 - g6 included only
        - :math:`g_5(X) = |f_3| - S_\text{max} \leq 0`        // if these responses are
        - :math:`g_6(X) = |f_4| - S_\text{max} \leq 0`        // not objectives

    where

        - 0.25*D  <=  x           <=  0.75*D   (in)
        - 0.8     <=  a0, a1, a2  <=  3.0      (in)

    Starting Point
        - x           = 0.50*D            (Center of space)
        - a0, a1, a2  = 1.9
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for a truss structure.

        This method initializes a new optimization problem with five independent variables
        and seven responses, specifically designed for a truss optimization scenario.
        It sets default values for the design variables, defines the bounds on the constraints,
        and specifies the objectives and constraints of the problem. The objectives include
        minimizing total displacement and total volume, while the constraints ensure
        geometric consistency and maximum allowable stress.

        Returns:
            OptProblem: An instance of the OptProblem class containing the defined optimization
            problem, including objectives, constraints, variable defaults, and a detailed
            description of the problem.

        Problem Description:
            A beam of length D has three bars suspended via pin joints (free-rotating).
            The middle bar is placed at position x and is vertical with length L.
            The left and right bars are adjusted in length and angle to join the middle bar
            at point P, with angles t0 and t1. A horizontal wind load W0 and a suspended load
            W1 are applied at point P. The design variables include the placement of the
            middle bar and the cross-sectional areas of the bars.

        Objectives:
            - f0: Squared Total Displacement
            - f1: Total Volume of Beams
            - f2, f3, f4: Absolute Stress of Left, Middle, and Right Bars (respectively)

        Design Variables:
            - x: Placement of middle bar along beam
            - a0, a1, a2: Cross-sectional areas of left, middle, and right bars (resp.)

        Constraints:
            - g0, g1, g2, g3: Geometric Consistency Constraints
            - g4, g5, g6: Maximum Stress Constraints

        State Variables:
            - t0, t1: Angles of left and right bars
            - u0, u1: Total Displacement (horizontal, vertical)
            - d0, d1, d2: Elongation of left, middle, and right bars
            - K: Structure stiffness matrix
            - A: Matrix relating bar forces to load forces

        Givens:
            - Modulus of Elasticity (steel) = EM = 29,000,000.0 lbf / in²
            - Length of middle bar, fixed = L = 720 in
            - Length of beam = D = 1440 in
            - Wind load (horizontal) = W0 = 100,000 lbf
            - Suspended load (vertical) = W1 = 1,000,000 lbf
            - Maximum stress allowable for bars = Smax = 550,000 lbf / in²

        Note:
            g1 = -g0, and g3 = -g2, because these are originally equality constraints
            that have been transformed to inequality constraints.

        Starting Point:
            - x = 0.50*D (Center of space)
            - a0, a1, a2 = 1.9
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=5, num_dependent=7, name="das_knee"
        )
        # Define default values
        defaults = self._def_initial_guess()

        for var, local_default in zip(new_prob.variables, defaults):
            var.bounds = [-4.0, 4.0]
            var.default = local_default

        # Define the bounds on the constraints
        for resp in new_prob.responses[2:]:
            resp.bounds = (-np.inf, 0)

        # Define and set the scales
        scales = [0.01, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0]
        resp_names = [f"f{items}" for items in [0, 1]] + [
            f"g{items}" for items in range(5)
        ]
        for resp, scale, name in zip(new_prob.responses, scales, resp_names):
            resp.scale = scale
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f0", "f1"]
        new_prob.constraints = ["g0", "g1", "g2", "g3", "g4"]

        # Define th description of the problem
        new_prob.description = r"""This is a truss optimization problem.  As a more realistic mid-sized problem
the objectives can be combined in various ways to explore problems of different
dimension.  Any stress response which is not included as an objective should
be included in a maximum stress constraint (g4-g6).

Problem Description:

    A beam of length D has three bars suspended via pin joints (free-rotating).

    The middle bar is placed at position x.  It is vertical and of length L.

    The left and right bars are placed at the beginning and end of the beam,
    and adjusted in length and angle to join the middle bar at point P.
    The angles are t0 and t1 for the left and right bars, respectively.

    A horizontal wind load W0 and a suspended load W1 are applied at point P.

    The placement of the middle bar, x, and the cross sectional areas (a0, a1,
    and a2) are the design variables.

    The objectives include total displacement, total volume, and maximum stress
    in each of the bars.

    The constraints are geometric consistency and maximum allowable stress.

Objectives:
    - **f0**: Squared Total Displacement
    - **f1**: Total Volume of Beams
    - **f2, f3, f4**: Absolute Stress of Left, Middle, and Right Bars (respectively)

Design Variables
    - **x**: Placement of middle bar along beam
    - **a0, a1, a2**: Cross-sectional areas of left, middle, and right bars (resp.)

Constraints:
    - **g0, g1, g2, g3**: Geometric Consistency Constraints*
    - **g4, g5, g6**: Maximum Stress Constraints

State Variables:
    - **t0, t1**: Angles of left and right bars
    - **u0, u1**: Total Displacement (horizontal, vertical)
    - **d0, d1, d2**: Elongation of left, middle, and right bars
    - **K**: Structure stiffness matrix
    - **A**: Matrix relating bar forces to load forces

Givens:
    - Modulus of Elasticity (steel) = EM = 29,000,000.0 lbf / in^2
    - Length of middle bar, fixed = L = 720 in
    - Length of beam = D = 1440 in
    - Wind load (horizontal) = W0 = 100,000 lbf
    - Suspended load (vertical) = W1 = 1,000,000 lbf
    - Maximum stress allowable for bars = Smax = 550,000 lbf / in^2

.. note::
    g1 = -g0, and g3 = -g2, because these are originally equality
    constraints that have been transformed to inequality constraints.

- Number of Objectives:               2-5     Medium
- Number of Design Variables:         4       Low-Medium
- Number of Constraints:              4-7     Low-Medium

minimize
    - :math:`f_0(X) = u_0^2 + u_1^2`
    - :math:`f_1(X) = a_0\frac{L}{\sin(t_0)} + a_1L + a_2\frac{L}{\sin(t_1)}`
    - :math:`f_2(X) = \frac{EM}{L} * |d_0| * \sin(t_0)`
    - :math:`f_3(X) = \frac{EM}{L} * |d_1|`
    - :math:`f_4(X) = \frac{EM}{L} * |d_2| * \sin(t_1)`

over

    - X = [x, a0, a1, a2]

such that
    - :math:`g_0(X) = x - L\cot(t_0) \leq 0`
    - :math:`g_1(X) = L\cot(t_0) - x \leq 0`
    - :math:`g_2(X) = (D-x) - L\cot(t_1) \leq 0`
    - :math:`g_3(X) = L\cot(t_1) - (D-x) \leq 0`
    - :math:`g_4(X) = |f_2| - S_\text{max} \leq 0`        // g4 - g6 included only
    - :math:`g_5(X) = |f_3| - S_\text{max} \leq 0`        // if these responses are
    - :math:`g_6(X) = |f_4| - S_\text{max} \leq 0`        // not objectives

where

    - 0.25*D  <=  x           <=  0.75*D   (in)
    - 0.8     <=  a0, a1, a2  <=  3.0      (in)

Starting Point
    - x           = 0.50*D            (Center of space)
    - a0, a1, a2  = 1.9"""
        # Define the citation
        new_prob.cite = "Das (1997) Nonlinear multicriteria optimization and robust optimality.  Ph.D. Thesis, Dept. of Computational and Applied Mathematics, Rice University, Houston, TX."
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def _evaluate(self, sites: pd.DataFrame):
        """Call to the DasKnee function

        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        sites["f0"] = (
            (sites.x0 * sites.x0)
            + (sites.x1 * sites.x1)
            + (sites.x2 * sites.x2)
            + (sites.x3 * sites.x3)
            + (sites.x4 * sites.x4)
        ) - (4.0 * sites.x3 * (sites.x1 + 3.4 * sites.x4) * (sites.x1 + 3.4 * sites.x4))
        sites["f1"] = (
            100.0 * (sites.x2 - sites.x4)
            + (3.0 * sites.x0 + 2.0 * sites.x1 - sites.x2 / 3.0)
            * (3.0 * sites.x0 + 2.0 * sites.x1 - sites.x2 / 3.0)
            + 0.01
            * (sites.x3 - sites.x4)
            * (sites.x3 - sites.x4)
            * (sites.x3 - sites.x4)
        )
        sites["g0"] = (
            sites.x0 + 2.0 * sites.x1 - sites.x2 + 0.5 * sites.x3 + sites.x4 - 2.0
        )
        sites["g1"] = -sites.g0
        sites["g2"] = (
            4.0 * sites.x0
            - 2.0 * sites.x1
            + 0.8 * sites.x2
            + 0.6 * sites.x3
            + 0.5 * sites.x4 * sites.x4
        )
        sites["g3"] = -sites.g2
        sites["g4"] = (
            (sites.x0 * sites.x0)
            + (sites.x1 * sites.x1)
            + (sites.x2 * sites.x2)
            + (sites.x3 * sites.x3)
            + (sites.x4 * sites.x4)
            - 10.0
        )
