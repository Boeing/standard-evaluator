"""A Python module to provide access to DasTruss optimization test problems"""

# pylint: disable=W0223

import numpy as np
import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class DasTruss(TestEvaluator):
    r"""This is a truss optimization problem.  As a more realistic mid-sized problem,
    the objectives can be combined in various ways to explore problems of different
    dimension.  Any stress response which is not included as an objective should
    be included in a maximum stress constraint (g4-g6).

    For example, cases considered in the reference are:

        A)  Minimize displacement (f0) and volume (f1)
        B)  Minimize stress in right bar (f4) and volume (f1)
        C)  Minimize stress in left bar (f2) and volume (f1)
        D)  Minimize stress in left (f2) and right bar (f4) and volume (f1)

    You will need to set up your own OptProblem and constraints for these smaller
    cases.

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
        - :math:`f_2(X) = \frac{EM}{L} * |d_0| * sin(t_0)`
        - :math:`f_3(X) = \frac{EM}{L} * |d_1|`
        - :math:`f_4(X) = \frac{EM}{L} * |d_2| * sin(t_1)`

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
        Creates and configures an optimization problem for a truss structure.

        This method initializes a new optimization problem with specified
        independent and dependent variables, sets default values, bounds,
        and scales for the design variables, and defines the objectives
        and constraints associated with the truss optimization problem.

        The truss optimization problem aims to minimize various objectives
        such as total displacement and stress in the bars while adhering
        to geometric consistency and maximum stress constraints.

        The following components are defined within the optimization problem:

        - **Design Variables**:
            - `x`: Placement of the middle bar along the beam.
            - `a0`, `a1`, `a2`: Cross-sectional areas of the left, middle,
            and right bars, respectively.

        - **Objectives**:
            - `f0`: Squared Total Displacement.
            - `f1`: Total Volume of Beams.
            - `f2`: Absolute Stress of the Left Bar.
            - `f3`: Absolute Stress of the Middle Bar.
            - `f4`: Absolute Stress of the Right Bar.

        - **Constraints**:
            - `g0`, `g1`, `g2`, `g3`: Geometric Consistency Constraints.
            - `g4`, `g5`, `g6`: Maximum Stress Constraints.

        - **State Variables**: Angles of the bars, total displacements,
        elongations, stiffness matrix, and force matrices.

        - **Givens**: Constants such as modulus of elasticity, lengths of
        the bars and beam, and applied loads.

        Returns:
            new_prob: An instance of the optimization problem configured
            with the defined objectives, constraints, and variables.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=5, num_dependent=12, name="das_truss"
        )
        fixed_var = f"beam_len"
        var_names = [f"a{items}" for items in range(3)] + [f"x" for items in [0]] + [fixed_var]
        # Define default values
        defaults = self._def_initial_guess()
        var_bounds = ([0.8, 3.0], [0.8, 3.0], [0.8, 3.0], [360.0, 1080.0], [1440,1440])
        var_scales = [1.0, 1.0, 1.0, 0.01, 1.0]
        for var, local_default, local_bound, local_scale, name  in zip(
            new_prob.variables, defaults, var_bounds, var_scales, var_names
        ):
            var.bounds = local_bound
            var.scale = local_scale
            var.default = local_default
            var.name = name

        # Define the bounds on the constraints
        for resp in new_prob.responses[5:9]:
            resp.bounds = (-np.inf, 0)

        # Define and set the scales
        scales = [
            0.01,
            0.001,
            1.0e-05,
            1.0e-05,
            1.0e-05,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0e-05,
            1.0e-05,
            1.0e-05,
        ]

        resp_names = [f"f{items}" for items in range(5)] + [
            f"g{items}" for items in range(7)
        ]
        for resp, scale, name in zip(new_prob.responses, scales, resp_names):
            resp.scale = scale
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f0", "f1", "f2", "f3", "f4"]
        new_prob.constraints = ["g0", "g1", "g2", "g3"]

        # Define th description of the problem
        new_prob.description = r"""$$
This is a truss optimization problem.  As a more realistic mid-sized problem,
the objectives can be combined in various ways to explore problems of different
dimension.  Any stress response which is not included as an objective should
be included in a maximum stress constraint (g4-g6).

For example, cases considered in the reference are:

    A)  Minimize displacement (f0) and volume (f1)
    B)  Minimize stress in right bar (f4) and volume (f1)
    C)  Minimize stress in left bar (f2) and volume (f1)
    D)  Minimize stress in left (f2) and right bar (f4) and volume (f1)

You will need to set up your own OptProblem and constraints for these smaller
cases.

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
    - :math:`f_2(X) = \frac{EM}{L} * |d_0| * sin(t_0)`
    - :math:`f_3(X) = \frac{EM}{L} * |d_1|`
    - :math:`f_4(X) = \frac{EM}{L} * |d_2| * sin(t_1)`

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
$$"""
        # Define the citation
        new_prob.cite = "Das (1997) Nonlinear multicriteria optimization and robust optimality.  Ph.D. Thesis, Dept. of Computational and Applied Mathematics, Rice University, Houston, TX."
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer

        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """       
        return [1.9, 1.9, 1.9, 720.0, 1440]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the DasTruss function

        :param df: The dataframe that contains the input values, and is updated with the responses
        :type df: DataFrame
        """
        # pylint: disable=R0914
        m_em = 29000000.0  # (lbf/in^2)          Modulus of Elasticity (steel)
        m_l = 720.0  # (in)                Length of middle bar, fixed
        m_d = sites.beam_len            #  Length of beam
        m_w0 = 100000.0  # (lbf)               Wind load (horizontal)
        m_w1 = 1000000.0  # (lbf)               Suspended load (vertical)
        m_smax = 550000.0  # (lbf/in^2)          Maximum stress allowable for bars

        t_0 = np.arctan(m_l / sites.x)
        t_1 = np.arctan(m_l / (m_d - sites.x))

        k_11 = (sites.a0 * np.sin(t_0) * np.cos(t_0) * np.cos(t_0)) + (
            sites.a2 * np.sin(t_1) * np.cos(t_1) * np.cos(t_1)
        )
        k_12 = (sites.a0 * np.sin(t_0) * np.sin(t_0) * np.cos(t_0)) - (
            sites.a2 * np.sin(t_1) * np.sin(t_1) * np.cos(t_1)
        )
        k_21 = (sites.a0 * np.sin(t_0) * np.sin(t_0) * np.cos(t_0)) - (
            sites.a2 * np.sin(t_1) * np.sin(t_1) * np.cos(t_1)
        )
        k_22 = (sites.a0 * np.sin(t_0) * np.sin(t_0) * np.sin(t_0)) + (
            sites.a1 + sites.a2 * np.sin(t_1) * np.sin(t_1) * np.sin(t_1)
        )
        k_11 *= m_em / m_l
        k_12 *= m_em / m_l
        k_21 *= m_em / m_l
        k_22 *= m_em / m_l
        u_0 = (m_w0 - (k_12 * m_w1) / k_22) / (k_11 - (k_12 * k_21) / k_22)
        u_1 = (m_w1 - k_21 * u_0) / k_22
        # Compute elongation
        d_0 = np.cos(t_0) * u_0 + np.sin(t_0) * u_1
        d_1 = u_1
        d_2 = -np.cos(t_1) * u_0 + np.sin(t_1) * u_1
        # Objective 0: Squared Total Displacement
        sites["f0"] = u_0 * u_0 + u_1 * u_1
        # Objective 1: Total Volume of Beams
        sites["f1"] = (
            sites.a0 * m_l / np.sin(t_0) + sites.a1 * m_l + sites.a2 * m_l / np.sin(t_1)
        )
        # Objective 2: Absolute stress of left bar
        sites["f2"] = (m_em / m_l) * abs(d_0) * np.sin(t_0)
        # Objective 3: Absolute stress of middle bar
        sites["f3"] = (m_em / m_l) * abs(d_1)
        # Objective 4: Absolute stress of right bar
        sites["f4"] = (m_em / m_l) * abs(d_2) * np.sin(t_1)
        # Constraint 0, 1, 2, 3: Geometric Consistency
        sites["g0"] = sites.x - m_l / np.tan(t_0)
        sites["g1"] = -sites.g0
        sites["g2"] = (m_d - sites.x) - (m_l / np.tan(t_1))
        sites["g3"] = -sites.g2
        sites["g4"] = abs(sites.f2) - m_smax
        sites["g5"] = abs(sites.f3) - m_smax
        sites["g6"] = abs(sites.f4) - m_smax
