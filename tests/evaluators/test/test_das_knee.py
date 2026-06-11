"""Testing of the DasKnee class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import DasKnee


def test_dasknee():
    """Test the DasKnee test function"""
    # Instantiate the test function
    test_func = DasKnee()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    # Duplicate to make sure vectorization works
    initial_guess = pd.concat(
        [
            initial_guess,
            initial_guess,
        ],
        ignore_index=True,
    )
    # Change x1, x2, and x3
    initial_guess.loc[1, "x0"] = 2.0
    initial_guess.loc[1, "x1"] = 1.4
    initial_guess.loc[1, "x2"] = 3.2
    initial_guess.loc[1, "x3"] = 1.8
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == ["x0", "x1", "x2", "x3", "x4"]
    assert test_func.outputs == ["f0", "f1", "g0", "g1", "g2", "g3", "g4"]
    assert initial_guess.iloc[0].f0 == pytest.approx(0.0)
    assert initial_guess.iloc[0].f1 == pytest.approx(0.0)
    assert initial_guess.iloc[0].g0 == pytest.approx(-2.0)
    assert initial_guess.iloc[0].g1 == pytest.approx(2.0)
    assert initial_guess.iloc[0].g2 == pytest.approx(0.0)
    assert initial_guess.iloc[0].g3 == pytest.approx(-0.0)
    assert initial_guess.iloc[0].g4 == pytest.approx(-10.0)
    assert initial_guess.iloc[1].f0 == pytest.approx(5.32800)
    assert initial_guess.iloc[1].f1 == pytest.approx(379.86276445)
    assert initial_guess.iloc[1].g0 == pytest.approx(0.4999999)
    assert initial_guess.iloc[1].g1 == pytest.approx(-0.49999999)
    assert initial_guess.iloc[1].g2 == pytest.approx(8.84)
    assert initial_guess.iloc[1].g3 == pytest.approx(-8.84)
    assert initial_guess.iloc[1].g4 == pytest.approx(9.44000)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = DasKnee()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 5, "Expected 5 variables"
    assert len(opt_problem.responses) == 7, "Expected 7 responses"

    # Check the names of variables
    expected_variable_names = ["x0", "x1", "x2", "x3", "x4"]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f0", "f1", "g0", "g1", "g2", "g3", "g4"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["f0", "f1"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = ["g0", "g1", "g2", "g3", "g4"]
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check the initial guess
    expected_initial_guess = (0.0, 0.0, 0.0, 0.0, 0.0)
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check bounds for variables
    for var in opt_problem.variables:
        assert var.bounds == (-4.0, 4.0), f"Bounds for {var.name} are not set correctly"

    # Check response bounds
    expected_resp_bounds = (
        (float("-inf"), 0.0),
        (float("-inf"), 0.0),
        (float("-inf"), 0.0),
        (float("-inf"), 0.0),
        (float("-inf"), 0.0),
    )
    for resp, expected_bounds in zip(opt_problem.responses[2:], expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the scales
    expected_scales = [0.01, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0]
    actual_scales = [resp.scale for resp in opt_problem.responses]
    assert (
        actual_scales == expected_scales
    ), f"Expected scales {expected_scales}, got {actual_scales}"

    # Check the description
    expected_description = r"""$$
This is a truss optimization problem.  As a more realistic mid-sized problem
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
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = "Das (1997) Nonlinear multicriteria optimization and robust optimality.  Ph.D. Thesis, Dept. of Computational and Applied Mathematics, Rice University, Houston, TX."
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = DasKnee()
    initial_guess_df = test_func.initial_guess()
    # Evaluate the initial guess
    test_func(initial_guess_df)

    # calculate the expected responses
    expected_f0 = (0.0**2 + 0.0**2 + 0.0**2 + 0.0**2 + 0.0**2) - (
        4.0 * 0.0 * (0.0 + 3.4 * 0.0) * (0.0 + 3.4 * 0.0)
    )
    expected_f1 = (
        100.0 * (0.0 - 0.0)
        + (3.0 * 0.0 + 2.0 * 0.0 - 0.0 / 3.0) ** 2
        + 0.01 * (0.0 - 0.0) ** 3
    )
    expected_g0 = 0.0 + 2.0 * 0.0 - 0.0 + 0.5 * 0.0 + 0.0 - 2.0
    expected_g1 = -expected_g0
    expected_g2 = 4.0 * 0.0 - 2.0 * 0.0 + 0.8 * 0.0 + 0.6 * 0.0 + 0.5 * 0.0**2
    expected_g3 = -expected_g2
    expected_g4 = 0.0**2 + 0.0**2 + 0.0**2 + 0.0**2 + 0.0**2 - 10.0

    # Assertion check for the evaluated responses and the expected values
    assert initial_guess_df.iloc[0].f0 == expected_f0
    assert initial_guess_df.iloc[0].f1 == expected_f1
    assert initial_guess_df.iloc[0].g0 == expected_g0
    assert initial_guess_df.iloc[0].g1 == expected_g1
    assert initial_guess_df.iloc[0].g2 == expected_g2
    assert initial_guess_df.iloc[0].g3 == expected_g3
    assert initial_guess_df.iloc[0].g4 == expected_g4
