"""Testing of the HS118 classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import HS118

"""Testing of the HS118"""


def test_hs118():
    """Test the HS118 test function"""
    # Instantiate the test function
    test_func = HS118()
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
    initial_guess.at[1, "x1"] = 9.5
    initial_guess.at[1, "x2"] = 47.5
    initial_guess.at[1, "x3"] = 12.9
    initial_guess.at[1, "x4"] = 60.0
    initial_guess.at[1, "x5"] = 98.0
    initial_guess.at[1, "x6"] = 44.0
    initial_guess.at[1, "x7"] = 66.6
    initial_guess.at[1, "x8"] = 23.9
    initial_guess.at[1, "x9"] = 43.2
    initial_guess.at[1, "x10"] = 33.3
    initial_guess.at[1, "x11"] = 55.8
    initial_guess.at[1, "x12"] = 18.6
    initial_guess.at[1, "x13"] = 77.7
    initial_guess.at[1, "x14"] = 99.0
    initial_guess.at[1, "x15"] = 32.8

    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
    ]
    assert test_func.outputs == [
        "f",
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
    assert initial_guess.iloc[0].f == pytest.approx(942.7162499999998)
    assert initial_guess.iloc[0].c1 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c2 == pytest.approx(12.0)
    assert initial_guess.iloc[0].c3 == pytest.approx(12.0)
    assert initial_guess.iloc[0].c4 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c5 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c6 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c7 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c8 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c9 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c10 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c11 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c12 == pytest.approx(7.0)
    assert initial_guess.iloc[0].c13 == pytest.approx(30.0)
    assert initial_guess.iloc[0].c14 == pytest.approx(50.0)
    assert initial_guess.iloc[0].c15 == pytest.approx(30.0)
    assert initial_guess.iloc[0].c16 == pytest.approx(15.0)
    assert initial_guess.iloc[0].c17 == pytest.approx(0.0)
    assert initial_guess.iloc[1].f == pytest.approx(1457.6403765000002)
    assert initial_guess.iloc[1].c1 == pytest.approx(57.5)
    assert initial_guess.iloc[1].c2 == pytest.approx(38.1)
    assert initial_guess.iloc[1].c3 == pytest.approx(57.5)
    assert initial_guess.iloc[1].c4 == pytest.approx(13.599999999999994)
    assert initial_guess.iloc[1].c5 == pytest.approx(6.200000000000003)
    assert initial_guess.iloc[1].c6 == pytest.approx(-67.1)
    assert initial_guess.iloc[1].c7 == pytest.approx(-26.299999999999997)
    assert initial_guess.iloc[1].c8 == pytest.approx(-17.6)
    assert initial_guess.iloc[1].c9 == pytest.approx(38.9)
    assert initial_guess.iloc[1].c10 == pytest.approx(51.400000000000006)
    assert initial_guess.iloc[1].c11 == pytest.approx(21.199999999999996)
    assert initial_guess.iloc[1].c12 == pytest.approx(50.2)
    assert initial_guess.iloc[1].c13 == pytest.approx(9.900000000000006)
    assert initial_guess.iloc[1].c14 == pytest.approx(152.0)
    assert initial_guess.iloc[1].c15 == pytest.approx(63.69999999999999)
    assert initial_guess.iloc[1].c16 == pytest.approx(22.69999999999999)
    assert initial_guess.iloc[1].c17 == pytest.approx(109.5)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = HS118()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 15, "Expected 15 variables"
    assert len(opt_problem.responses) == 18, "Expected 18 responses"

    # Check the names of variables
    expected_variable_names = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
    ]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = [
        "f",
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
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["f"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = [
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
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds and defaults
    expected_var_bounds = (
        (8.0, 21.0),
        (43.0, 57.0),
        (3.0, 16.0),
        (0.0, 90.0),
        (0.0, 120.0),
        (0.0, 60.0),
        (0.0, 90.0),
        (0.0, 120.0),
        (0.0, 60.0),
        (0.0, 90.0),
        (0.0, 120.0),
        (0.0, 60.0),
        (0.0, 90.0),
        (0.0, 120.0),
        (0.0, 60.0),
    )
    for var, expected_bounds in zip(opt_problem.variables, expected_var_bounds):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"

    # Check the initial guess
    expected_initial_guess = (
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
    )
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check response bounds
    expected_resp_bounds = (
        (0.0, 13.0),
        (0.0, 13.0),
        (0.0, 14.0),
        (0.0, 13.0),
        (0.0, 13.0),
        (0.0, 14.0),
        (0.0, 13.0),
        (0.0, 13.0),
        (0.0, 14.0),
        (0.0, 13.0),
        (0.0, 13.0),
        (0.0, 14.0),
        (0.0, float("inf")),
        (0.0, float("inf")),
        (0.0, float("inf")),
        (0.0, float("inf")),
        (0.0, float("inf")),
    )
    for resp, expected_bounds in zip(opt_problem.responses[1:], expected_resp_bounds):
        assert (
            resp.bounds == expected_bounds
        ), f"Response bounds do not match for response {resp}: {resp.bounds} != {expected_bounds}"

    # Check the scales
    expected_scales = [
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
    actual_scales = [resp.scale for resp in opt_problem.responses]
    assert (
        actual_scales == expected_scales
    ), f"Expected scales {expected_scales}, got {actual_scales}"

    # Check the description
    expected_description = r"""$$
$$
\begin{align}
    x0 &= ( 20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0 ) \\
    f(x0) &= 942.7162499999998 \\
    x^* &= ( 8.0, 49.0, 3.0, 1.0, 56.0, 0.0, 1.0, 63.0, 6.0, 3.0, 70.0, 12.0, 5.0, 77.0, 18.0 ) \\
    f(x^*) &= 664.82045000
\end{align}
$$
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = 'Hock, Willi, and Klaus Schittkowski. "Test examples for nonlinear programming codes." Journal of optimization theory and applications 30 (1980): 127-129.'
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = HS118()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(942.7162499999998)
    assert initial_guess_df.iloc[0].c1 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c2 == pytest.approx(12.0)
    assert initial_guess_df.iloc[0].c3 == pytest.approx(12.0)
    assert initial_guess_df.iloc[0].c4 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c5 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c6 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c7 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c8 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c9 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c10 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c11 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c12 == pytest.approx(7.0)
    assert initial_guess_df.iloc[0].c13 == pytest.approx(30.0)
    assert initial_guess_df.iloc[0].c14 == pytest.approx(50.0)
    assert initial_guess_df.iloc[0].c15 == pytest.approx(30.0)
    assert initial_guess_df.iloc[0].c16 == pytest.approx(15.0)
    assert initial_guess_df.iloc[0].c17 == pytest.approx(0.0)


def test_hs118_opt():
    "Test the optimal solution of the HS118 test function"
    # Instantiate the test function
    test_func = HS118()
    # Get the initial guess
    known_sol = test_func.known_solution
    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0].f == pytest.approx(664.82045000)
    assert known_sol.iloc[0].c1 == pytest.approx(0.0)
    assert known_sol.iloc[0].c2 == pytest.approx(4.0)
    assert known_sol.iloc[0].c3 == pytest.approx(14.0)
    assert known_sol.iloc[0].c4 == pytest.approx(7.0)
    assert known_sol.iloc[0].c5 == pytest.approx(13.0)
    assert known_sol.iloc[0].c6 == pytest.approx(14.0)
    assert known_sol.iloc[0].c7 == pytest.approx(9.0)
    assert known_sol.iloc[0].c8 == pytest.approx(13.0)
    assert known_sol.iloc[0].c9 == pytest.approx(14.0)
    assert known_sol.iloc[0].c10 == pytest.approx(9.0)
    assert known_sol.iloc[0].c11 == pytest.approx(13.0)
    assert known_sol.iloc[0].c12 == pytest.approx(14.0)
    assert known_sol.iloc[0].c13 == pytest.approx(0.0)
    assert known_sol.iloc[0].c14 == pytest.approx(7.0)
    assert known_sol.iloc[0].c15 == pytest.approx(0.0)
    assert known_sol.iloc[0].c16 == pytest.approx(0.0)
    assert known_sol.iloc[0].c17 == pytest.approx(0.0)
