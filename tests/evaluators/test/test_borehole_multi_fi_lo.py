"""Testing of the BoreholeMultiFiLo class"""

import pytest
import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test import BoreholeMultiFiLo


def test_borehole_multi_fi_lo():
    """Test the BoreholeMultiFiLo test function"""
    # Instantiate the test function
    test_func = BoreholeMultiFiLo()
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
    # Change variable values for the second site
    initial_guess.loc[1, "rad_borehole"] = 0.37824
    initial_guess.loc[1, "rad_influence"] = 47734.98
    initial_guess.loc[1, "trans_upper"] = 97563.1352
    initial_guess.loc[1, "pot_upper"] = 1005.55
    initial_guess.loc[1, "trans_lower"] = 98.3462
    initial_guess.loc[1, "pot_lower"] = 800.999
    initial_guess.loc[1, "len_borehole"] = 1322.8976
    initial_guess.loc[1, "hyd_con_borehole"] = 10012.745
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.inputs == [
        "rad_borehole",
        "rad_influence",
        "trans_upper",
        "pot_upper",
        "trans_lower",
        "pot_lower",
        "len_borehole",
        "hyd_con_borehole",
    ]
    assert test_func.outputs == ["water_flow_rate"]
    assert initial_guess.iloc[0].water_flow_rate == pytest.approx(68.60360708719466)
    assert initial_guess.iloc[1].water_flow_rate == pytest.approx(520.0578959207066)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = BoreholeMultiFiLo()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 8, "Expected 8 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 responses"

    # Check the names of variables
    expected_variable_names = [
        "rad_borehole",
        "rad_influence",
        "trans_upper",
        "pot_upper",
        "trans_lower",
        "pot_lower",
        "len_borehole",
        "hyd_con_borehole",
    ]
    actual_variable_names = [var.name for var in opt_problem.variables]
    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["water_flow_rate"]
    actual_response_names = [resp.name for resp in opt_problem.responses]
    assert (
        actual_response_names == expected_response_names
    ), f"Responses names do not match expected values"

    # Check the objective function
    expected_objectives = ["water_flow_rate"]
    assert (
        opt_problem.objectives == expected_objectives
    ), f"Expected objectives {expected_objectives}, got {opt_problem.objectives}"

    # Check the constraints
    expected_constraints = []
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds ,defaults and units
    expected_var_bounds = (
        (0.05, 0.15),
        (100.0, 50000.0),
        (63070.0, 115600.0),
        (990.0, 1110.0),
        (63.1, 116.0),
        (700.0, 820.0),
        (1120.0, 1680.0),
        (9855.0, 12045.0),
    )
    expected_var_unit_list = ["m", "m", "m**2/yr", "m", "m**2/yr", "m", "m", "m/yr"]
    for var, expected_bounds, expected_unit_list in zip(
        opt_problem.variables, expected_var_bounds, expected_var_unit_list
    ):
        assert (
            var.bounds == expected_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_bounds}"
        assert (
            var.units == expected_unit_list
        ), f"Variable bounds do not match for variable {var}: {var.units} != {expected_unit_list}"

    # Check the initial guess
    expected_initial_guess = (
        0.12899,
        679.88,
        72089.56,
        999.45,
        77.896,
        777.18,
        1340.55,
        10015.24,
    )
    actual_initial_guess = tuple(var.default for var in opt_problem.variables)
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check units of responses
    expected_resp_unit_list = ["m**3/yr"]
    for resp, expected_resp_unit in zip(opt_problem.responses, expected_resp_unit_list):
        assert (
            resp.units == expected_resp_unit
        ), f"Variable bounds do not match for variable {resp}: {resp.units} != {expected_resp_unit}"

    # Check the description
    expected_description = """The Borehole function models water flow through a borehole. Its simplicity and quick evaluation makes it a
commonly used function for testing a wide variety of methods in computer experiments.
The response is water flow rate in  𝑚3/𝑦𝑟 .

Input Domain

| Radius of borehole ( 𝑚 ) -  𝑟𝑤∈[0.05,0.15]
| Radius of influence ( 𝑚 ) -  𝑟∈[100,50000]
| Transmissivity of upper aquifier ( 𝑚2/𝑦𝑟 ) -  𝑇𝑢∈[63070,115600]
| Potentiometric head of upper aquifier ( 𝑚 ) -  𝐻𝑢∈[990,1110]
| Transmissivity of lower aquifier ( 𝑚2/𝑦𝑟 ) -  𝑇𝑙∈[63.1,116]
| Potentiometric head of lower aquifier ( 𝑚 ) -  𝐻𝑙∈[700,820]
| Length of borehole ( 𝑚 ) -  𝐿∈[1120,1680]
| Hydraulic conductivity of borehole ( 𝑚/𝑦𝑟 ) -  𝐾𝑤∈[9855,12045]"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = ""
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    test_func = BoreholeMultiFiLo()
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].water_flow_rate == pytest.approx(68.60360708719466)
