"""Testing of the ForresterMultiFi_Lo class"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import ForresterMultiFiLo


def test_forrester_multi_fi_lo():
    """Test the ForresterMultiFi_Lo test function"""
    # Instantiate the test function
    test_func = ForresterMultiFiLo()
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
    # Change the variable x
    initial_guess.loc[1, "x"] = 0.5678
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert test_func.variables == ["x"]
    assert test_func.responses == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(-2.86730552047389)
    assert initial_guess.iloc[1].f == pytest.approx(-1.8495059088635726)


def test_forrester_multi_fi_lo_kwargs():
    "Test the modified values of A, B & C for ForresterMultiFiLo test function"
    # Instantiate the test function with custom options
    from standard_evaluator.evaluators.test.forrester_multi_fi_base import ForresterMultiFiOptions
    options = ForresterMultiFiOptions(A=0.6, B=3.0, C=-1.2)
    test_func = ForresterMultiFiLo(options=options)
    
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

    # Change the variable x
    initial_guess.loc[1, "x"] = 0.45678
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert initial_guess.iloc[0].f == pytest.approx(-4.609180)
    assert initial_guess.iloc[1].f == pytest.approx(-0.866548)
    # Test that the options values are accessible
    assert test_func.lookup_option_value("A") == pytest.approx(0.6)
    assert test_func.lookup_option_value("B") == pytest.approx(3.0)
    assert test_func.lookup_option_value("C") == pytest.approx(-1.2)


def test_create_opt_problem():
    # Create an instance of the class that contains the _create_opt_problem method
    opt_problem = ForresterMultiFiLo()._create_opt_problem()

    # Check the number of variables and responses
    assert len(opt_problem.variables) == 1, "Expected 1 variables"
    assert len(opt_problem.responses) == 1, "Expected 1 responses"

    # Check the names of variables
    expected_variable_names = ["x"]
    actual_variable_names = [var.name for var in opt_problem.variables]

    assert (
        actual_variable_names == expected_variable_names
    ), f"Variable names do not match expected values"

    # Check the names of responses
    expected_response_names = ["f"]
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
    expected_constraints = []
    assert (
        opt_problem.constraints == expected_constraints
    ), f"Expected constraints {expected_constraints}, got {opt_problem.constraints}"

    # Check variable bounds and defaults
    expected_var_bounds = (0.0, 1.0)
    for var in opt_problem.variables:
        assert (
            var.bounds == expected_var_bounds
        ), f"Variable bounds do not match for variable {var}: {var.bounds} != {expected_var_bounds}"

    # Check the initial guess
    expected_initial_guess = 0.75936838
    actual_initial_guess = var.default
    assert (
        actual_initial_guess == expected_initial_guess
    ), f"Expected initial guess {expected_initial_guess}, got {actual_initial_guess}"

    # Check the description
    expected_description = r"""$$
Multifidelity analysis occurs when different analysis or simulation codes predict the same
response. Often in engineering different analysis codes are distinguished by their
computational complexity and accuracy, for example, a quick calculation may be done using
empirical equations while an expensive calculation may be cone using finite element analysis.
Mathematically, we have a greater quanitity of low-fidelity data  𝑋𝑙,𝑦𝑙  and a more accurate
but lower quantity of high-fidelity data  𝑋ℎ,𝑦ℎ . We would like to build a model that
leverages all collected data.

One method for accomplishing is to construct a correction model of the form

𝑦ℎ=𝑍𝜌𝑦𝑙+𝑍𝑑
$$"""
    assert (
        opt_problem.description.strip() == expected_description.strip()
    ), "Description does not match expected value"

    # Check the citation
    expected_citation = ""
    assert (
        opt_problem.cite == expected_citation
    ), f"Expected citation '{expected_citation}', got '{opt_problem.cite}'"

    # Test to check response values of evaluated initial guess
    from standard_evaluator.evaluators.test.forrester_multi_fi_base import ForresterMultiFiOptions
    options = ForresterMultiFiOptions(A=0.7, B=6.0, C=-4.3)
    test_func = ForresterMultiFiLo(options=options)
    initial_guess_df = test_func.initial_guess()

    # Evaluate the initial guess
    test_func(initial_guess_df)

    # Assert that the evaluated responses match the expected values
    assert initial_guess_df.iloc[0].f == pytest.approx(-8.109196821912594)
    # Test that the options values are accessible
    assert test_func.lookup_option_value("A") == pytest.approx(0.7)
    assert test_func.lookup_option_value("B") == pytest.approx(6.0)
    assert test_func.lookup_option_value("C") == pytest.approx(-4.3)

def test_forrester_multi_fi_lo_options():
    """Test the options functionality for ForresterMultiFiLo"""
    from standard_evaluator.evaluators.test.forrester_multi_fi_base import ForresterMultiFiOptions
    
    # Test default options
    test_func = ForresterMultiFiLo()
    assert test_func.lookup_option_value("A") == pytest.approx(0.2)
    assert test_func.lookup_option_value("B") == pytest.approx(5.0)
    assert test_func.lookup_option_value("C") == pytest.approx(-2.0)
    
    # Test required_options class method
    required_opts = ForresterMultiFiLo.required_options()
    assert required_opts.A == pytest.approx(0.2)
    assert required_opts.B == pytest.approx(5.0)
    assert required_opts.C == pytest.approx(-2.0)
    
    # Test required_options_names
    option_names = ForresterMultiFiLo.required_options_names()
    assert option_names == {"A", "B", "C"}
    
    # Test custom options
    custom_options = ForresterMultiFiOptions(A=1.5, B=3.5, C=0.5)
    test_func_custom = ForresterMultiFiLo(options=custom_options)
    assert test_func_custom.lookup_option_value("A") == pytest.approx(1.5)
    assert test_func_custom.lookup_option_value("B") == pytest.approx(3.5)
    assert test_func_custom.lookup_option_value("C") == pytest.approx(0.5)


def test_forrester_multi_fi_lo_options_validation():
    """Test that options validation works correctly"""
    from standard_evaluator.evaluators.test.forrester_multi_fi_base import ForresterMultiFiOptions
    
    # Test valid options
    valid_options = ForresterMultiFiOptions(A=1.0, B=2.0, C=3.0)
    test_func = ForresterMultiFiLo(options=valid_options)
    assert test_func.lookup_option_value("A") == 1.0
    
    # Test that invalid types raise validation errors
    with pytest.raises(Exception):  # Pydantic will raise ValidationError
        ForresterMultiFiOptions(A="invalid", B=2.0, C=3.0)


def test_forrester_multi_fi_lo_parameter_usage():
    """Test that the parameters are correctly used in the low-fidelity calculation"""
    from standard_evaluator.evaluators.test.forrester_multi_fi_base import ForresterMultiFiOptions
    from standard_evaluator.evaluators.test import ForresterMultiFiHi
    
    # Create test function with known parameters
    options = ForresterMultiFiOptions(A=1.0, B=0.0, C=0.0)
    test_func = ForresterMultiFiLo(options=options)
    
    # Create test data
    test_data = pd.DataFrame({"x": [0.5]})
    
    # Evaluate
    test_func(test_data)
    
    # With A=1.0, B=0.0, C=0.0, the result should be just the baseline function
    # f_lo = A * f_baseline + B * (x - 0.5)^2 + C
    # f_lo = 1.0 * f_baseline + 0.0 * 0 + 0.0 = f_baseline
    baseline_func = ForresterMultiFiHi()  # Hi-fi is just the baseline
    baseline_data = pd.DataFrame({"x": [0.5]})
    baseline_func(baseline_data)
    
    assert test_data.iloc[0].f == pytest.approx(baseline_data.iloc[0].f)