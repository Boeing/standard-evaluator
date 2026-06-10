"""Testing of the cosine tensor product test evaluator classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test.cosine_tensor_product import (
    CosineTensorProduct,
    CosineTensorProductOptions,
)

def test_cosine_tensor_product_nind_2():
    """Test the cosine tensor product test function"""

    # Instantiate the test function

    test_func = CosineTensorProduct(num_independent=2)
    
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
    
    # Change x0, x1
    
    initial_guess.loc[1, "x0"] = 0.5
    initial_guess.loc[1, "x1"] = 0.5
    
    # Evaluate the initial guess
    
    test_func(initial_guess)
    
    # Check some specific responses
    assert test_func.variables == ["x0", "x1"]
    assert test_func.responses == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(1.0)
    assert initial_guess.iloc[1].f == pytest.approx(0.0)
    # Test that the options values are accessible
    assert test_func.lookup_option_value("parameter_a") == pytest.approx(1.0)

def test_cosine_tensor_product_a_2():
    """Test the cosine tensor product test function"""

    # Instantiate the test function

    test_func = CosineTensorProduct(
        num_independent=3,
        options=CosineTensorProductOptions(parameter_a=2.0),
    )
    
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
    
    # Change x0, x1, and x2
    
    initial_guess.loc[1, "x0"] = 0.25
    initial_guess.loc[1, "x1"] = 0.25
    initial_guess.loc[1, "x2"] = 0.25
    
    # Evaluate the initial guess
    
    test_func(initial_guess)
    
    # Check some specific responses
    assert test_func.variables == ["x0", "x1", "x2"]
    assert test_func.responses == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(1.0)
    assert initial_guess.iloc[1].f == pytest.approx(0.0)
    # Test that the options values are accessible
    assert test_func.lookup_option_value("parameter_a") == pytest.approx(2.0)


def test_cosine_tensor_product_options():
    """Test the options functionality for CosineTensorProduct."""
    # Test _define_options returns the correct class
    assert CosineTensorProduct._define_options() is CosineTensorProductOptions

    # Test required_options_names
    assert CosineTensorProduct.required_options_names() == {"parameter_a"}

    # Test default options
    test_func = CosineTensorProduct()
    assert test_func.lookup_option_value("parameter_a") == pytest.approx(1.0)

    # Test required_options returns defaults
    required_opts = CosineTensorProduct.required_options()
    assert isinstance(required_opts, CosineTensorProductOptions)
    assert required_opts.parameter_a == pytest.approx(1.0)

    # Test custom options with num_independent as __init__ param and parameter_a as option
    custom_func = CosineTensorProduct(
        num_independent=4,
        options=CosineTensorProductOptions(parameter_a=3.5),
    )
    assert custom_func.lookup_option_value("parameter_a") == pytest.approx(3.5)


def test_cosine_tensor_product_options_validation():
    """Test that options validation works correctly."""
    # Test __init__ clamps num_independent < 1 to 1
    test_func = CosineTensorProduct(num_independent=0)
    assert test_func._num_independent == 1

    # Test __init__ clamps num_dependent < 1 to 1
    test_func = CosineTensorProduct(num_dependent=0)
    assert test_func._num_dependent == 1

    # Test invalid option key raises KeyError
    with pytest.raises(KeyError):
        CosineTensorProduct().lookup_option_value("nonexistent_option")

    # Test wrong options type raises TypeError
    with pytest.raises(TypeError):
        CosineTensorProduct(options={"parameter_a": 3.0})

    with pytest.raises(TypeError):
        CosineTensorProduct(options="invalid")

    # Test invalid types raise validation errors
    with pytest.raises(Exception):
        CosineTensorProductOptions(parameter_a="invalid")
