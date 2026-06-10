"""Testing of the hyperbolic tangent tensor product test evaluator classes"""

import pytest
import numpy as np
import pandas as pd
from standard_evaluator.evaluators.test.hyperbolic_tangent_tensor_product import (
    HyperbolicTangentTensorProduct,
    HyperbolicTangentTensorProductOptions,
)

def test_hyperbolic_tangent_tensor_product_nind_2():
    """Test the hyperbolic tangent tensor product test function"""

    # Instantiate the test function

    test_func = HyperbolicTangentTensorProduct(num_independent=2)
    
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
    assert initial_guess.iloc[0].f == pytest.approx(0.0)
    assert initial_guess.iloc[1].f == pytest.approx(np.tanh(0.5)**2)
    # Test that the options values are accessible
    assert test_func.lookup_option_value("parameter_a") == pytest.approx(1.0)

def test_hyperbolic_tangent_tensor_product_a_2():
    """Test the hyperbolic tangent tensor product test function"""

    # Instantiate the test function

    test_func = HyperbolicTangentTensorProduct(
        num_independent=3,
        options=HyperbolicTangentTensorProductOptions(parameter_a=2.0),
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
    assert initial_guess.iloc[0].f == pytest.approx(0.0)
    assert initial_guess.iloc[1].f == pytest.approx(np.tanh(2.0*0.25)**3)
    # Test that the options values are accessible
    assert test_func.lookup_option_value("parameter_a") == pytest.approx(2.0)


def test_hyperbolic_tangent_tensor_product_options():
    """Test the options functionality for HyperbolicTangentTensorProduct."""
    # Test _define_options returns the correct class
    assert HyperbolicTangentTensorProduct._define_options() is HyperbolicTangentTensorProductOptions

    # Test required_options_names
    assert HyperbolicTangentTensorProduct.required_options_names() == {"parameter_a"}

    # Test default options
    test_func = HyperbolicTangentTensorProduct()
    assert test_func.lookup_option_value("parameter_a") == pytest.approx(1.0)

    # Test required_options returns defaults
    required_opts = HyperbolicTangentTensorProduct.required_options()
    assert isinstance(required_opts, HyperbolicTangentTensorProductOptions)
    assert required_opts.parameter_a == pytest.approx(1.0)

    # Test custom options with num_independent as __init__ param and parameter_a as option
    custom_func = HyperbolicTangentTensorProduct(
        num_independent=4,
        options=HyperbolicTangentTensorProductOptions(parameter_a=3.5),
    )
    assert custom_func.lookup_option_value("parameter_a") == pytest.approx(3.5)


def test_hyperbolic_tangent_tensor_product_options_validation():
    """Test that options validation works correctly."""
    # Test __init__ clamps num_independent < 1 to 1
    test_func = HyperbolicTangentTensorProduct(num_independent=0)
    assert test_func._num_independent == 1

    # Test __init__ clamps num_dependent < 1 to 1
    test_func = HyperbolicTangentTensorProduct(num_dependent=0)
    assert test_func._num_dependent == 1

    # Test invalid option key raises KeyError
    with pytest.raises(KeyError):
        HyperbolicTangentTensorProduct().lookup_option_value("nonexistent_option")

    # Test wrong options type raises TypeError
    with pytest.raises(TypeError):
        HyperbolicTangentTensorProduct(options={"parameter_a": 3.0})

    with pytest.raises(TypeError):
        HyperbolicTangentTensorProduct(options="invalid")

    # Test invalid types raise validation errors
    with pytest.raises(Exception):
        HyperbolicTangentTensorProductOptions(parameter_a="invalid")
