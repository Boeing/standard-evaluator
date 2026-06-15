"""Testing of the exponential tensor product test evaluator classes"""

import pytest
import numpy as np
import pandas as pd
from standard_evaluator.evaluators.test import ExponentialTensorProduct

def test_exponential_tensor_product_nind_2():
    """Test the exponential tensor product test function"""

    # Instantiate the test function

    test_func = ExponentialTensorProduct(num_independent=2)
    
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
    assert test_func.inputs == ["x0", "x1"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(1.0)
    assert initial_guess.iloc[1].f == pytest.approx(np.exp(0.5)**2)

def test_exponential_tensor_product_a_2():
    """Test the exponential tensor product test function"""

    # Instantiate the test function

    test_func = ExponentialTensorProduct(num_independent=3, parameter_a=2.0)
    
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
    assert test_func.inputs == ["x0", "x1", "x2"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(1.0)
    assert initial_guess.iloc[1].f == pytest.approx(np.exp(2.0*0.25)**3)