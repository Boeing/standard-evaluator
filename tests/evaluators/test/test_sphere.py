"""Testing of the sphere test evaluator classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import Sphere

def test_sphere():
    """Test the sphere test function"""

    # Instantiate the test function

    test_func = Sphere(num_independent=3)
    
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
    
    initial_guess.loc[1, "x0"] = 2.0
    initial_guess.loc[1, "x1"] = 2.0
    initial_guess.loc[1, "x2"] = 2.0
    
    # Evaluate the initial guess
    
    test_func(initial_guess)
    
    # Check some specific responses
    assert test_func.inputs == ["x0", "x1", "x2"]
    assert test_func.outputs == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(3.0)
    assert initial_guess.iloc[1].f == pytest.approx(12.0)