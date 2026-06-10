"""Testing of the rosenbrock test evaluator classes"""

import pytest
import pandas as pd
from standard_evaluator.evaluators.test import Rosenbrock

def test_rosenbrock_default():
    """Test the Rosenbrock test function"""

    # Instantiate the test function

    test_func = Rosenbrock(num_independent=2)
    
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
    
    initial_guess.loc[1, "x0"] = 2.0
    initial_guess.loc[1, "x1"] = 2.0
    
    # Evaluate the initial guess
    
    test_func(initial_guess)
    
    # Check some specific responses
    assert test_func.variables == ["x0", "x1"]
    assert test_func.responses == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(0.0)
    assert initial_guess.iloc[1].f == pytest.approx(5.0)

def test_rosenbrock_nind_3():
    """Test the Rosenbrock test function"""

    # Instantiate the test function

    test_func = Rosenbrock(num_independent=3)
    
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
    
    initial_guess.loc[1, "x0"] = 2.0
    initial_guess.loc[1, "x1"] = 2.0
    initial_guess.loc[1, "x2"] = 2.0
    
    # Evaluate the initial guess
    
    test_func(initial_guess)
    
    # Check some specific responses
    assert test_func.variables == ["x0", "x1", "x2"]
    assert test_func.responses == ["f"]
    assert initial_guess.iloc[0].f == pytest.approx(0.0)
    assert initial_guess.iloc[1].f == pytest.approx(10.0)