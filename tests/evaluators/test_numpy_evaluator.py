""" Testing of the optpytest classes """
import pandas as pd
import numpy as np

from standard_evaluator.evaluators import NumpyEvaluator
from standard_evaluator import EvaluatorInfo
from standard_evaluator.utilities import create_opt_problem


class DummyEvaluator(NumpyEvaluator):
    def eval_np(self, input_data: np.ndarray, names: list = None) -> np.ndarray:
        """Method to allow calling the function with a numpy array
        :param input_data: The sites to be evaluated (implemented as a list of list)
        :type input_data: Numpy array
        :return: 2D numpy array with all the response values for all sites evaluated
        :rtype: Numpy array
        """
        results = np.sum(input_data, axis=1)
        return results

class DummyVectorizedEvaluator(NumpyEvaluator):
    def eval_np(self, input_data: np.ndarray, names: list = None) -> np.ndarray:
        """Method to allow calling the function with a numpy array
        :param input_data: The sites to be evaluated (implemented as a list of list)
        :type input_data: Numpy array
        :return: 2D numpy array with all the response values for all sites evaluated
        :rtype: Numpy array
        """
        val = np.sum(input_data, axis=1).reshape(-1,1)
        results = np.hstack(
            (val ,val+val))
        return results

    
    
def test_numpy_evaluator():
    """Test a super simple NumpyEvaluator"""
    test_instance = DummyEvaluator(opt_problem=create_opt_problem(6, 1))
    # initial = test_instance.initial_guess()
    initial = pd.DataFrame(data={f"x{i}": [0] for i in range(6)})
    result = test_instance.eval_np(initial[test_instance.inputs].values)
    np.testing.assert_allclose(
        result,
        np.array([0.0]),
    )
    multi = [[i + j + 0.2 for i in range(test_instance.nind)] for j in range(3)]
    multi_np = np.array(multi)
    result2 = test_instance.eval_np(multi_np)
    np.testing.assert_allclose(
        result2,
        np.array([16.2, 22.2, 28.2]),
    )
    # Test using the pandas based evaluator
    test_instance(initial)
    np.testing.assert_allclose(
        initial[test_instance.outputs].values,
        np.array([[0.0]]),
    )

def test_vectorized_numpy_evaluator():
    """Test a simple NumpyEvaluator that tests array roll/unroll functionality."""
    test_instance = DummyVectorizedEvaluator(
        interface=EvaluatorInfo(name='bla', 
            inputs=[{'name': 'x', 'shape': (5,)}], 
            outputs=[{'name': 'f', 'shape': (2,)}])
    )
    expected_df = pd.DataFrame({
        'x':[np.array([7, -7, -5, 10,  3]),
             np.array([-10, 6, -8, 4, -2]),
             np.array([5, 6, 9, -2, 3])], 
        'f':[np.array([8, 16]), 
             np.array([-10, -20]), 
             np.array([21, 42])]
    })
    sites = pd.DataFrame({
        'x':[np.array([7, -7, -5, 10,  3]),
             np.array([-10, 6, -8, 4, -2]),
             np.array([5, 6, 9, -2, 3])]})
    test_instance(sites)
    assert sites.equals(expected_df)