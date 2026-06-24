import numpy as np
import pandas as pd
import pytest
from typing import Callable

from standard_evaluator import EvaluatorInfo, ArrayVariable, FloatVariable
from standard_evaluator.evaluators import PyEvaluator
from standard_evaluator.utilities import create_df_from_evaluator_info, unroll_data_frame_using_variables, unroll_names_using_variables, roll_data_frame_using_variables
from standard_evaluator.problem import OptProblem


class ClassEvaluator:
    op_map = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "%": lambda x, y: x % y,
    }

    def __init__(self, ops, variables) -> None:
        self.ops = ops
        self.variables = variables

    def __call__(self, sites) -> None:
        self.eval(sites)

    def eval(self, sites) -> None:
        sites["f0"] = sites["x0"]
        for i in range(1, len(self.variables)):
            op = self.ops[i % len(self.ops) - 1]
            sites["f0"] = self.op_map[op](sites["f0"], sites[f"x{i}"])
        sites["f1"] = sites[self.variables].prod(axis=1)


def evaluate(sites):
    variables = ["x0", "x1", "x2", "x3", "x4"]
    sites["f0"] = sites[variables].sum(axis=1)
    sites["f1"] = sites[variables].prod(axis=1)


@pytest.fixture
def opt_problem():
    return OptProblem(
        name="test_problem",
        variables=[
            FloatVariable(name=f"x{i}", bounds=[-float("inf"), float("inf")])
            for i in range(5)
        ],
        responses=[
            FloatVariable(name=f"f{i}") for i in range(2)
        ],
        objectives=[],
    )



@pytest.fixture
def sites():
    return pd.DataFrame(
        data={
            "x0": [7, -10, 5],
            "x1": [-7, 6, 6],
            "x2": [-5, -8, 9],
            "x3": [10, 4, -2],
            "x4": [3, -2, 3],
        }
    )


@pytest.fixture
def expected_sum():
    return pd.DataFrame(
        data={
            "x0": [7, -10, 5],
            "x1": [-7, 6, 6],
            "x2": [-5, -8, 9],
            "x3": [10, 4, -2],
            "x4": [3, -2, 3],
            "f0": [8, -10, 21],
            "f1": [7350, -3840, -1620],
        }
    )


@pytest.fixture
def expected_combo():
    return pd.DataFrame(
        data={
            "x0": [7, -10, 5],
            "x1": [-7, 6, 6],
            "x2": [-5, -8, 9],
            "x3": [10, 4, -2],
            "x4": [3, -2, 3],
            "f0": [18, 0, -3],
            "f1": [7350, -3840, -1620],
        }
    )

@pytest.fixture
def evaluator_info() -> EvaluatorInfo:

    return EvaluatorInfo(
        name='my info',
        inputs=[ArrayVariable(name='x', default=np.array([[1],[1.0], [.4]])),
                FloatVariable(name='w', default=3.9)],
        outputs=[ArrayVariable(name='sum', shape=(3,1)), 
                 ArrayVariable(name='scalar_prod', shape=(3,1))])

@pytest.fixture
def rolled_sites(evaluator_info: EvaluatorInfo) -> pd.DataFrame:
    return create_df_from_evaluator_info(
        evaluator_info,
        [[np.array([[2],[1],[4]]), 5], [np.array([[.1],[3],[.2]]), 2]],
        evaluator_info.inputs)

@pytest.fixture
def expected_df(evaluator_info: EvaluatorInfo) -> pd.DataFrame:
    np_expected = np.array([
        [np.array([[2], [1], [4]]), 5.0, np.array([[4.], [2.], [8.]]), np.array([[10.], [5.], [20.]])],
        [np.array([[0.1], [3.], [0.2]]), 2.0, np.array([[0.2], [6.], [0.4]]), np.array([[0.2], [6.], [0.4]])]
        ], dtype=object)
    return create_df_from_evaluator_info(evaluator_info, np_expected)

@pytest.fixture
def vector_func() -> Callable[[pd.DataFrame], None]:
    def rolled_eval_func(rolled_sites:pd.DataFrame):   
        rolled_sites['sum'] = rolled_sites.x + rolled_sites.x    
        rolled_sites['scalar_prod'] = rolled_sites.x.multiply(rolled_sites.w)
    return rolled_eval_func
    

def test_init(opt_problem):
    with pytest.raises(TypeError, match="must be a callable!"):
        eval = PyEvaluator(72, opt_problem=opt_problem)
    with pytest.raises(TypeError, match="must be a callable!"):
        eval = PyEvaluator([evaluate], opt_problem=opt_problem)
    with pytest.raises(TypeError, match="must be a callable!"):
        eval = PyEvaluator("evaluate", opt_problem=opt_problem)

    eval = PyEvaluator(evaluate, opt_problem=opt_problem)

    assert eval._func is evaluate


# Evaluation function is a normal python function
def test_evaluate_function(
    opt_problem, sites: pd.DataFrame, expected_sum: pd.DataFrame
):
    eval = PyEvaluator(evaluate, opt_problem=opt_problem)

    ret = eval(sites)

    assert ret is None
    assert (sites == expected_sum).all(axis=None)


# Evaluation function is a class method
def test_evaluate_method(
    opt_problem,
    sites: pd.DataFrame,
    expected_sum: pd.DataFrame,
    expected_combo: pd.DataFrame,
):
    # Create tester class that only adds
    variables = [f"x{i}" for i in range(5)]
    tester = ClassEvaluator(["+"], variables)
    eval = PyEvaluator(tester.eval, opt_problem=opt_problem)

    # Evaluate and check sum
    sites_copy = sites.copy()
    eval(sites_copy)
    assert (sites_copy == expected_sum).all(axis=None)

    # Change the operations that the tester class uses
    sites_copy = sites.copy()
    tester.ops = ["*", "+", "%"]
    eval(sites_copy)
    assert (sites_copy == expected_combo).all(axis=None)


# Evaluation function is a class with a __call__ method
def test_evaluate_class(
    opt_problem,
    sites: pd.DataFrame,
    expected_sum: pd.DataFrame,
    expected_combo: pd.DataFrame,
):
    # Create tester class that only adds
    variables = [f"x{i}" for i in range(5)]
    tester = ClassEvaluator(["+"], variables)
    eval = PyEvaluator(tester, opt_problem=opt_problem)

    # Evaluate and check sum
    sites_copy = sites.copy()
    eval(sites_copy)
    assert (sites_copy == expected_sum).all(axis=None)

    # Change the operations that the tester class uses
    sites_copy = sites.copy()
    tester.ops = ["*", "+", "%"]
    eval(sites_copy)
    assert (sites_copy == expected_combo).all(axis=None)

# Test eval_list functionality using evaluator info
def test_eval_list(
    evaluator_info: EvaluatorInfo,
    vector_func: Callable[[pd.DataFrame], None],
    rolled_sites: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    
    sites = rolled_sites
    vector_eval = PyEvaluator(vector_func, interface=evaluator_info)
    expected_results = expected_df


    # Test rolled input as well as unrolled input
    rolled_sites_np = sites.to_numpy()
    unrolled_sites_np = unroll_data_frame_using_variables(sites, evaluator_info.inputs)


    
    # Test that rolled input works
    results_np = vector_eval.eval_list(rolled_sites_np)
    results_np = np.hstack([rolled_sites_np, results_np])
    results = create_df_from_evaluator_info(evaluator_info, results_np) 
    assert results.equals(expected_results)

    # Test that unrolled input works
    results_np = vector_eval.eval_list(unrolled_sites_np)
    results_np = np.hstack([unrolled_sites_np, results_np])
    names = unroll_names_using_variables(evaluator_info.inputs + evaluator_info.outputs)
    unrolled_df = pd.DataFrame(results_np, columns=names) 
    results = roll_data_frame_using_variables(unrolled_df, evaluator_info.inputs + evaluator_info.outputs) 
    assert results.equals(expected_results)

    expected_results.drop(columns=['scalar_prod'], inplace=True)

    result_np = vector_eval.eval_list(rolled_sites_np, ['sum'])
    result_np = np.hstack([rolled_sites_np, result_np])
    results = create_df_from_evaluator_info(evaluator_info, result_np, names=[v.name for v in evaluator_info.inputs] + ['sum']) 
    # Tests that eval_list works subset of names given
    assert results.equals(expected_results)