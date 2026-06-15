import math
import pathlib
import sqlite3 as sql
import os

import numpy as np
import pandas as pd
import pytest

from pydantic import BaseModel, conint, confloat, Field

from standard_evaluator.evaluators import Evaluator
import standard_evaluator as se
from standard_evaluator import (
    OptProblem,
    IntVariable,
    FloatVariable,
    CategoricalVariable,
)
from standard_evaluator.evaluators.test import ConstrainedBetts


class MyOptions(BaseModel):
    alpha: float = 1.0
    beta: confloat(ge=0.0, le=100.0) = 10.0
    max_iter: conint(gt=0) = 50
    label: str = "default"


class DummyEvaluator(Evaluator):
    def _evaluate(self, sites: pd.DataFrame) -> None:
        sites["f"] = sites["x1"] + sites["x2"]
        sites["c1"] = sites["x1"] ** 2 + sites["x2"] ** 2 - 2
        sites["c2"] = -sites["x2"]

        # Assign values to categorical response
        sites.loc[sites["f"] > 1, "cat_res"] = "f > 1"
        sites.loc[(sites["f"] >= -1) & (sites["f"] <= 1), "cat_res"] = "-1 <= f <= 1"
        sites.loc[sites["f"] < -1, "cat_res"] = "f < -1"


class DummyEvaluatorWithOptions(Evaluator):
    @classmethod
    def _define_options(cls):
        return MyOptions

    def _evaluate(self, sites: pd.DataFrame) -> None:
        sites["f"] = sites["x1"] * self._options.alpha


class FailEvaluator(Evaluator):
    # Missing the 'cat_res' response. So should throw an error
    def _evaluate(self, sites: pd.DataFrame) -> None:
        sites["f"] = sites["x1"] + sites["x2"]
        sites["c1"] = sites["x1"] ** 2 + sites["x2"] ** 2 - 2
        sites["c2"] = -sites["x2"]

class DummyArrayEvaluator(Evaluator):
    def _evaluate(self, sites: pd.DataFrame) -> None:
        pass

@pytest.fixture
def opt_problem() -> OptProblem:
    new_prob = se.utilities.create_opt_problem(num_independent=2, num_dependent=4)

    var_names = ["x1", "x2"]

    # Define default values
    defaults = [1.0, 2.3]
    var_bounds = ([-5.0, 5.0], [-5.0, 3.0])
    for var, local_default, local_bound, name in zip(
        new_prob.variables, defaults, var_bounds, var_names
    ):
        var.bounds = local_bound
        var.default = local_default
        var.name = name

    resp_names = ["f", "c1", "c2", "cat_res"]
    resp_bounds = (
        [np.inf, np.inf],
        [-np.inf, 0.0],
        [-np.inf, 0.0],
        ["f > 1", "-1 <= f <= 1", "f < -1"],
    )

    # Assigning float class type to responses 0, 1, and 2
    for idx in range(3):
        local_bound = resp_bounds[idx]
        name = resp_names[idx]
        new_prob.responses[idx] = FloatVariable(
            name=name,
            bounds=local_bound,
            units=None,
            description="",
            options={},
            class_type="float",
        )

    # Assigning Categorical class type to response res
    idx = 3
    local_bound = resp_bounds[idx]
    name = resp_names[idx]
    new_prob.responses[idx] = CategoricalVariable(
        name=name,
        bounds=local_bound,
        units=None,
        description="",
        options={},
        class_type="cat",
    )

    # Define objectives and constraints
    new_prob.objectives = ["f"]
    new_prob.constraints = ["c1", "c2"]

    # Re-validate the OptProblem
    new_prob = OptProblem(**new_prob.model_dump())

    return new_prob

@pytest.fixture
def ndarray_opt_problem():
    """Construct an OptProblem with scalar and array variables/responses of various dims."""
    # Variables: scalar A, array C (2,), array D (2,2), array F (2,3,4)
    A = se.FloatVariable(name="A", bounds=(-np.inf, np.inf))
    C = se.ArrayVariable(name="C", shape=(2,), bounds=(-np.inf, np.inf))
    D = se.ArrayVariable(name="D", shape=(2, 2), bounds=(-np.inf, np.inf))
    F = se.ArrayVariable(name="F", shape=(2, 3, 4), bounds=(-np.inf, np.inf))

    # Responses: scalar B, array E (3,4), array G (3,3,5)
    B = se.FloatVariable(name="B", bounds=(-np.inf, np.inf))
    E = se.ArrayVariable(name="E", shape=(3, 4), bounds=(-np.inf, np.inf))
    G = se.ArrayVariable(name="G", shape=(3, 3, 5), bounds=(-np.inf, np.inf))

    opt = se.OptProblem(
        name="map_test_problem",
        variables=[A, C, D, F],
        responses=[B, E, G],
        objectives=["E[0,0]"],
        constraints=["B", "G[0,0,0]", "G[0,0,1]"],
    )
    return opt


@pytest.fixture
def sites() -> pd.DataFrame:
    return pd.DataFrame(
        data={"x1": [1, 1, -1], "x2": [-1, 1, -1], "misc_col": ["a", "b", "c"]}
    )


@pytest.fixture
def eval_sites() -> pd.DataFrame:
    sites = pd.DataFrame(
        data={
            "x1": [1, 1, -1],
            "x2": [-1, 1, -1],
            "f": [0, 2, -2],
            "c1": [0, 0, 0],
            "c2": [1, -1, 1],
            "cat_res": ["-1 <= f <= 1", "f > 1", "f < -1"],
            "misc_col": ["a", "b", "c"],
        }
    )
    return sites.astype(
        {
            "x1": "float",
            "x2": "float",
            "f": "float",
            "c1": "float",
            "c2": "float",
            "cat_res": pd.CategoricalDtype(
                ["f > 1", "-1 <= f <= 1", "f < -1"], ordered=True
            ),
        }
    )

def assert_opt_problems_equivalent(a: OptProblem, b: OptProblem) -> None:
    """
    Assert that two ``OptProblem`` instances are equivalent.

    This helper compares selected top-level fields, variable and response fields,
    derived maps, and partials-related bookkeeping. Missing optional attributes
    are treated as ``None``.

    :param a: First ``OptProblem`` instance.
    :type a: OptProblem

    :param b: Second ``OptProblem`` instance.
    :type b: OptProblem

    :raises AssertionError: If any compared attribute differs.
    """


    # Top-level fields
    for attr in ["name", "class_type", "objectives", "constraints", "description", "cite", "options"]:
        assert getattr(a, attr, None) == getattr(b, attr, None), f"Mismatch in {attr}"

    # Compare variables and responses by name and any shared optional fields
    for attr in ["variables", "responses"]:
        a_items = getattr(a, attr, [])
        b_items = getattr(b, attr, [])
        assert len(a_items) == len(b_items), f"Mismatch in length of {attr}"

        for i, (a_item, b_item) in enumerate(zip(a_items, b_items)):
            # Always compare these if present
            for field in ["name", "class_type", "bounds", "default"]:
                assert getattr(a_item, field, None) == getattr(b_item, field, None), (
                    f"Mismatch in {attr}[{i}].{field}"
                )

            # Optional attributes; missing is treated as None
            for field in ["shape", "shift", "scale", "units", "description", "options"]:
                assert getattr(a_item, field, None) == getattr(b_item, field, None), (
                    f"Mismatch in {attr}[{i}].{field}"
                )

    # Compare derived maps if both are present
    if a.var_map is None or b.var_map is None:
        assert a.var_map is b.var_map, "Mismatch in var_map presence"
    else:
        pd.testing.assert_frame_equal(
            a.var_map.reset_index(drop=True),
            b.var_map.reset_index(drop=True),
            check_dtype=False,
        )

    if a.res_map is None or b.res_map is None:
        assert a.res_map is b.res_map, "Mismatch in res_map presence"
    else:
        pd.testing.assert_frame_equal(
            a.res_map.reset_index(drop=True),
            b.res_map.reset_index(drop=True),
            check_dtype=False,
        )

    # Compare partials-related arrays if both objects have them
    array_attrs = [
        "flat_partials_res_indices",
        "grad_cols",
        "jac_cols",
        "free_flat_var_mask",
        "free_flat_var_positions",
        "full_to_free_var_index",
    ]
    for attr in array_attrs:
        a_val = getattr(a, attr, None)
        b_val = getattr(b, attr, None)
        if a_val is None or b_val is None:
            assert a_val is b_val, f"Mismatch in {attr} presence"
        else:
            np.testing.assert_array_equal(a_val, b_val, err_msg=f"Mismatch in {attr}")

    scalar_attrs = [
        "num_partials_responses",
        "num_objs_total",
        "num_cons_total",
        "num_flat_vars",
    ]
    for attr in scalar_attrs:
        assert getattr(a, attr, None) == getattr(b, attr, None), f"Mismatch in {attr}"


# Make sure abstract class can't be instatiated
def test_abstract():
    with pytest.raises(TypeError):
        Evaluator()


def test_init(opt_problem: OptProblem):
    # Initialize evaluator with opt_problem
    eval = DummyEvaluator(opt_problem=opt_problem)

    # Check responses, objectives, constraints
    eval_resp_names = [resp.name for resp in eval.opt_problem.responses]
    opt_resp_names = [resp.name for resp in opt_problem.responses]
    assert eval_resp_names == opt_resp_names
    assert eval.opt_problem.objectives == opt_problem.objectives
    assert eval.opt_problem.constraints == opt_problem.constraints

    # Check variables info
    eval_vars = {var.name: var for var in eval.opt_problem.variables}
    opt_vars = {var.name: var for var in opt_problem.variables}
    for var_name, var_obj in eval_vars.items():
        opt_var_obj = opt_vars[var_name]
        assert var_obj.class_type == opt_var_obj.class_type
        assert var_obj.bounds == opt_var_obj.bounds
        assert var_obj.default == opt_var_obj.default

    # Check variable and response names lists
    assert eval.inputs == [var.name for var in opt_problem.variables]
    assert eval.outputs == [resp.name for resp in opt_problem.responses]

    # Check number of independent and dependent variables
    assert eval.nind == len(opt_problem.variables)
    assert eval.ndep == len(opt_problem.responses)

    # Check specific default values for variables "x1" and "x2"
    x1_var = next(var for var in opt_problem.variables if var.name == "x1")
    x2_var = next(var for var in opt_problem.variables if var.name == "x2")
    assert x1_var.default == 1.0
    assert x2_var.default == 2.3

    # Test creating evaluator with opt_problem from create_opt_problem
    nind = 5
    ndep = 2
    eval2 = DummyEvaluator(opt_problem=se.utilities.create_opt_problem(nind, ndep))
    assert hasattr(eval2.opt_problem, "variables")
    assert hasattr(eval2.opt_problem, "responses")
    assert hasattr(eval2.opt_problem, "objectives")
    assert hasattr(eval2.opt_problem, "constraints")
    assert eval2.nind == nind
    assert eval2.ndep == ndep

    # Check variables in eval2
    var_names = [var.name for var in eval2.opt_problem.variables]
    for i in range(nind):
        assert f"x{i}" in var_names
        for var2 in eval2.opt_problem.variables:
            if var2.name == f"x{i}":
                assert var2.bounds == (-float("inf"), float("inf"))
                assert var2.class_type == "float"
                assert var2.default == 0.0
                # Add checks for scale and shift if available

    # Check responses in eval2
    resp_names = [resp.name for resp in eval2.opt_problem.responses]
    for i in range(ndep):
        assert f"f{i}" in resp_names
        for resp2 in eval2.opt_problem.responses:
            if resp2.name == f"f{i}":
                assert resp2.bounds == (-float("inf"), float("inf"))
                assert resp2.class_type == "float"
                # Add checks for scale and shift if available

    assert eval2.opt_problem.objectives == ["f0"]
    assert eval2.opt_problem.constraints == []

    # Check evaluator name
    assert eval.name == "DummyEvaluator"

    eval_named = DummyEvaluator(name="the name", opt_problem=opt_problem)
    assert eval_named.name == "DummyEvaluator_the name"

    # Create variables
    variables = [
        IntVariable(name="ivar", bounds=[-10, 15], class_type="int"),
        FloatVariable(name="fvar", bounds=[-2.5, float("inf")], class_type="float"),
        CategoricalVariable(
            name="cvar", bounds=["cat1", "cat2", "cat3"], class_type="cat"
        ),
    ]

    # Create responses
    responses = [
        IntVariable(name="ires", bounds=None, class_type="int"),
        FloatVariable(name="fres", bounds=None, class_type="float"),
        CategoricalVariable(
            name="cres", bounds=["r1", "r2", "r3", "r4"], class_type="cat"
        ),
    ]

    custom_opt_problem = OptProblem(
        variables=variables, responses=responses, objectives=[], constraints=[]
    )

    eval = DummyEvaluator(opt_problem=custom_opt_problem)

    assert len(eval._dtypes) == eval.nind + eval.ndep
    assert eval._dtypes["ivar"] == "Int64"
    assert eval._dtypes["fvar"] == "float64"
    assert eval._dtypes["cvar"] == pd.CategoricalDtype(
        ["cat1", "cat2", "cat3"], ordered=True
    )
    assert eval._dtypes["ires"] == "Int64"
    assert eval._dtypes["fres"] == "float64"
    assert eval._dtypes["cres"] == pd.CategoricalDtype(
        ["r1", "r2", "r3", "r4"], ordered=True
    )


def test_call(opt_problem: OptProblem, sites: pd.DataFrame, eval_sites: pd.DataFrame):
    eval = DummyEvaluator(opt_problem=opt_problem)

    # Need to pass dataframe
    with pytest.raises(TypeError):
        eval([[1, -1], [1, 1], [-1, -1]])

    # Missing columns
    # Variables missing
    with pytest.raises(ValueError):
        eval(pd.DataFrame(data={"x1": [1, 1, -1], "invalid": [-1, 1, -1]}))

    faileval = FailEvaluator(opt_problem=opt_problem)
    with pytest.raises(ValueError):
        faileval(sites)

    # Make sure correct values were evaluated
    ret = eval(sites)

    # Check that nothing is returned
    assert ret is None
    # Check that all columns are there
    assert len(sites.columns) == len(eval_sites.columns)
    # Check that columns are the same, dtypes are the same, and evaluated values
    # are as expected
    for col in sites.columns:
        assert col in eval_sites.columns
        assert sites.dtypes[col] == eval_sites.dtypes[col]
        assert (sites[col] == eval_sites[col]).all()


def test_eval_np(
    opt_problem: OptProblem, sites: pd.DataFrame, eval_sites: pd.DataFrame
):
    eval = DummyEvaluator(opt_problem=opt_problem)
    var_names = [var.name for var in opt_problem.variables]
    site_np = sites[var_names].to_numpy()

    # Extract response names from the list of response objects
    resp_names = [resp.name for resp in opt_problem.responses]
    # All responses
    result = eval.eval_np(site_np)
    check = eval_sites[resp_names].to_numpy()
    # Check float responses
    assert np.allclose(result[:, :3].astype(float), check[:, :3].astype(float), rtol=1e-8, atol=1e-12)
    # Check string responses
    assert np.array_equal(result[:, 3].astype(str), check[:, 3].astype(str))

    # # Subset of responses
    result = eval.eval_np(site_np, names=["c1", "cat_res"])
    check = eval_sites[["c1", "cat_res"]].to_numpy()
    # Check float responses
    assert np.allclose(result[:, :1].astype(float), check[:, :1].astype(float), rtol=1e-8, atol=1e-12)
    # Check string responses
    assert np.array_equal(result[:, 1].astype(str), check[:, 1].astype(str))

    # # Single response
    # result = eval.eval_np(site_np, names=["f"])
    result = eval.eval_np(site_np, names=["f"])
    check = eval_sites[["f"]].to_numpy()
    # Check float responses
    assert np.allclose(result[:, :1].astype(float), check[:, :1].astype(float), rtol=1e-8, atol=1e-12)


def test_eval_list(
    opt_problem: OptProblem, sites: pd.DataFrame, eval_sites: pd.DataFrame
):
    eval = DummyEvaluator(opt_problem=opt_problem)

    var_names = [var.name for var in opt_problem.variables]
    site_list = sites[var_names].to_numpy().tolist()

    # Extract response names from the list of response objects
    resp_names = [resp.name for resp in opt_problem.responses]

    # All responses
    result = eval.eval_list(site_list)
    assert result == eval_sites[resp_names].to_numpy().tolist()

    # Subset of responses are return in order
    result = eval.eval_list(site_list, names=["cat_res", "f"])
    assert result == eval_sites[["f", "cat_res"]].to_numpy().tolist()

    # Single response
    result = eval.eval_list(site_list, names=["c2"])
    assert result == eval_sites[["c2"]].to_numpy().tolist()


def test_comp_cost(opt_problem: OptProblem):
    # Set value through init
    eval = DummyEvaluator(opt_problem=opt_problem)
    assert eval.comp_cost == 100

    eval = DummyEvaluator(opt_problem=opt_problem, comp_cost=198)
    assert eval.comp_cost == 198

    eval = DummyEvaluator(opt_problem=opt_problem, comp_cost=29.4)
    assert eval.comp_cost == 29.4

    # Set value directly
    eval = DummyEvaluator(opt_problem=opt_problem)

    eval.comp_cost = 45
    assert eval.comp_cost == 45

    eval.comp_cost = 845.1
    assert eval.comp_cost == 845.1

    # Test error checking
    with pytest.raises(TypeError, match="must be a numeric value"):
        DummyEvaluator(opt_problem=opt_problem, comp_cost="12")
    with pytest.raises(TypeError, match="must be a numeric value"):
        eval.comp_cost = [3, 7, 26]

    with pytest.raises(ValueError, match="must be non-negative"):
        DummyEvaluator(opt_problem=opt_problem, comp_cost=-12)
    with pytest.raises(ValueError, match="must be non-negative"):
        DummyEvaluator(opt_problem=opt_problem, comp_cost=-326.3)


def test_initial_guess_default_site(opt_problem: OptProblem):
    opt_problem = se.utilities.create_opt_problem(num_independent=5, num_dependent=1)
    var_names = ["x1", "x2", "x3", "x4", "x5"]

    # Define var bounds
    var_bounds = (
        [-np.inf, np.inf],
        [-4, np.inf],
        [-np.inf, 19],
        [-1, 11],
        ["a", "b", "c", "d"],
    )

    # Assigning float class type to variables 0, 1, 2 and 3
    for idx in range(4):
        local_bound = var_bounds[idx]
        name = var_names[idx]
        opt_problem.variables[idx] = FloatVariable(
            name=name,
            bounds=local_bound,
            units=None,
            description="",
            options={},
            class_type="float",
        )

    # Assigning Categorical class type to response res
    idx = 4
    local_bound = var_bounds[idx]
    name = var_names[idx]
    opt_problem.variables[idx] = CategoricalVariable(
        name=name,
        bounds=local_bound,
        units=None,
        description="",
        options={},
        class_type="cat",
    )

    resp_name = ["f"]
    for resp, name in zip(opt_problem.responses, resp_name):
        resp.name = name

    # Define objectives and constraints
    opt_problem.objectives = ["f"]
    opt_problem.constraints = []

    expected = pd.DataFrame(
        data={"x1": [0], "x2": [-4], "x3": [19], "x4": [5], "x5": ["a"]}
    )
    eval = DummyEvaluator(opt_problem=opt_problem)

    guess = eval.initial_guess()
    assert (guess == expected).all(axis=None)

    # Also check that the default site is as expected.
    assert (eval.default_site() == expected).all(axis=None)


def test_store_default(opt_problem: OptProblem):
    opt_problem = se.utilities.create_opt_problem(num_independent=5, num_dependent=1)
    var_names = ["x1", "x2", "x3", "x4", "x5"]

    # Define default values
    var_defaults = [0, -4, 19, 5, "a"]
    var_bounds = (
        [-np.inf, np.inf],
        [-4, np.inf],
        [-np.inf, 19],
        [-1, 11],
        ["a", "b", "c", "d"],
    )

    # Assigning float class type to variables 0, 1, 2 and 3
    for idx in range(4):
        local_bound = var_bounds[idx]
        name = var_names[idx]
        local_default = var_defaults[idx]
        opt_problem.variables[idx] = FloatVariable(
            name=name,
            bounds=local_bound,
            default=local_default,
            units=None,
            description="",
            options={},
            class_type="float",
        )

    # Assigning Categorical class type to response res
    idx = 4
    local_bound = var_bounds[idx]
    name = var_names[idx]
    local_default = var_defaults[idx]
    opt_problem.variables[idx] = CategoricalVariable(
        name=name,
        bounds=local_bound,
        units=None,
        description="",
        options={},
        class_type="cat",
    )

    resp_name = ["f"]

    for resp, name in zip(opt_problem.responses, resp_name):
        resp.name = name

    # Define objectives and constraints
    opt_problem.objectives = ["f"]
    opt_problem.constraints = []

    expected = {"x1": 0, "x2": -4, "x3": 19, "x4": 5, "x5": "a"}
    eval = DummyEvaluator(opt_problem=opt_problem)

    # Make sure only defaults were added
    assert eval.opt_problem.responses == opt_problem.responses
    assert eval.opt_problem.objectives == opt_problem.objectives
    assert eval.opt_problem.constraints == opt_problem.constraints
    for var, idx in zip(eval.opt_problem.variables, range(5)):
        assert var.bounds == opt_problem.variables[idx].bounds
        assert var.default == expected[var.name]

    # Check the class_type of the first three variables (should be float)
    for idx in range(4):
        assert (
            opt_problem.variables[idx].class_type == "float"
        ), f"Variable x[{idx}] should be of class type 'float'"

    assert (
        opt_problem.variables[4].class_type == "cat"
    ), f"Variable x[{idx}] should be of class type 'categorical'"

    # Define default values
    var2_defaults = [-37, -4, 19, 9.5, "b"]

    expected_defaults = {"x1": -37, "x2": -4, "x3": 19, "x4": 9.5, "x5": "b"}
    eval = DummyEvaluator(opt_problem=opt_problem)

    for idx in range(4):
        local_bound = var_bounds[idx]
        name = var_names[idx]
        local_default = var2_defaults[idx]
        opt_problem.variables[idx] = FloatVariable(
            default=local_default,
            name=name,
            bounds=local_bound,
            units=None,
            description="",
            options={},
            class_type="float",
        )

    # Assigning Categorical class type to response res
    idx = 4
    local_bound = var_bounds[idx]
    name = var_names[idx]
    local_default = var2_defaults[idx]
    opt_problem.variables[idx] = CategoricalVariable(
        default=local_default,
        name=name,
        bounds=local_bound,
        units=None,
        description="",
        options={},
        class_type="cat",
    )

    for var2, idx in zip(eval.opt_problem.variables, range(4)):
        assert var2.default == expected_defaults[var2.name]


def test_logging(
    opt_problem: OptProblem, sites: pd.DataFrame, eval_sites: pd.DataFrame
):
    # Create evaluator with logging
    eval = DummyEvaluator(opt_problem=opt_problem, logging=True)

    # Make sure the get_log function is accessible
    assert hasattr(eval, "get_log")
    assert not hasattr(eval, "get_cache")
    assert isinstance(eval.get_log(), pd.DataFrame)
    assert len(eval.get_log()) == 0

    # Evaluate some sites
    eval(sites)
    # Get the log
    log = eval.get_log()
    # Make sure everything in the log is as expected
    assert (log.loc[:, eval_sites.columns] == eval_sites).all(axis=None)
    assert (log["call_num"] == 1).all()

    # Shuffle the sites and evaluate them
    new_sites = sites.copy().sample(frac=1, random_state=14).reset_index(drop=True)
    eval(new_sites)
    # Get new log
    log = eval.get_log()
    expected_log = pd.concat((sites, new_sites), ignore_index=True)
    # Make sure log is as expected
    assert (log.loc[:, expected_log.columns] == expected_log).all(axis=None)
    assert (log["call_num"] == [1] * len(new_sites) + [2] * len(new_sites)).all()


def test_caching(
    opt_problem: OptProblem, sites: pd.DataFrame, eval_sites: pd.DataFrame, tmp_path: pathlib.Path
):
    cache_file_name = tmp_path / "test.db"
    if os.path.exists(cache_file_name):
        os.remove(cache_file_name)

    eval = DummyEvaluator(opt_problem=opt_problem, cache=cache_file_name)

    # Make sure decorator got applied correctly
    assert not hasattr(eval, "get_log")
    assert hasattr(eval, "get_cache")
    assert isinstance(eval.get_cache(), pd.DataFrame)
    assert len(eval.get_cache()) == 0

    # Evaluate some sites so they get added to the cache
    eval(sites)
    # Make sure the cache has the correct values
    assert (
        eval.get_cache() == eval_sites.loc[:, eval_sites.columns != "misc_col"]
    ).all(axis=None)

    # Evaluate sites again
    eval(sites)
    # Nothing should get added to the cache
    assert len(eval.get_cache()) == len(sites)

    # Delete database
    eval._evaluate.__del__()
    del eval
    cache_file_name.unlink()

    # Custom table name, new delta, auto update on
    eval = DummyEvaluator(
        opt_problem=opt_problem,
        cache=cache_file_name,
        cache_options={"table_name": "double_dip", "delta": 1, "auto_update": True},
    )

    # Make sure cache is empty
    assert len(eval.get_cache()) == 0
    assert cache_file_name.exists()

    eval(sites)

    # Add a site with obviously wrong responses to database
    connection = sql.connect(cache_file_name)
    pd.DataFrame(
        {
            "x1": [-1],
            "x2": [1],
            "f": [1_000],  # Real value: 0
            "c1": [-28],  # Real value: 0
            "c2": [90210],  # Real value: -1
            "cat_res": [np.inf],
        }
    ).to_sql("double_dip", connection, index=False, if_exists="append")
    connection.close()

    # Evaluate a site near the fake site
    df = pd.DataFrame({"x1": [-0.5], "x2": [0.25]})
    eval(df)

    # Make sure the cached site was used for the responses
    assert all(
        df.loc[0, ["x1", "x2", "f", "c1", "c2"]].to_numpy()
        == [-0.5, 0.25, 1_000, -28, 90210]
    )
    assert math.isnan(df.loc[0, "cat_res"])

    # Close force close connection to database
    eval._evaluate.__del__()

def test_opt_problem_direct(opt_problem: OptProblem
):
    cache_file_name = "test.db"
    if os.path.exists(cache_file_name):
        os.remove(cache_file_name)    
    # Create variables
    cust_variables = [
        FloatVariable(name="x1", bounds=[-5, 5], default=1.0, class_type="float"),
        FloatVariable(name="x2", bounds=[-5, 3], default=2.3, class_type="float"),
    ]

    # Create responses
    cust_responses = [
        FloatVariable(name="f", bounds=[np.inf, np.inf], class_type="float"),
        FloatVariable(name="c1", bounds=[-np.inf, 0.0], class_type="float"),
        FloatVariable(name="c2", bounds=[-np.inf, 0.0], class_type="float"),
        CategoricalVariable(
            name="cat_res", bounds=["f > 1", "-1 <= f <= 1", "f < -1"], class_type="cat"
        ),
    ]    

    custom_opt_problem = OptProblem(
        variables=cust_variables, responses=cust_responses, objectives=["f"], constraints=["c1","c2"]
    )

    eval = DummyEvaluator(opt_problem=custom_opt_problem, cache=cache_file_name)    
    assert eval._opt_problem.name != opt_problem.name
    
    eval._opt_problem.name = "opt"
    assert_opt_problems_equivalent(eval._opt_problem, opt_problem)

    # Delete database
    eval._evaluate.__del__()
    del eval
    os.remove(cache_file_name)    
   
def test_log_cache(
    opt_problem: OptProblem, sites: pd.DataFrame, eval_sites: pd.DataFrame, tmp_path: pathlib.Path
):
    cache_file_name = tmp_path / "test2.db"
    if os.path.exists(cache_file_name):
        os.remove(cache_file_name)

    eval_sites.drop(columns="misc_col", inplace=True)

    # Make sure that sites that are cached get logged
    eval = DummyEvaluator(
        opt_problem=opt_problem,
        logging=True,
        cache=cache_file_name,
        cache_options={"delta": 1},
    )

    # Make sure decorators were applied properly
    assert hasattr(eval, "get_log")
    assert hasattr(eval, "get_cache")
    assert len(eval.get_log()) == 0
    assert len(eval.get_cache()) == 0

    # Evaluate sites
    eval(sites)

    # Make sure log and cache are correct
    log = eval.get_log()
    cache = eval.get_cache()
    assert (log[eval_sites.columns] == eval_sites).all(axis=None)
    assert (log["call_num"] == 1).all()
    assert (cache == eval_sites).all(axis=None)

    # Create dataframe with one uncached and one cached site
    df = pd.DataFrame({"x1": [-1, 0.5], "x2": [1, 0.25]})
    # Create dataframe for expected log
    add_sites = pd.DataFrame(
        {
            "x1": [-1, 0.5],
            "x2": [1, 0.25],
            "f": [0, 2],  # Real values: [0, 0.75]
            "c1": [0, 0],  # Real values: [0, -1.6875]
            "c2": [-1, -1],  # Real values: [-1, -0.25]
            "cat_res": ["-1 <= f <= 1", "f > 1"],
        }
    )
    expected_log = pd.concat(
        (
            eval_sites,
            pd.DataFrame(
                {
                    "x1": [-1, 0.5],
                    "x2": [1, 0.25],
                    "f": [0, 2],  # Real values: [0, 0.75]
                    "c1": [0, 0],  # Real values: [0, -1.6875]
                    "c2": [-1, -1],  # Real values: [-1, -0.25]
                    "cat_res": ["-1 <= f <= 1", "f > 1"],
                }
            ),
        ),
        ignore_index=True,
    )
    # Create dataframe for expected cache
    expected_cache = pd.concat(
        (
            eval_sites,
            pd.DataFrame(
                {
                    "x1": [-1],
                    "x2": [1],
                    "f": [0],
                    "c1": [0],
                    "c2": [-1],
                    "cat_res": ["-1 <= f <= 1"],
                }
            ),
        ),
        ignore_index=True,
    )

    # Evaluate newly added sites
    eval(df)

    # Make sure log has both new sites and the cache has the uncached site added
    log = eval.get_log()
    cache = eval.get_cache()
    assert (log[eval_sites.columns] == expected_log).all(axis=None)
    assert (log["call_num"] == ([1] * len(sites) + [2] * 2)).all()
    assert (cache == expected_cache).all(axis=None)

def test_compute_signature_is_deterministic_and_sensitive_to_content():
    evaluator = ConstrainedBetts()
    opt_problem = evaluator.opt_problem

    sites1 = pd.DataFrame({"x1": [0.1, 0.2], "x2": [1.0, 2.0]}, index=[10, 20])
    sites2 = sites1.copy()
    sites3 = pd.DataFrame({"x1": [0.1, 0.2], "x2": [1.0, 2.1]}, index=[10, 20])
    sites4 = sites1.copy().reset_index(drop=True)

    sig1 = evaluator._compute_signature(sites1)
    sig2 = evaluator._compute_signature(sites2)
    sig3 = evaluator._compute_signature(sites3)
    sig4 = evaluator._compute_signature(sites4)

    # Same content and same index => same signature
    assert sig1 == sig2

    # Different content => different signature
    assert sig1 != sig3

    # Same values but index changed/reset => signature should differ
    assert sig1 != sig4


def test_get_partials_by_central_difference_matches_expected_shape():
    evaluator = ConstrainedBetts()
    opt_problem = evaluator.opt_problem

    sites = pd.DataFrame(
        {
            "x1": [0.1, 0.2],
            "x2": [1.0, 2.0],
        }
    )

    partials = evaluator._get_partials_by_central_difference(sites, opt_problem)

    # For ConstrainedBetts:
    # - 2 sites
    # - 1 selected response for partials if objective/constraint selection is one objective and one constraint
    # - 2 free variables
    assert partials.shape == (2, opt_problem.num_partials_responses, opt_problem.num_flat_vars)

    # Values should be finite
    assert np.isfinite(partials).all()


def test_compute_or_get_partials_uses_cache(monkeypatch):
    evaluator = ConstrainedBetts()
    opt_problem = evaluator.opt_problem

    sites = pd.DataFrame(
        {
            "x1": [0.1, 0.2],
            "x2": [1.0, 2.0],
        }
    )

    call_count = {"n": 0}

    def fake_get_partials(sites_arg, opt_problem_arg):
        call_count["n"] += 1
        return np.ones((len(sites_arg), opt_problem_arg.num_partials_responses, opt_problem_arg.num_flat_vars))

    monkeypatch.setattr(evaluator, "_get_partials_by_central_difference", fake_get_partials)

    # First call should compute
    p1 = evaluator._compute_or_get_partials(sites, opt_problem)
    assert call_count["n"] == 1
    assert np.array_equal(p1, np.ones((2, opt_problem.num_partials_responses, opt_problem.num_flat_vars)))

    # Second call with identical sites should use cache
    p2 = evaluator._compute_or_get_partials(sites.copy(), opt_problem)
    assert call_count["n"] == 1
    assert np.array_equal(p1, p2)

    # Modify content -> cache miss
    sites2 = sites.copy()
    sites2.loc[0, "x2"] = 1.5
    p3 = evaluator._compute_or_get_partials(sites2, opt_problem)
    assert call_count["n"] == 2
    assert np.array_equal(p3, np.ones((2, opt_problem.num_partials_responses, opt_problem.num_flat_vars)))


def test_evaluate_partials_matches_analytic_results():
    evaluator = ConstrainedBetts()
    opt_problem = evaluator.opt_problem

    sites = pd.DataFrame(
        {
            "x1": [0.1, 0.2],
            "x2": [1.0, 2.0],
        }
    )

    grad_fd, jac_fd = evaluator._evaluate_partials(sites, opt_problem)
    grad_an = evaluator.evaluate_analytic_gradient(sites.copy(), opt_problem)
    jac_an = evaluator.evaluate_analytic_jacobian(sites.copy(), opt_problem)

    assert grad_fd.shape == grad_an.shape
    assert jac_fd.shape == jac_an.shape

    # Finite differences should be close to analytic values
    assert np.allclose(grad_fd, grad_an, rtol=1e-6, atol=1e-6)
    assert np.allclose(jac_fd, jac_an, rtol=1e-6, atol=1e-6)


def test_evaluate_gradient_matches_analytic_gradient():
    evaluator = ConstrainedBetts()
    opt_problem = evaluator.opt_problem

    sites = pd.DataFrame(
        {
            "x1": [0.1, 0.2],
            "x2": [1.0, 2.0],
        }
    )

    grad_fd = evaluator.evaluate_gradient(sites, opt_problem)
    grad_an = evaluator.evaluate_analytic_gradient(sites.copy(), opt_problem)

    assert grad_fd.shape == grad_an.shape
    assert np.allclose(grad_fd, grad_an, rtol=1e-6, atol=1e-6)

    # Optional explicit expected values based on your example
    expected = np.array([[[2e-03, 2e+00]], [[4e-03, 4e+00]]])
    assert np.allclose(grad_an, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        grad_fd,
        np.array([[[2e-03, 2e+00]], [[4e-03, 4e+00]]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_evaluate_jacobian_matches_analytic_jacobian():
    evaluator = ConstrainedBetts()
    opt_problem = evaluator.opt_problem

    sites = pd.DataFrame(
        {
            "x1": [0.1, 0.2],
            "x2": [1.0, 2.0],
        }
    )

    jac_fd = evaluator.evaluate_jacobian(sites, opt_problem)
    jac_an = evaluator.evaluate_analytic_jacobian(sites.copy(), opt_problem)

    assert jac_fd.shape == jac_an.shape
    assert np.allclose(jac_fd, jac_an, rtol=1e-6, atol=1e-6)

    # Optional explicit expected values based on your example
    expected = np.array([[[10.0, -1.0]], [[10.0, -1.0]]])
    assert np.allclose(jac_an, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(jac_fd, np.array([[[10.00000012, -1.00000008]], [[10.00000012, -1.00000008]]]), rtol=1e-6, atol=1e-6)


# =============================================
# |   Options Management Tests               |
# =============================================

@pytest.fixture
def simple_opt_problem() -> OptProblem:
    """A minimal OptProblem with 1 variable and 1 response for options tests."""
    prob = se.utilities.create_opt_problem(num_independent=1, num_dependent=1)
    prob.variables[0].name = "x1"
    prob.responses[0].name = "f"
    prob.objectives = ["f"]
    prob = OptProblem(**prob.model_dump())
    return prob


class TestDefineOptions:
    def test_default_evaluator_has_empty_options(self, opt_problem):
        eval = DummyEvaluator(opt_problem=opt_problem)
        assert eval.required_options_names() == set()
        assert eval.full_options_names == set()

    def test_subclass_returns_custom_model(self):
        assert DummyEvaluatorWithOptions._define_options() is MyOptions

    def test_default_define_options_returns_empty_model(self):
        model_cls = DummyEvaluator._define_options()
        assert len(model_cls.model_fields) == 0


class TestRequiredOptions:
    def test_returns_instance_with_defaults(self):
        opts = DummyEvaluatorWithOptions.required_options()
        assert isinstance(opts, MyOptions)
        assert opts.alpha == 1.0
        assert opts.beta == 10.0
        assert opts.max_iter == 50
        assert opts.label == "default"

    def test_default_evaluator_returns_empty_instance(self):
        opts = DummyEvaluator.required_options()
        assert len(type(opts).model_fields) == 0


class TestRequiredOptionsNames:
    def test_returns_correct_names(self):
        names = DummyEvaluatorWithOptions.required_options_names()
        assert names == {"alpha", "beta", "max_iter", "label"}

    def test_empty_for_default_evaluator(self):
        assert DummyEvaluator.required_options_names() == set()


class TestFullOptions:
    def test_defaults_to_required_options(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        full = eval.full_options()
        req = eval.required_options()
        assert full.model_dump() == req.model_dump()

    def test_full_options_names_property(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        assert eval.full_options_names == {"alpha", "beta", "max_iter", "label"}


class TestLookupOptionValue:
    def test_lookup_default_value(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        assert eval.lookup_option_value("alpha") == 1.0
        assert eval.lookup_option_value("beta") == 10.0

    def test_lookup_overridden_value(self, simple_opt_problem):
        opts = MyOptions(alpha=5.5)
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem, options=opts)
        assert eval.lookup_option_value("alpha") == 5.5

    def test_lookup_unknown_raises_key_error(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        with pytest.raises(KeyError, match="key not found"):
            eval.lookup_option_value("nonexistent")


class TestCheckOptions:
    def test_valid_single_name(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        result = eval._check_options("alpha")
        assert result == ["alpha"]

    def test_valid_list_of_names(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        result = eval._check_options(["alpha", "beta"])
        assert result == ["alpha", "beta"]

    def test_unknown_name_raises_value_error(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        with pytest.raises(ValueError, match="not defined as an option"):
            eval._check_options("unknown_option")

    def test_non_string_raises_type_error(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        with pytest.raises(TypeError, match="not a string"):
            eval._check_options(123)

    def test_non_string_entry_in_list_raises_type_error(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        with pytest.raises(TypeError, match="not a string"):
            eval._check_options(["alpha", 42])

    def test_valid_values_pass(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        result = eval._check_options(["beta"], values={"beta": 50.0})
        assert result == ["beta"]

    def test_invalid_values_raise_value_error(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        with pytest.raises(ValueError, match="validation failed"):
            eval._check_options(["beta"], values={"beta": 200.0})  # exceeds le=100

    def test_invalid_type_in_values_raises_value_error(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        with pytest.raises(ValueError, match="validation failed"):
            eval._check_options(["max_iter"], values={"max_iter": -5})  # violates gt=0


class TestOptionsInit:
    def test_default_options_initialized(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        assert isinstance(eval._options, MyOptions)
        assert eval._options.alpha == 1.0

    def test_custom_options_accepted(self, simple_opt_problem):
        opts = MyOptions(alpha=2.5, beta=20.0, max_iter=100, label="custom")
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem, options=opts)
        assert eval._options.alpha == 2.5
        assert eval._options.beta == 20.0
        assert eval._options.max_iter == 100
        assert eval._options.label == "custom"

    def test_wrong_type_raises_type_error(self, simple_opt_problem):
        class OtherModel(BaseModel):
            gamma: float = 1.0

        with pytest.raises(TypeError, match="must be an instance of"):
            DummyEvaluatorWithOptions(opt_problem=simple_opt_problem, options=OtherModel())

    def test_mutated_invalid_options_raises_value_error(self, simple_opt_problem):
        opts = MyOptions(beta=50.0)
        # Mutate to an invalid value bypassing Pydantic validation
        object.__setattr__(opts, "beta", -999.0)
        with pytest.raises(ValueError, match="failed validation"):
            DummyEvaluatorWithOptions(opt_problem=simple_opt_problem, options=opts)

    def test_mutated_invalid_type_raises_value_error(self, simple_opt_problem):
        opts = MyOptions()
        # Mutate max_iter to invalid type bypassing Pydantic validation
        object.__setattr__(opts, "max_iter", "not_a_number")
        with pytest.raises(ValueError, match="failed validation"):
            DummyEvaluatorWithOptions(opt_problem=simple_opt_problem, options=opts)


class TestCurrentOptions:
    def test_current_options_returns_defaults(self, simple_opt_problem):
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem)
        current = eval.current_options()
        assert current.alpha == 1.0
        assert current.beta == 10.0

    def test_current_options_reflects_overrides(self, simple_opt_problem):
        opts = MyOptions(alpha=7.7, label="overridden")
        eval = DummyEvaluatorWithOptions(opt_problem=simple_opt_problem, options=opts)
        current = eval.current_options()
        assert current.alpha == 7.7
        assert current.label == "overridden"
        # Non-overridden values should still be defaults
        assert current.beta == 10.0
        assert current.max_iter == 50
