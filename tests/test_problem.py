import copy
import pytest
import pandas as pd
import numpy as np
from standard_evaluator import FloatVariable, IntVariable, ArrayVariable
from standard_evaluator import EvaluatorInfo, OptProblem, MAXINT
from standard_evaluator import (
    evaluator_info_to_opt_problem,
    opt_problem_to_evaluator_info,
    CategoricalVariable,
)

def test_float_variable():
    my_float_var = FloatVariable(name="dummy")
    # Check that when no bounds are defined we set bounds to [-inf, inf]
    assert my_float_var.bounds[0] == -np.inf
    assert my_float_var.bounds[1] == np.inf
    with pytest.raises(ValueError):
        # Check that is lower bound is larger than upper bound we throw an error
        FloatVariable(name="dummy", bounds=[2.0, 1.0])
    # Check that if bounds are set they are saved correctly
    my_float_var2 = FloatVariable(name="dummy", bounds=[2.0, 4.0])
    assert my_float_var2.bounds[0] == 2.0
    assert my_float_var2.bounds[1] == 4.0

    with pytest.raises(ValueError):
        # Check that if scale is 0. we throw an error
        FloatVariable(name="dummy", scale=0.0)

    with pytest.raises(ValueError):
        # Check that if name contains spaces we throw an error
        FloatVariable(name="dummy 3")

    # Check that we can use an empty name
    my_float_var3 = FloatVariable(name="")
    assert my_float_var3.name == ""

def test_categorical_variable():
    my_cat_var = CategoricalVariable(name="x1", bounds=["x1 = 1", 4.3, 6.5])
    assert my_cat_var.bounds == ["x1 = 1", 4.3, 6.5]
    with pytest.raises(ValueError):
        # Check that is lower bound is larger than upper bound we throw an error
        IntVariable(name="dummy", bounds=[2.0, 1.0, "ret"], default="no")

def test_int_variable():
    my_int_var = IntVariable(name="dummy")
    # Check that when no bounds are defined we set bounds to [-inf, inf]
    assert my_int_var.bounds[0] == -MAXINT
    assert my_int_var.bounds[1] == MAXINT
    with pytest.raises(ValueError):
        # Check that is lower bound is larger than upper bound we throw an error
        IntVariable(name="dummy", bounds=[2.0, 1.0])
    # Check that if bounds are set they are saved correctly
    my_int_var2 = FloatVariable(name="dummy", bounds=[2, 4])
    assert my_int_var2.bounds[0] == 2
    assert my_int_var2.bounds[1] == 4

    with pytest.raises(ValueError):
        # Check that if scale is 0. we throw an error
        IntVariable(name="dummy", scale=0)


def test_array_variable():
    my_array_var = ArrayVariable(name="dummy", shape=(2, 3))
    # Check that when no bounds are defined we set bounds to [-inf, inf]
    assert np.all(np.isinf(my_array_var.bounds[0]))
    assert np.all(np.isinf(my_array_var.bounds[1]))
    # make sure the shift and scale are set correctly
    np.testing.assert_array_equal(my_array_var.shift, 0.0 * np.ones(shape=(2, 3)))
    np.testing.assert_array_equal(my_array_var.scale, 1.0 * np.ones(shape=(2, 3)))
    with pytest.raises(ValueError):
        # Check that is lower bound is larger than upper bound we throw an error
        ArrayVariable(name="dummy", bounds=[2.0, 1.0], shape=(3, 2))
    with pytest.raises(ValueError):
        # Check that either shape or default need to be defined
        ArrayVariable(name="dummy")
    # Check that if bounds are set they are saved correctly and expanded to the shape defined
    my_array_var2 = ArrayVariable(
        name="dummy", bounds=[2.0, 4.0], shape=(3, 2), shift=8.0, scale=0.1
    )
    np.testing.assert_array_equal(my_array_var2.bounds[0], 2.0 * np.ones(shape=(3, 2)))
    np.testing.assert_array_equal(my_array_var2.bounds[1], 4.0 * np.ones(shape=(3, 2)))
    # make sure the shift and scale are set correctly
    np.testing.assert_array_equal(my_array_var2.shift, 8.0 * np.ones(shape=(3, 2)))
    np.testing.assert_array_equal(my_array_var2.scale, 0.1 * np.ones(shape=(3, 2)))

    with pytest.raises(ValueError):
        # Check that if scale is 0. we throw an error
        ArrayVariable(name="dummy", scale=0.0, shape=(1, 2))

    # Check that if the default is given we take the shape from it
    default = np.array([[3.2, 6.3, 3.2], [1.2, 1.2, 34.4]])
    my_array_var3 = ArrayVariable(name="x4", default=default, scale=2.3)
    assert my_array_var3.shape == (2, 3)

    with pytest.raises(ValueError):
        # We cannot set the scale to a different shape than default
        ArrayVariable(name="x4", default=default, scale=np.array([2.3]))

    with pytest.raises(ValueError):
        # We cannot set the shift to a different shape than default
        ArrayVariable(name="x4", default=default, shift=np.array([2.3]))

    with pytest.raises(ValueError):
        # We cannot set the bounds to a different shape than default
        ArrayVariable(
            name="x4", default=default, bounds=[np.array([2.3]), np.array([2.3])]
        )

    with pytest.raises(ValueError):
        # We cannot set the shape to a different shape than default
        ArrayVariable(name="x4", default=default, shape=(2,))

    my_array_var4 = ArrayVariable(
        name="dummy",
        default=default,
        bounds=[-default, default],
        shift=8.0 * default,
        scale=0.1 * default,
    )
    np.testing.assert_array_equal(my_array_var4.bounds[0], -default)
    np.testing.assert_array_equal(my_array_var4.bounds[1], default)
    # make sure the shift and scale are set correctly
    np.testing.assert_array_equal(my_array_var4.shift, 8.0 * default)
    np.testing.assert_array_equal(my_array_var4.scale, 0.1 * default)

    with pytest.raises(ValueError):
        # Check that if any element of scale is 0. we throw an error
        ArrayVariable(name="dummy", scale=np.array([3.2, 0.0]), shape=(1, 2))


def test_opt_problem():
    with pytest.raises(ValueError):
        # Check that if if you try to name two inputs the same it throws an error
        OptProblem(name="bla", variables=[{"name": "x0"}, {"name": "x0"}])

    with pytest.raises(ValueError):
        # Check that if if you try to name two outputs the same it throws an error
        OptProblem(
            name="bla",
            variables=[{"name": "x0"}],
            responses=[{"name": "x0"}, {"name": "x0"}],
        )

    with pytest.raises(ValueError):
        # We need to define at least one input
        OptProblem(name="bla", variables=[])

    variables = [
        ArrayVariable(name="x1", default=[[299.3, 243.5]]),
        FloatVariable(name="p2", class_type="float"),
    ]
    responses = [FloatVariable(name="y2", class_type="float")]
    my_prob = OptProblem(
        variables=variables, responses=responses, objectives=["y2"], constraints=["y2"]
    )
    expected = {
        "name": "opt_problem",
        "class_type": "OptProblem",
        "variables": [
            {
                "name": "x1",
                "default": np.array([[299.3, 243.5]]),
                "bounds": None,
                "shift": None,
                "scale": None,
                "units": None,
                "description": "",
                "class_type": "floatarray",
                "shape": (1, 2),
                "options": {},
            },
            {
                "name": "p2",
                "default": None,
                "bounds": (-np.inf, np.inf),
                "shift": 0.0,
                "scale": 1.0,
                "units": None,
                "description": "",
                "class_type": "float",
                "options": {},
            },
        ],
        "responses": [
            {
                "name": "y2",
                "default": None,
                "bounds": (-np.inf, np.inf),
                "shift": 0.0,
                "scale": 1.0,
                "units": None,
                "description": "",
                "class_type": "float",
                "options": {},
            }
        ],
        "objectives": ["y2"],
        "constraints": ["y2"],
        "description": None,
        "cite": None,
        "options": {},
    }
    np.testing.assert_equal(expected, my_prob.model_dump())

    my_prob = OptProblem(
        name="dummy",
        variables=variables,
        responses=responses,
        objectives=["y2"],
        constraints=["y2"],
    )
    expected = {
        "name": "dummy",
        "class_type": "OptProblem",
        "variables": [
            {
                "name": "x1",
                "default": np.array([[299.3, 243.5]]),
                "bounds": None,
                "shift": None,
                "scale": None,
                "units": None,
                "description": "",
                "class_type": "floatarray",
                "shape": (1, 2),
                "options": {},
            },
            {
                "name": "p2",
                "default": None,
                "bounds": (-np.inf, np.inf),
                "shift": 0.0,
                "scale": 1.0,
                "units": None,
                "description": "",
                "class_type": "float",
                "options": {},
            },
        ],
        "responses": [
            {
                "name": "y2",
                "default": None,
                "bounds": (-np.inf, np.inf),
                "shift": 0.0,
                "scale": 1.0,
                "units": None,
                "description": "",
                "class_type": "float",
                "options": {},
            }
        ],
        "objectives": ["y2"],
        "constraints": ["y2"],
        "description": None,
        "cite": None,
        "options": {},
    }
    np.testing.assert_equal(expected, my_prob.model_dump())

    with pytest.raises(ValueError):
        # A variable cannot be a constraint
        OptProblem(
            variables=variables,
            responses=responses,
            objectives=["y2"],
            constraints=["p2"],
        )

    with pytest.raises(ValueError):
        # An element in an array variable cannot be outside the shape
        OptProblem(
            variables=variables,
            responses=responses,
            objectives=["x1[0,3]"],
            constraints=["y2"],
        )

    with pytest.raises(ValueError):
        # You cannot have a constraint that is not a response
        OptProblem(
            variables=variables,
            responses=responses,
            objectives=["y2"],
            constraints=["Hello39"],
        )

    my_prob3 = OptProblem(
        name="dummy",
        variables=variables,
        responses=responses,
        objectives=["x1[0,1]"],
        constraints=["y2"],
    )
    expected = {
        "name": "dummy",
        "class_type": "OptProblem",
        "variables": [
            {
                "name": "x1",
                "default": np.array([[299.3, 243.5]]),
                "bounds": None,
                "shift": None,
                "scale": None,
                "units": None,
                "description": "",
                "class_type": "floatarray",
                "shape": (1, 2),
                "options": {},
            },
            {
                "name": "p2",
                "default": None,
                "bounds": (-np.inf, np.inf),
                "shift": 0.0,
                "scale": 1.0,
                "units": None,
                "description": "",
                "class_type": "float",
                "options": {},
            },
        ],
        "responses": [
            {
                "name": "y2",
                "default": None,
                "bounds": (-np.inf, np.inf),
                "shift": 0.0,
                "scale": 1.0,
                "units": None,
                "description": "",
                "class_type": "float",
                "options": {},
            }
        ],
        "objectives": ["x1[0,1]"],
        "constraints": ["y2"],
        "description": None,
        "cite": None,
        "options": {},
    }
    np.testing.assert_equal(expected, my_prob3.model_dump())
