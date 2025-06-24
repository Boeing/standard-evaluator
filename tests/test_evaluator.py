import copy
import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError
from standard_evaluator import FloatVariable, IntVariable, ArrayVariable, CategoricalVariable
from standard_evaluator import EvaluatorInfo, GroupInfo, JoinedInfo,OptProblem, MAXINT

def test_evaluator_info():
    with pytest.raises(ValueError):
        # Check that if if you try to name two inputs the same it throws an error
        EvaluatorInfo(name="bla", inputs=[{"name": "x0"}, {"name": "x0"}])

    with pytest.raises(ValueError):
        # Check that if if you try to name two outputs the same it throws an error
        EvaluatorInfo(name="bla", outputs=[{"name": "x0"}, {"name": "x0"}])
    my_eval = EvaluatorInfo(
        name="dummy",
        inputs=[
            ArrayVariable(name="x1", default=[[299.3, 243.5]]),
            FloatVariable(name="p2", class_type="float"),
        ],
        outputs=[FloatVariable(name="y2", class_type="float")],
    )
    expected = {
        "name": "dummy",
        "class_type": "EvaluatorInfo",
        "inputs": [
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
        "outputs": [
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
        "description": None,
        "cite": None,
        "tool": None,
        "evaluator_identifier": None,
        "version": None,
        "component_type": None,
        "options": {},
    }
    np.testing.assert_equal(expected, my_eval.model_dump())

def test_group():
    x1 = ArrayVariable(name="x1", default=[[299.3, 243.5]])
    p2 = FloatVariable(name="p2")
    q1 = FloatVariable(name="q1")
    y2 = FloatVariable(name="y2")
    y4 = FloatVariable(name="y4")
    my_eval1 = EvaluatorInfo(
            name="Part1",
            inputs=[x1, p2, q1],
            outputs=[y2],
        )
    my_eval2 = EvaluatorInfo(
            name="Part2",
            inputs=[x1, p2, y2],
            outputs=[y4],
        )
    my_group = GroupInfo(name="MyGroup",
            inputs=[x1, p2, q1],
            outputs=[y4],
            components={my_eval1.name: my_eval1, my_eval2.name: my_eval2},
            component_order=[my_eval1.name, my_eval2.name],
            linkage=[("Part1.x1", "x1"), ("Part1.p2", "p2"), ("Part1.q1", "q1"),
                    ("Part1.y2", "Part2.y2"), ("Part2.x1", "x1"), ("Part2.p2", "p2"),
                    ("Part2.y4", "y4")])
    expected = {'name': 'MyGroup',
 'class_type': 'GroupInfo',
 'inputs': [{'name': 'x1',
   'default': np.array([[299.3, 243.5]]),
   'bounds': None,
   'shift': None,
   'scale': None,
   'units': None,
   'description': '',
   'options': {},
   'class_type': 'floatarray',
   'shape': (1, 2)},
  {'name': 'p2',
   'default': None,
   'bounds': (-np.inf, np.inf),
   'shift': 0.0,
   'scale': 1.0,
   'units': None,
   'description': '',
   'options': {},
   'class_type': 'float'},
  {'name': 'q1',
   'default': None,
   'bounds': (-np.inf, np.inf),
   'shift': 0.0,
   'scale': 1.0,
   'units': None,
   'description': '',
   'options': {},
   'class_type': 'float'}],
 'outputs': [{'name': 'y4',
   'default': None,
   'bounds': (-np.inf, np.inf),
   'shift': 0.0,
   'scale': 1.0,
   'units': None,
   'description': '',
   'options': {},
   'class_type': 'float'}],
 'description': None,
 'cite': None,
 'tool': None,
 'evaluator_identifier': None,
 'version': None,
 'component_type': None,
 'options': {},
 'component_order': ['Part1', 'Part2'],
 'components': {'Part1': {'name': 'Part1',
   'class_type': 'EvaluatorInfo',
   'inputs': [{'name': 'x1',
     'default': np.array([[299.3, 243.5]]),
     'bounds': None,
     'shift': None,
     'scale': None,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'floatarray',
     'shape': (1, 2)},
    {'name': 'p2',
     'default': None,
     'bounds': (-np.inf, np.inf),
     'shift': 0.0,
     'scale': 1.0,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'float'},
    {'name': 'q1',
     'default': None,
     'bounds': (-np.inf, np.inf),
     'shift': 0.0,
     'scale': 1.0,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'float'}],
   'outputs': [{'name': 'y2',
     'default': None,
     'bounds': (-np.inf, np.inf),
     'shift': 0.0,
     'scale': 1.0,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'float'}],
   'description': None,
   'cite': None,
   'tool': None,
   'evaluator_identifier': None,
   'version': None,
   'component_type': None,
   'options': {}},
  'Part2': {'name': 'Part2',
   'class_type': 'EvaluatorInfo',
   'inputs': [{'name': 'x1',
     'default': np.array([[299.3, 243.5]]),
     'bounds': None,
     'shift': None,
     'scale': None,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'floatarray',
     'shape': (1, 2)},
    {'name': 'p2',
     'default': None,
     'bounds': (-np.inf, np.inf),
     'shift': 0.0,
     'scale': 1.0,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'float'},
    {'name': 'y2',
     'default': None,
     'bounds': (-np.inf, np.inf),
     'shift': 0.0,
     'scale': 1.0,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'float'}],
   'outputs': [{'name': 'y4',
     'default': None,
     'bounds': (-np.inf, np.inf),
     'shift': 0.0,
     'scale': 1.0,
     'units': None,
     'description': '',
     'options': {},
     'class_type': 'float'}],
   'description': None,
   'cite': None,
   'tool': None,
   'evaluator_identifier': None,
   'version': None,
   'component_type': None,
   'options': {}}},
 'linkage': [('Part1.x1', 'x1'),
  ('Part1.p2', 'p2'),
  ('Part1.q1', 'q1'),
  ('Part1.y2', 'Part2.y2'),
  ('Part2.x1', 'x1'),
  ('Part2.p2', 'p2'),
  ('Part2.y4', 'y4')],
  'promotions': {},
  }
    np.testing.assert_equal(expected, my_group.model_dump())
  
def test_evaluator_info_default():
    lower_bound = np.array([[1, 2], [-np.inf, 4], [5, 4]])
    upper_bound = np.array([[3, 4], [6, np.inf], [np.inf, 8]])
    expected_default = np.array([[2., 3.], [6., 4.], [5., 6.]])
    x1 = FloatVariable(name='x1', bounds=[2.,5])
    x2 = IntVariable(name='x2', bounds=[2,5])
    cat = CategoricalVariable(name="cat1", bounds=["x2", "f3", "f1>1"])
    array1 = ArrayVariable(name="array1", bounds=(lower_bound, upper_bound), shape=lower_bound.shape)
    y1 = FloatVariable(name='y1', bounds=[2.,5])
    y2 = IntVariable(name='y2', bounds=[2,5])
    my_prob = EvaluatorInfo(name="opt", inputs=[x1, x2, cat, array1], outputs=[y1, y2])
    my_prob.calculate_default()
    assert my_prob.inputs[0].default == 3.5
    assert my_prob.inputs[1].default == 3
    assert my_prob.inputs[2].default == "x2"
    np.testing.assert_equal(my_prob.inputs[3].default, expected_default)

    with pytest.raises(ValidationError):
        my_prob.set_defaults({'x1': 4.0, 'x2': 'sd', 'cat1': 'f3'})