# Design Document: Simplify Evaluator Interface

## Overview

This design removes deprecated initialization paths from the `Evaluator` base class, leaving only `opt_problem` and `interface` as supported mechanisms for defining an evaluator's inputs and outputs. Legacy classes, functions, properties, and branches are deleted from the codebase, and all subclasses and tests are updated accordingly.

## Architecture

The change is entirely within the `standard_evaluator.evaluators` package and its test suite. No new modules or architectural layers are introduced. The inheritance hierarchy remains:

```
Evaluator (ABC)
├── TestEvaluator (ABC) — unchanged
├── NumpyEvaluator (ABC)
├── PyEvaluator
├── MatlabEvaluator
├── OpenMDAOEvaluator
├── ShiftScaleEvaluator
└── ExecutableEvaluator
```

## Components

### 1. `abstract_evaluator.py` — Removals

**Classes and functions to delete:**
- `ValidInputs` (Pydantic model)
- `InputModel` (Pydantic model)
- `validate_numbers()` function

**From `Evaluator.__init__` signature, remove:**
- `num_independent: int = None`
- `num_dependent: int = None`

**From `Evaluator.__init__` body, replace the branching logic:**

```python
# BEFORE (simplified):
if opt_problem is not None:
    self._opt_problem = opt_problem
elif interface is not None:
    self._opt_problem = evaluator_info_to_opt_problem(interface)
elif validate_numbers(num_dependent, num_independent):
    self._opt_problem = create_opt_problem(num_independent, num_dependent)
else:
    problem = self._def_problem(**kwargs)
    utils.check_prob(problem)
    self._opt_problem = utils.legacy_to_opt_problem(problem)

# AFTER:
if opt_problem is not None:
    if not isinstance(opt_problem, se.OptProblem):
        raise TypeError("opt_problem argument must be of type OptProblem if set.")
    self._opt_problem = opt_problem
elif interface is not None:
    self._opt_problem = evaluator_info_to_opt_problem(interface)
else:
    raise ValueError(
        f"{type(self).__name__}: Either 'opt_problem' or 'interface' must be provided."
    )
```

**Properties to delete:**
- `problem` — returned `self._problem` with FutureWarning
- `variables` — deprecated alias for `self._inputs`
- `responses` — deprecated alias for `self._outputs`

**Methods to delete:**
- `_def_problem()` — legacy problem-dict builder

**Attribute assignments to remove:**
- `self._problem = opt_problem_to_legacy(self._opt_problem)` — no longer needed

**Imports to remove (from abstract_evaluator.py):**
- `opt_problem_to_legacy` from `standard_evaluator.utilities.problem_dict_utility`
- `create_opt_problem` from `standard_evaluator.utilities.problem_dict_utility`

**Note:** The `initial_guess()` method currently references `self._problem`. It must be updated to derive the DataFrame directly from `self._opt_problem` or `self._interface` instead.

### 2. `open_mdao_evaluator.py` — Refactor

Currently calls `super().__init__(name=name, comp_cost=comp_cost, problem=problem)` where `problem` is a legacy dict. Change to:

```python
super().__init__(name=name, comp_cost=comp_cost, opt_problem=problem)
```

The variable `problem` in this file is already an `OptProblem` returned from `get_om_opt_problem()` or `self._main_problem` (also an OptProblem). Remove the `import pprint` and `pprint.pprint(problem)` debug statements.

### 3. `shift_scale_evaluator.py` — Remove `problem` parameter

```python
# BEFORE:
def __init__(self, evaluate: Evaluator, interface: EvaluatorInfo = None,
             opt_problem: OptProblem = None, problem: dict = None) -> None:

# AFTER:
def __init__(self, evaluate: Evaluator, interface: EvaluatorInfo = None,
             opt_problem: OptProblem = None) -> None:
```

Remove the `elif problem is not None:` branch and associated FutureWarning. Remove the `legacy_to_opt_problem` import.

Update the `else` clause to raise `ValueError` if neither `opt_problem` nor `interface` is provided (this behavior already exists in the current code when all three are None, just clean up the message).

### 4. `evaluator.py` (PyEvaluator) — Docstring cleanup

Remove the `**kwargs: Parameters sent to problem definition` line from the docstring. The `**kwargs` in the signature still passes through to `super().__init__()` for options, cache, logging, etc.

### 5. `matlab_evaluator.py` — Docstring cleanup

Remove reference to `problem definition` in the `**kwargs` docstring if present.

### 6. Test Updates

#### `tests/evaluators/test_numpy_evaluator.py`

```python
# BEFORE:
class DummyEvaluator(NumpyEvaluator):
    def _def_problem(self, num_independent, num_dependent):
        return self._auto_problem(num_independent, num_dependent)

test_instance = DummyEvaluator(num_independent=6, num_dependent=1)

# AFTER:
from standard_evaluator.utilities import create_opt_problem

class DummyEvaluator(NumpyEvaluator):
    pass  # eval_np is the only required override

test_instance = DummyEvaluator(opt_problem=create_opt_problem(6, 1))
```

Replace `test_instance.variables` → `test_instance.inputs`, `test_instance.responses` → `test_instance.outputs`.

#### `tests/evaluators/test_abstract_evaluator.py`

- Remove tests that instantiate with `num_independent`/`num_dependent`.
- Remove tests that assert on `evaluator.problem`.
- Tests using the `opt_problem` fixture remain unchanged.

#### `tests/evaluators/test_openmdao_evaluator.py`

- Replace `my_evaluator.responses` → `my_evaluator.outputs`.
- Replace `my_evaluator.variables` → `my_evaluator.inputs`.
- Replace `my_evaluator.problem` → assertions on `my_evaluator.opt_problem` (compare OptProblem attributes instead of a legacy dict).

## Data Models

No new data models are introduced. `OptProblem` and `EvaluatorInfo` remain the sole interface contracts.

## Error Handling

| Condition | Error | Message |
|-----------|-------|---------|
| Neither `opt_problem` nor `interface` provided | `ValueError` | `"{ClassName}: Either 'opt_problem' or 'interface' must be provided."` |
| `opt_problem` is not an `OptProblem` instance | `TypeError` | `"opt_problem argument must be of type OptProblem if set."` |
| `ShiftScaleEvaluator` receives neither `opt_problem` nor `interface` | `ValueError` | `"One of 'opt_problem' or 'interface' must be provided."` |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: OptProblem initialization preserves interface

*For any* valid `OptProblem` instance with N variables and M responses, constructing an `Evaluator` subclass with that `opt_problem` SHALL result in `evaluator.inputs` having length N and `evaluator.outputs` having length M, with names matching the OptProblem's variable and response names.

**Validates: Requirements 2.1**

### Property 2: EvaluatorInfo initialization preserves interface

*For any* valid `EvaluatorInfo` instance with N inputs and M outputs, constructing an `Evaluator` subclass with that `interface` SHALL result in `evaluator.inputs` having length N and `evaluator.outputs` having length M, with names matching the EvaluatorInfo's input and output names.

**Validates: Requirements 2.2**

### Property 3: create_opt_problem utility produces valid OptProblem

*For any* pair of positive integers (num_independent, num_dependent), calling `create_opt_problem(num_independent, num_dependent)` SHALL return a valid `OptProblem` with exactly `num_independent` variables and `num_dependent` responses.

**Validates: Requirements 8.1**

## Notes

- `TestEvaluator` and its `_create_opt_problem()` pattern are explicitly excluded from changes.
- `create_opt_problem()` remains available in `standard_evaluator.utilities` for users who want to construct an OptProblem from counts.
- The `_store_default_in_problem()` method and `initial_guess()` method need minor updates to work without `self._problem`. They should use `self._opt_problem` directly (which they already partially do).
