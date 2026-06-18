# Requirements Document

## Introduction

Simplify the `Evaluator` base class interface in the `standard-evaluator` Python library by removing deprecated legacy code paths. This includes removing the `num_independent`/`num_dependent` constructor parameters, the legacy `problem` dict initialization path (including `_def_problem`, `problem` property, `variables` property, `responses` property), and related helper classes. After this change, `Evaluator.__init__` requires either `opt_problem` or `interface` to be provided. The `TestEvaluator` subclass and its `_create_opt_problem()` pattern remain unchanged.

## Glossary

- **Evaluator**: The abstract base class (`abstract_evaluator.py`) from which all evaluator subclasses inherit.
- **OptProblem**: A Pydantic model representing an optimization problem with variables, responses, objectives, and constraints.
- **EvaluatorInfo**: A Pydantic model representing the interface of an evaluator with typed inputs and outputs.
- **TestEvaluator**: A specialized abstract subclass of `Evaluator` used for test functions that defines problems via `_create_opt_problem()`.
- **ShiftScaleEvaluator**: A subclass that applies shift/scale transformations between optimizer and design spaces.
- **OpenMDAOEvaluator**: A subclass that wraps an OpenMDAO problem as an evaluator.
- **PyEvaluator**: A subclass that wraps a plain Python callable as an evaluator.
- **MatlabEvaluator**: A subclass that wraps a MATLAB function as an evaluator.

## Requirements

### Requirement 1

**User Story:** As a library maintainer, I want to remove the deprecated `num_independent` and `num_dependent` parameters from `Evaluator.__init__`, so that the constructor interface is simplified and users are guided toward `opt_problem` or `interface`.

#### Acceptance Criteria

1. THE Evaluator class SHALL NOT accept `num_independent` or `num_dependent` as constructor parameters.
2. WHEN a caller passes `num_independent` or `num_dependent` as keyword arguments, THE Evaluator SHALL raise a `TypeError` (standard Python behavior for unexpected keyword arguments).
3. THE `ValidInputs` class SHALL be removed from `abstract_evaluator.py`.
4. THE `InputModel` class SHALL be removed from `abstract_evaluator.py`.
5. THE `validate_numbers()` function SHALL be removed from `abstract_evaluator.py`.

### Requirement 2

**User Story:** As a library maintainer, I want to remove the legacy `problem` dict initialization path from `Evaluator.__init__`, so that only `opt_problem` and `interface` remain as supported initialization mechanisms.

#### Acceptance Criteria

1. THE Evaluator `__init__` method SHALL support initialization via the `opt_problem` parameter.
2. THE Evaluator `__init__` method SHALL support initialization via the `interface` parameter.
3. WHEN neither `opt_problem` nor `interface` is provided to `Evaluator.__init__`, THE Evaluator SHALL raise a `ValueError` with a message indicating that one of them must be provided.
4. THE `_def_problem()` method SHALL be removed from the `Evaluator` class.
5. THE `else` branch in `Evaluator.__init__` that calls `self._def_problem(**kwargs)` SHALL be removed.
6. THE `elif validate_numbers(...)` branch in `Evaluator.__init__` SHALL be removed.

### Requirement 3

**User Story:** As a library maintainer, I want to remove deprecated properties from the `Evaluator` class, so that users transition to the supported `inputs`, `outputs`, `opt_problem`, and `interface` properties.

#### Acceptance Criteria

1. THE `problem` property SHALL be removed from the `Evaluator` class.
2. THE `variables` property (deprecated alias for `self._inputs`) SHALL be removed from the `Evaluator` class.
3. THE `responses` property (deprecated alias for `self._outputs`) SHALL be removed from the `Evaluator` class.
4. THE assignment `self._problem = opt_problem_to_legacy(self._opt_problem)` SHALL be removed from `Evaluator.__init__`.
5. THE import of `opt_problem_to_legacy` SHALL be removed from `abstract_evaluator.py` when it is no longer referenced.
6. THE import of `create_opt_problem` SHALL be removed from `abstract_evaluator.py` when it is no longer referenced.

### Requirement 4

**User Story:** As a library maintainer, I want the `OpenMDAOEvaluator` to pass an `opt_problem` to `super().__init__()` instead of a legacy `problem` dict, so that it is compatible with the simplified interface.

#### Acceptance Criteria

1. WHEN `OpenMDAOEvaluator.__init__` calls `super().__init__()`, THE OpenMDAOEvaluator SHALL pass an `OptProblem` instance via the `opt_problem` keyword argument.
2. THE OpenMDAOEvaluator SHALL NOT pass a `problem` dictionary to `super().__init__()`.

### Requirement 5

**User Story:** As a library maintainer, I want the `ShiftScaleEvaluator` to remove the legacy `problem` dict parameter, so that it only accepts `opt_problem` or `interface`.

#### Acceptance Criteria

1. THE `ShiftScaleEvaluator.__init__` signature SHALL NOT include a `problem: dict` parameter.
2. THE branch that handles the `problem` parameter (calling `legacy_to_opt_problem`) SHALL be removed from `ShiftScaleEvaluator.__init__`.
3. THE FutureWarning for the legacy `problem` parameter SHALL be removed from `ShiftScaleEvaluator.__init__`.
4. WHEN neither `opt_problem` nor `interface` is provided to `ShiftScaleEvaluator.__init__`, THE ShiftScaleEvaluator SHALL raise a `ValueError`.
5. THE import of `legacy_to_opt_problem` SHALL be removed from `shift_scale_evaluator.py`.

### Requirement 6

**User Story:** As a library maintainer, I want the `PyEvaluator` and `MatlabEvaluator` docstrings cleaned up to remove references to removed parameters, so that documentation remains accurate.

#### Acceptance Criteria

1. THE `PyEvaluator.__init__` docstring SHALL NOT reference `problem definition` or `_def_problem` in the `**kwargs` description.
2. THE `MatlabEvaluator.__init__` docstring SHALL NOT reference `problem definition` or `_def_problem` in the `**kwargs` description.

### Requirement 7

**User Story:** As a library maintainer, I want all tests updated to use the current interface (`opt_problem`, `interface`, `inputs`, `outputs`), so that the test suite passes after removing deprecated paths.

#### Acceptance Criteria

1. THE `DummyEvaluator` in `test_numpy_evaluator.py` SHALL be refactored to not override `_def_problem`, and SHALL be instantiated with an `opt_problem` or `interface` argument.
2. THE test `test_numpy_evaluator` SHALL use `test_instance.inputs` instead of `test_instance.variables` and `test_instance.outputs` instead of `test_instance.responses`.
3. WHEN tests in `test_abstract_evaluator.py` exercise `num_independent`/`num_dependent` initialization, THOSE tests SHALL be removed or converted to test the `opt_problem`/`interface` paths.
4. THE test `test_openmdao_evaluator.py` SHALL use `my_evaluator.outputs` instead of `my_evaluator.responses` and `my_evaluator.inputs` instead of `my_evaluator.variables`.
5. WHEN tests reference the removed `evaluator.problem` property, THOSE tests SHALL be updated to use `evaluator.opt_problem` or removed.

### Requirement 8

**User Story:** As a library maintainer, I want the `create_opt_problem()` utility function to remain available for external use, so that users who previously relied on `num_independent`/`num_dependent` can call it themselves before passing the result to `Evaluator.__init__`.

#### Acceptance Criteria

1. THE `create_opt_problem()` function SHALL remain in `standard_evaluator.utilities.problem_dict_utility`.
2. THE `create_opt_problem()` function SHALL remain accessible via `standard_evaluator.utilities.create_opt_problem`.
