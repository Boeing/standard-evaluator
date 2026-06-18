# Implementation Plan: Simplify Evaluator Interface

## Overview

Remove deprecated legacy code paths from `Evaluator.__init__` and its subclasses. The implementation proceeds from the core outward: first the base class, then subclasses, then tests.

## Tasks

- [x] 1. Remove legacy classes, functions, and imports from `abstract_evaluator.py`
  - [x] 1.1 Remove `ValidInputs` class, `InputModel` class, and `validate_numbers()` function
    - Delete all three definitions from `abstract_evaluator.py`
    - Remove the `from standard_evaluator.utilities.problem_dict_utility import opt_problem_to_legacy, create_opt_problem` import (both symbols)
    - _Requirements: 1.3, 1.4, 1.5, 3.5, 3.6_
  - [x] 1.2 Simplify `Evaluator.__init__` signature and body
    - Remove `num_independent: int = None` and `num_dependent: int = None` from the parameter list
    - Replace the `elif validate_numbers(...)` branch and `else` branch with `else: raise ValueError(...)`
    - Remove the `self._problem = opt_problem_to_legacy(self._opt_problem)` assignment
    - Update `initial_guess()` to build the DataFrame from `self._opt_problem` instead of `self._problem`
    - _Requirements: 1.1, 1.2, 2.3, 2.5, 2.6, 3.4_
  - [x] 1.3 Remove deprecated properties and `_def_problem` method
    - Delete the `problem` property
    - Delete the `variables` property
    - Delete the `responses` property
    - Delete the `_def_problem()` method
    - _Requirements: 3.1, 3.2, 3.3, 2.4_

- [x] 2. Update subclasses to use the simplified interface
  - [x] 2.1 Refactor `OpenMDAOEvaluator.__init__` to pass `opt_problem`
    - Change `super().__init__(name=name, comp_cost=comp_cost, problem=problem)` to `super().__init__(name=name, comp_cost=comp_cost, opt_problem=problem)`
    - Remove the `import pprint` and `pprint.pprint(problem)` debug lines
    - _Requirements: 4.1, 4.2_
  - [x] 2.2 Remove `problem` parameter from `ShiftScaleEvaluator.__init__`
    - Remove `problem: dict = None` from the signature
    - Remove the `elif problem is not None:` branch and its FutureWarning
    - Remove the `from standard_evaluator.utilities.problem_dict_utility import legacy_to_opt_problem` import
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [x] 2.3 Clean up `PyEvaluator` and `MatlabEvaluator` docstrings
    - Remove `**kwargs: Parameters sent to problem definition` from the `PyEvaluator.__init__` docstring
    - Remove any reference to `problem definition` in `MatlabEvaluator.__init__` docstring
    - _Requirements: 6.1, 6.2_

- [x] 3. Checkpoint — Verify the source code is internally consistent
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Update test files
  - [x] 4.1 Refactor `tests/evaluators/test_numpy_evaluator.py`
    - Remove `_def_problem` override from `DummyEvaluator`
    - Change instantiation to use `opt_problem=create_opt_problem(6, 1)`
    - Replace `test_instance.variables` with `test_instance.inputs`
    - Replace `test_instance.responses` with `test_instance.outputs`
    - _Requirements: 7.1, 7.2_
  - [x] 4.2 Update `tests/evaluators/test_abstract_evaluator.py`
    - Remove or convert tests that exercise `num_independent`/`num_dependent` initialization
    - Remove or convert tests that assert on `evaluator.problem`
    - Keep tests that use the `opt_problem` fixture as-is
    - _Requirements: 7.3, 7.5_
  - [x] 4.3 Update `tests/evaluators/test_openmdao_evaluator.py`
    - Replace `my_evaluator.responses` with `my_evaluator.outputs`
    - Replace `my_evaluator.variables` with `my_evaluator.inputs`
    - Replace assertions on `my_evaluator.problem` with assertions on `my_evaluator.opt_problem`
    - _Requirements: 7.4, 7.5_
  - [x] 4.4 Search for and fix any remaining references to removed APIs across the test suite
    - Grep for `.variables`, `.responses`, `.problem`, `num_independent`, `num_dependent`, `_def_problem` in test files
    - Update any remaining references
    - _Requirements: 7.5_

- [x] 5. Checkpoint — Run full test suite
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Write property tests for the simplified interface
  - [x] 6.1 Write property test for OptProblem initialization
    - **Property 1: OptProblem initialization preserves interface**
    - Generate random valid OptProblem instances and verify inputs/outputs match
    - **Validates: Requirements 2.1**
  - [x] 6.2 Write property test for EvaluatorInfo initialization
    - **Property 2: EvaluatorInfo initialization preserves interface**
    - Generate random valid EvaluatorInfo instances and verify inputs/outputs match
    - **Validates: Requirements 2.2**
  - [x] 6.3 Write property test for create_opt_problem utility
    - **Property 3: create_opt_problem utility produces valid OptProblem**
    - Generate random positive integer pairs and verify the resulting OptProblem has correct variable/response counts
    - **Validates: Requirements 8.1**

- [x] 7. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster delivery
- `TestEvaluator` and its subclasses in `evaluators/test/` are NOT modified
- `create_opt_problem()` remains in `standard_evaluator.utilities` for external use
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
