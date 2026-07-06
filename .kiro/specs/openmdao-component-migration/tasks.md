# Implementation Plan: OpenMDAO Component Migration

## Overview

Migrate `EvaluatorOpenMdaoComponent` from boeing-standard-evaluator into the standard-evaluator library. The component wraps any `Evaluator` as an OpenMDAO `ExplicitComponent`, adapting from the legacy dict-based `problem` API to the pydantic `OptProblem` API. Implementation uses Python and targets the `standard_evaluator.components` subpackage.

## Tasks

- [x] 1. Create components subpackage and implement EvaluatorOpenMdaoComponent
  - [x] 1.1 Create the `standard_evaluator/components/` package with `__init__.py`
    - Create `src/standard_evaluator/components/__init__.py`
    - Export `EvaluatorOpenMdaoComponent` in `__all__`
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.2 Implement the `EvaluatorOpenMdaoComponent` class in `evaluator_om_component.py`
    - Create `src/standard_evaluator/components/evaluator_om_component.py`
    - Implement `__init__` with deep copy of evaluator, type checking, kwarg conflict detection, and fallback via `SurrogateModel.from_dict()`
    - Implement `initialize` to declare `evaluator_options` option when evaluator has `to_dict()`
    - Implement `setup` to map `opt_problem.variables` to OpenMDAO inputs and `opt_problem.responses` to OpenMDAO outputs with bounds, scaling, and defaults
    - Implement `compute` to build DataFrame from inputs, call evaluator, and extract responses to outputs
    - Handle all error cases: TypeError for non-Evaluator, AttributeError for missing evaluator/options, ValueError for discrete inputs/outputs, ValueError for scale=0.0
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 2. Checkpoint - Verify component module structure
  - Ensure the package imports resolve correctly, ask the user if questions arise.

- [x] 3. Write unit and integration tests
  - [x] 3.1 Create test directory structure and unit tests
    - Create `tests/components/__init__.py`
    - Create `tests/components/test_evaluator_om_component.py`
    - Test import resolution (`from standard_evaluator.components import EvaluatorOpenMdaoComponent`)
    - Test TypeError when non-Evaluator is passed as first arg
    - Test AttributeError when no evaluator and no evaluator_options provided
    - Test kwarg conflict detection raises clear error
    - Test ValueError for discrete_inputs not None
    - Test ValueError for discrete_outputs not None
    - Test ValueError for response with scale=0.0
    - Test exception propagation from SurrogateModel.from_dict with invalid options
    - Test exception propagation from evaluator during compute
    - _Requirements: 1.1, 1.2, 1.3, 2.2, 2.4, 3.2, 3.3, 6.6, 7.4, 7.5, 7.6_

  - [x] 3.2 Write integration tests with TestEvaluators
    - Test end-to-end with HS100 evaluator: setup, run_model, verify outputs match direct evaluator call within 1e-10 tolerance
    - Test SurrogateModel.from_dict reconstruction path
    - Test with multiple TestEvaluators from `standard_evaluator.evaluators.test`
    - _Requirements: 3.1, 8.1, 8.2, 8.3_

- [x] 4. Checkpoint - Ensure all unit and integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Write property-based tests
  - [x] 5.1 Write property test for deep copy isolation
    - **Property 1: Deep Copy Isolation**
    - Construct component with evaluator, mutate original evaluator's opt_problem, verify component's stored evaluator is unaffected
    - **Validates: Requirements 2.1**

  - [x] 5.2 Write property test for serialization option round trip
    - **Property 2: Serialization Option Round Trip**
    - Use SurrogateModel instances with to_dict, verify options["evaluator_options"] equals to_dict() result
    - **Validates: Requirements 4.2**

  - [x] 5.3 Write property test for variable-to-input default mapping
    - **Property 3: Variable-to-Input Default Mapping**
    - Generate FloatVariables with various defaults/bounds combinations, verify correct input default values
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5**

  - [x] 5.4 Write property test for response-to-output bounds mapping
    - **Property 4: Response-to-Output Bounds Mapping**
    - Generate response Variables with mix of finite/infinite bounds, verify OpenMDAO output bounds are correct
    - **Validates: Requirements 6.3, 6.4**

  - [x] 5.5 Write property test for response scaling formula
    - **Property 5: Response Scaling Formula**
    - Generate response Variables with arbitrary shift/scale (scale != 0), verify ref0 = -shift and ref = (1.0/scale) + ref0
    - **Validates: Requirements 6.5**

  - [x] 5.6 Write property test for compute round trip
    - **Property 6: Compute Round Trip**
    - Parametrize over TestEvaluators, verify component outputs match direct evaluator call within 1e-10 tolerance
    - **Validates: Requirements 7.1, 7.2, 7.3, 8.2, 8.3**

- [x] 6. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- **CRITICAL: All task execution MUST follow the workspace steering files** defined in `.kiro/steering/`, especially `.kiro/steering/python-environment.md`. This means:
  - NEVER use `python`, `python3`, `py`, or any system Python directly
  - ALWAYS use `.venv\Scripts\python.exe` for all Python commands
  - For pytest: `.venv\Scripts\python.exe -m pytest`
  - For pip installs: `.venv\Scripts\python.exe -m pip install ...`
  - Working directory: `c:\dev\standard-evaluator`
- Also follow `.kiro/steering/git-rules.md` for any git operations
- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses the pydantic `OptProblem` API (`opt_problem.variables`, `opt_problem.responses`) instead of the legacy dict-based `problem` API
- Test file for property-based tests: `tests/components/test_pbt_evaluator_om_component.py`
- Hypothesis is already available as a test dependency in pyproject.toml

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["3.1", "3.2"] },
    { "id": 3, "tasks": ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6"] }
  ]
}
```
