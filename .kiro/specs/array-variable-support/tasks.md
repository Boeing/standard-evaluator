# Implementation Plan: Array Variable Support

## Overview

Extend `OpenMDAOEvaluator` and `EvaluatorOpenMdaoComponent` to handle `ArrayVariable` instances alongside `FloatVariable`. Changes are localized to `open_mdao_evaluator.py` and `evaluator_om_component.py`, with new property-based and unit tests validating correctness.

## Tasks

- [x] 1. Update `_expand_info_to_variable` to produce `ArrayVariable` for array shapes
  - [x] 1.1 Modify `_expand_info_to_variable` in `src/standard_evaluator/evaluators/open_mdao_evaluator.py`
    - Import `ArrayVariable` from `standard_evaluator.problem`
    - Remove the `TypeError` raised when `shape != (1,)`
    - Add branch: if `shape != (1,)`, create `ArrayVariable` with detected shape, bounds `(-inf, +inf)`, shift/scale from `om_utils.determine_adder_scaler` (default shift=0.0, scale=1.0 if no scaling metadata), default from `val` array or `np.zeros(shape)`, and units from metadata if present
    - Preserve existing scalar `FloatVariable` path for `shape == (1,)`, adding units support
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 1.2 Write property test for `_expand_info_to_variable` ArrayVariable production
    - **Property 1: Scan produces ArrayVariable with correct fields for array shapes**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7**
    - Create test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for random shapes (not `(1,)`), scaling metadata, val arrays, and units strings

  - [x] 1.3 Write property test for `_expand_info_to_variable` FloatVariable production
    - **Property 2: Scan produces FloatVariable with correct fields for scalar shapes**
    - **Validates: Requirements 1.5, 7.3**
    - Create test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for scalar info dicts with `shape == (1,)` and optional units

- [x] 2. Update `_map_elements` to produce `ArrayVariable` for `size > 1`
  - [x] 2.1 Modify `_map_elements` in `src/standard_evaluator/evaluators/open_mdao_evaluator.py`
    - Check `size` in each element's metadata
    - If `size > 1`: produce `ArrayVariable` with shape, bounds (lower/upper from metadata, defaulting independently to -inf/+inf), scale (scaler), shift (adder), and units from metadata
    - If `size == 1` or absent: produce `FloatVariable` with bounds and units from metadata
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 Write property test for `_map_elements` ArrayVariable production
    - **Property 3: Map produces ArrayVariable for size > 1**
    - **Validates: Requirements 2.1, 2.2, 2.4, 2.5**
    - Create test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for metadata dicts with `size > 1`, optional units

  - [x] 2.3 Write property test for `_map_elements` FloatVariable production
    - **Property 4: Map produces FloatVariable for size ≤ 1**
    - **Validates: Requirements 2.3, 2.5**
    - Create test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for metadata dicts with `size == 1` or absent, optional units

- [x] 3. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Update `_evaluate` to handle mixed scalar/array data
  - [x] 4.1 Modify `_evaluate` in `src/standard_evaluator/evaluators/open_mdao_evaluator.py`
    - For each input, check the corresponding variable type in `self.opt_problem.variables`
    - If `ArrayVariable`: call `set_val` with the full NumPy array from the DataFrame cell
    - If `FloatVariable`: call `set_val` with scalar value (unchanged)
    - For each output, check the corresponding response type in `self.opt_problem.responses`
    - If `ArrayVariable`: store full `get_val` NumPy array in DataFrame cell without flattening
    - If `FloatVariable`: store scalar from `get_val` (unchanged)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.2 Write property test for `_evaluate` data flow
    - **Property 5: Evaluate correctly passes and extracts mixed scalar/array data**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    - Create test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for random OptProblems with mixed types, mock `om_problem`

- [x] 5. Update `EvaluatorOpenMdaoComponent.setup()` for array inputs
  - [x] 5.1 Modify input registration in `src/standard_evaluator/components/evaluator_om_component.py`
    - Import `ArrayVariable` from `standard_evaluator.problem`
    - If variable is `ArrayVariable`: compute `val` from `var.default` (or midpoint of bounds if None), call `add_input(name, shape=var.shape, val=val)`, pass `units` if not None
    - If variable is `FloatVariable`: preserve existing logic, add `units` support (pass if not None, omit if None)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 5.2 Write property test for component input registration
    - **Property 6: Component setup registers inputs correctly with units**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.6, 4.7**
    - Create test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for random ArrayVariables and FloatVariables with optional units, mock `add_input`

- [x] 6. Update `EvaluatorOpenMdaoComponent.setup()` for array outputs
  - [x] 6.1 Modify output registration in `src/standard_evaluator/components/evaluator_om_component.py`
    - If response is `ArrayVariable`: compute val as `np.zeros(shape)`, determine lower/upper (scalar if uniform, array if mixed, omit if all-inf), compute ref0/ref from shift/scale if non-default, raise `ValueError` if any scale is 0, pass units if not None
    - If response is `FloatVariable`: preserve existing logic, add `units` support (pass if not None, omit if None)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 6.2 Write property test for component output registration
    - **Property 7: Component setup registers outputs correctly with units**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    - Add test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for random ArrayVariable and FloatVariable responses with optional units, mock `add_output`

  - [x] 6.3 Write property test for scaling conversion round-trip
    - **Property 9: Scaling conversion round-trip**
    - **Validates: Requirements 5.5**
    - Verify that converting (shift, scale) → (ref0, ref) → (shift, scale) produces original values within floating-point tolerance
    - Use Hypothesis strategies for random non-zero scale arrays and shift arrays

- [x] 7. Update `EvaluatorOpenMdaoComponent.compute()` for array marshaling
  - [x] 7.1 Modify `compute` in `src/standard_evaluator/components/evaluator_om_component.py`
    - For `ArrayVariable` inputs: place `inputs[name]` (NumPy array) directly into the DataFrame cell
    - For `FloatVariable` inputs: place scalar value from `inputs[name]` (unchanged)
    - For `ArrayVariable` outputs: assign DataFrame cell (NumPy array) to `outputs[name]`
    - For `FloatVariable` outputs: assign scalar from DataFrame to `outputs[name]` (unchanged)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 7.2 Write property test for component compute marshaling
    - **Property 8: Component compute marshals mixed scalar/array data correctly**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    - Add test in `tests/components/test_pbt_evaluator_om_component.py`
    - Use Hypothesis strategies for random mixed OptProblems, mock evaluator

- [x] 8. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Backward compatibility and integration tests
  - [x] 9.1 Write backward compatibility unit tests
    - Add test in `tests/evaluators/test_openmdao_evaluator.py` verifying scalar-only models produce identical `FloatVariable` instances and evaluated output
    - Add test in `tests/components/test_evaluator_om_component.py` verifying scalar-only OptProblems register identical inputs/outputs
    - Verify no new mandatory constructor parameters are required
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 9.2 Write integration test for mixed scalar/array round-trip
    - Create OpenMDAO model with both scalar and array variables
    - Wrap with `OpenMDAOEvaluator` → verify correct OptProblem
    - Wrap with `EvaluatorOpenMdaoComponent` → run in OpenMDAO group
    - Verify outputs match direct model execution
    - _Requirements: 3.5, 6.4, 7.3_

- [x] 10. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- **CRITICAL: All task execution MUST follow the workspace steering files** defined in `.kiro/steering/`, especially `.kiro/steering/python-environment.md`. This means:
  - NEVER use `python`, `python3`, `py`, or any system Python directly
  - ALWAYS use `.venv\Scripts\python.exe` for all Python commands
  - For pytest: `.venv\Scripts\python.exe -m pytest`
  - For pip installs: `.venv\Scripts\python.exe -m pip install ...`
  - Working directory: `c:\dev\standard-evaluator`
- Also follow `.kiro/steering/git-rules.md` for any git operations
- The implementation uses the Pydantic `OptProblem` API (`opt_problem.variables`, `opt_problem.responses`) instead of the legacy dict-based `problem` API
- Key source files: `src/standard_evaluator/evaluators/open_mdao_evaluator.py`, `src/standard_evaluator/components/evaluator_om_component.py`
- Test file for property-based tests: `tests/components/test_pbt_evaluator_om_component.py`
- Hypothesis is already available as a test dependency in pyproject.toml
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.1"] },
    { "id": 3, "tasks": ["4.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "7.1"] },
    { "id": 6, "tasks": ["7.2", "9.1"] },
    { "id": 7, "tasks": ["9.2"] }
  ]
}
```
