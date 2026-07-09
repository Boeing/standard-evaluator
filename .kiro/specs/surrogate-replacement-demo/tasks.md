# Implementation Plan: Surrogate Replacement Demo

## Overview

This plan implements two bug fixes in `om_converter.py` and a Jupyter notebook demonstrating the surrogate replacement workflow. The bug fixes are implemented first (they unblock the notebook), followed by tests, then the notebook is built incrementally section by section. Python is used throughout, with Hypothesis for property-based tests.

## Tasks

- [x] 1. Fix get_linkages IndexError (Bug Fix 1)
  - [x] 1.1 Update `get_linkages()` in `om_converter.py` to handle 2-tuple connection metadata
    - Check tuple length before accessing elements beyond index 1
    - For 2-tuples, extract source from `value[0]` without accessing `value[2]`
    - For tuples with >2 elements, preserve existing diagnostic print logic
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 1.2 Write property test for get_linkages tuple handling
    - **Property 1: get_linkages correctly extracts linkages from any valid connection metadata**
    - Generate random connection metadata dictionaries with varying tuple lengths (2 and 3)
    - Verify `get_linkages` always returns correct `(source, target)` pairs without raising IndexError
    - Use `@settings(max_examples=100, deadline=None)`
    - **Validates: Requirements 1.1, 1.4**

  - [x] 1.3 Write unit tests for get_linkages
    - Test with mock Group having 2-tuple metadata (OpenMDAO 3.43 format)
    - Test with mock Group having 3-tuple metadata with None values (legacy format)
    - Test with mock Group having mixed 2-tuple and 3-tuple metadata
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Fix create_problem TypeError (Bug Fix 2)
  - [x] 2.1 Update `create_openmdao_options()` in `om_converter.py` to convert list values to tuples
    - Convert any list-valued option in `_dict[name]['val']` to a tuple before returning
    - Preserve element order and content during conversion
    - Do not alter non-list values (strings, ints, dicts, etc.)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 2.2 Write property test for create_openmdao_options list-to-tuple conversion
    - **Property 2: create_openmdao_options converts list-valued options to tuples**
    - Generate random `openmdao_options` `_dict` entries with list values of varying lengths
    - Verify all list values are converted to tuples, preserving element order and content
    - Use `@settings(max_examples=100, deadline=None)`
    - **Validates: Requirements 2.1, 2.2**

  - [x] 2.3 Write unit tests for create_openmdao_options
    - Test with `default_shape: [3]` (single-element list)
    - Test with `default_shape: [2, 4]` (multi-element list)
    - Test with non-list values (strings, ints) to verify they pass through unchanged
    - Test with aviary_options containing `__aviary_values__` key
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Checkpoint - Verify bug fixes
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Create notebook: assembly setup and interface capture
  - [x] 4.1 Create notebook file with imports and assembly creation cells
    - Create `docs/source/demos/surrogate_replacement_workflow.ipynb`
    - Add markdown cell explaining the workflow overview
    - Add imports cell: `import standard_evaluator as se`, `import openmdao.api as om`, numpy, pandas, matplotlib
    - Build 3-group assembly (aero, structures, performance) using ExecComp
    - Include internal `connect()` calls within aero sub-group (pressure_calc output to lift_calc and drag_calc)
    - Use promotions at top-level assembly
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 9.1, 9.2, 9.3, 9.5_

  - [x] 4.2 Add interface capture and aero extraction cells
    - Add markdown cell explaining interface capture
    - Call `se.get_interface()` on the assembly model to obtain GroupInfo
    - Display assembly structure using `se.show_structure()`
    - Extract aero sub-group as standalone Problem via `se.create_problem()`
    - Wrap extracted aero Problem as `OpenMDAOEvaluator` with `scan_model=True` and `use_defined_problem=False`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Create notebook: surrogate training and replacement
  - [x] 5.1 Add surrogate training cells
    - Add markdown cell explaining surrogate training
    - Generate training sites using `np.random.uniform` within aero evaluator bounds
    - Evaluate training sites using aero OpenMDAOEvaluator
    - Build `PolynomialModel(degree=2)` from training data
    - Wrap surrogate as OpenMDAO component using `EvaluatorOpenMdaoComponent`
    - Wrap surrogate component in Problem and then in OpenMDAOEvaluator
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.2 Add component replacement cells
    - Add markdown cell explaining replacement process
    - Capture surrogate evaluator's interface using `se.get_interface()`
    - Replace aero entry in original GroupInfo with surrogate's EvaluatorInfo
    - Instantiate modified assembly using `se.create_problem()`
    - Wrap modified assembly as OpenMDAOEvaluator
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 6. Create notebook: global surrogate and comparison
  - [x] 6.1 Add global surrogate cells
    - Add markdown cell explaining global surrogate approach
    - Wrap original assembly as OpenMDAOEvaluator
    - Generate training data by evaluating original assembly evaluator
    - Build `PolynomialModel(degree=2)` mapping full assembly inputs to all outputs
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 6.2 Add comparison and visualization cells
    - Add markdown cell explaining comparison methodology
    - Generate new test sites using `np.random.uniform` (distinct from training sites)
    - Evaluate all three evaluators (original, local-surrogate-replaced, global) on test sites
    - Compute error metrics using original as baseline truth
    - Display error summary table (mean absolute error and max absolute error per response per evaluator)
    - Display scatter plots of predicted vs actual values for each response
    - Display percent error box plots for each response grouped by evaluator
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 7. Checkpoint - Verify notebook execution
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Integration round-trip test
  - [x] 8.1 Write property test for interface serialization round-trip
    - **Property 3: Interface serialization round-trip preserves assembly functionality**
    - Generate random ExecComp-based assemblies with varying equation counts, shapes, and connection topologies
    - Verify `get_interface()` followed by `create_problem()` produces numerically equivalent outputs
    - Use `@settings(max_examples=100, deadline=None)`
    - **Validates: Requirements 2.4**

  - [x] 8.2 Write notebook execution test
    - Add `"surrogate_replacement_workflow.ipynb"` to the `BASE_NOTEBOOKS` list in `tests/test_notebook_execution.py`
    - This leverages the existing parametrized `test_notebook_executes_without_errors` test (uses `nbformat` + `ExecutePreprocessor`, `timeout=60`, `kernel_name="python3"`)
    - The test is already marked `@pytest.mark.integration` and `@pytest.mark.slow`
    - Run with: `.venv\Scripts\python.exe -m pytest tests/test_notebook_execution.py -k surrogate_replacement -v`
    - Do NOT use bare `pytest` or `python` — always use `.venv\Scripts\python.exe -m pytest` per project steering rules
    - _Requirements: 9.4_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The notebook is built incrementally to allow testing at each stage
- Python with Hypothesis is used for all property-based tests (project already uses Hypothesis)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.2", "2.3"] },
    { "id": 2, "tasks": ["4.1"] },
    { "id": 3, "tasks": ["4.2"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2"] },
    { "id": 6, "tasks": ["6.1"] },
    { "id": 7, "tasks": ["6.2"] },
    { "id": 8, "tasks": ["8.1", "8.2"] }
  ]
}
```
