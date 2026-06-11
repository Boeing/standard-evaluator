# Implementation Plan: Optional Aviary Dependency

## Overview

Make `aviary` and `dymos` optional dependencies by moving them to an extras group in `pyproject.toml` and adding lazy import guards to the three modules that use aviary types (`aviary_encoder.py`, `standard_evaluator.py`, `om_converter.py`). The package will import cleanly without aviary, and aviary-dependent code paths raise a clear `ImportError` at execution time.

## Tasks

- [x] 1. Update package configuration
  - [x] 1.1 Move aviary/dymos from dependencies to optional-dependencies in pyproject.toml
    - Remove `"aviary"` from the `dependencies` array
    - Add a new `aviary` extras group under `[project.optional-dependencies]` containing `"aviary"` and `"dymos"`
    - Verify `dymos` is not listed in the hard `dependencies` array
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 2. Add lazy import guard to aviary_encoder.py
  - [x] 2.1 Implement try/except guard and _require_aviary() helper in aviary_encoder.py
    - Wrap all `import aviary.*` and `import dymos` statements in a try/except block
    - Set `_AVIARY_AVAILABLE = True` on success, `_AVIARY_AVAILABLE = False` on `ImportError`
    - Set sentinel values (`None`) for aviary/dymos types when unavailable
    - Add `_require_aviary()` helper function that raises `ImportError` with message containing `pip install standard-evaluator[aviary]`
    - Add `__init__` method to `AviaryEncoder` that calls `_require_aviary()` before `super().__init__()`
    - _Requirements: 1.1, 3.1, 5.1, 5.2, 5.3_

- [x] 3. Add lazy import guard to standard_evaluator.py
  - [x] 3.1 Implement try/except guard and _require_aviary() helper in standard_evaluator.py
    - Wrap `from aviary.variable_info.variable_meta_data import _MetaData` in try/except
    - Set `_AVIARY_AVAILABLE = True/False` and `_MetaData = None` sentinel
    - Add `_require_aviary()` helper function with consistent error message
    - Call `_require_aviary()` at the start of `StandardEval.initialize()` before using `_MetaData`
    - _Requirements: 1.1, 3.4, 5.1, 5.2, 5.3_

- [x] 4. Add lazy import guard to om_converter.py
  - [x] 4.1 Implement try/except guard and _require_aviary() helper in om_converter.py
    - Wrap `from aviary.utils.aviary_values import AviaryValues` and `from aviary.subsystems.propulsion.engine_deck import EngineDeck` in try/except
    - Set `_AVIARY_AVAILABLE = True/False` and sentinel values for `AviaryValues` and `EngineDeck`
    - Add `_require_aviary()` helper function with consistent error message
    - Add `_require_aviary()` call at the start of `convert_aviary()`, `convert_engine_deck()`, and `convert_enum()`
    - _Requirements: 1.1, 3.2, 3.3, 5.1, 5.2, 5.3_

- [x] 5. Verify package imports cleanly without aviary
  - [x] 5.1 Ensure __init__.py imports succeed without aviary installed
    - Verify that `import standard_evaluator` succeeds when aviary is absent (use mock patching to simulate)
    - Confirm all non-aviary symbols in `__all__` are accessible without error
    - Fix any transitive import issues in the import chain
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 6. Checkpoint - Ensure core implementation is correct
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Write property-based tests
  - [x] 7.1 Write property test for lazy guard ImportError on all entry points
    - **Property 1: Lazy guard raises with install instructions for all guarded entry points**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 5.2, 5.3**
    - Use `hypothesis` with `@settings(max_examples=100)`
    - Generate random selections from {AviaryEncoder, StandardEval, convert_aviary, convert_engine_deck}
    - For each, mock aviary absence, verify module import succeeds, verify invocation raises `ImportError` with "pip install standard-evaluator[aviary]" in message

  - [x] 7.2 Write property test for AviaryEncoder type-marker round trip
    - **Property 2: AviaryEncoder type-marker round trip**
    - **Validates: Requirements 4.1**
    - Use `hypothesis` with `@settings(max_examples=100)`
    - Generate random aviary-typed objects (Enum values, AviaryValues dicts, WindowsPath, numpy arrays, sets, tuples)
    - Verify `AviaryEncoder().default(obj)` returns a dict containing the expected type-marker key

- [x] 8. Write example-based unit tests
  - [x] 8.1 Write unit tests for guarded import behavior
    - Test that `import standard_evaluator` succeeds with aviary mocked away
    - Test that all non-aviary symbols in `__all__` are accessible without aviary
    - Test that `AviaryEncoder()` raises `ImportError` when aviary is absent
    - Test that `StandardEval` instantiation raises `ImportError` when aviary is absent
    - Test that `convert_aviary({})` raises `ImportError` when aviary is absent
    - Test that `convert_engine_deck({})` raises `ImportError` when aviary is absent
    - Test that all error messages contain "pip install standard-evaluator[aviary]"
    - _Requirements: 1.1, 1.2, 3.1, 3.2, 3.3, 3.4, 5.3_

  - [x] 8.2 Write unit tests for backward compatibility with aviary present
    - Test that `StandardEval().options["metadata"]` equals `_MetaData` when aviary is installed
    - Test that `AviaryEncoder` serializes aviary types correctly when aviary is installed
    - Test that `convert_aviary` and `convert_engine_deck` work identically when aviary is installed
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Use `unittest.mock.patch.dict(sys.modules, ...)` combined with `importlib.reload()` to simulate aviary absence in tests
- The `_require_aviary()` helper must raise `ImportError` (not `RuntimeError`) for ecosystem consistency
- Tasks 2, 3, and 4 are independent and can be implemented in parallel after task 1
- Property tests use `hypothesis` which is already a test dependency in the project

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["5.1"] },
    { "id": 3, "tasks": ["7.1", "7.2", "8.1", "8.2"] }
  ]
}
```
