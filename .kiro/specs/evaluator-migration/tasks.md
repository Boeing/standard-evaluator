# Implementation Plan: Evaluator Migration

## Overview

Migrate the evaluator class hierarchy from `boeing_standard_evaluator` into `standard_evaluator`, making the target library independently functional. The implementation proceeds bottom-up: utilities first, then the base class, concrete evaluators, decorators, test evaluators, OptProblem extension, package exports, and finally tests.

## Tasks

- [x] 1. Create utilities package and migrate utility modules
  - [x] 1.1 Create `src/standard_evaluator/utilities/` package with `__init__.py`
    - Create the directory structure and `__init__.py`
    - Copy and migrate `mapping.py`, `option_merge.py`, `problem_dict_utility.py`, `se_arrays.py`, `shift_scale.py`, `utility.py` from `boeing_standard_evaluator/src/boeing_standard_evaluator/utilities/`
    - Replace all `boeing_standard_evaluator` import paths with `standard_evaluator`
    - Ensure `utilities/__init__.py` re-exports all public symbols from each module
    - Resolve any transitive utility dependencies within the utilities package
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.6_

- [x] 2. Extend OptProblem with build_maps and _setup_partials
  - [x] 2.1 Implement `build_maps` method on OptProblem class
    - Add `build_maps()` method to `src/standard_evaluator/problem.py` that returns `(var_map, res_map)` tuple of DataFrames
    - `var_map` columns: `name`, `multi`, `flat`, `fixed`, `jac_row`, `grad_row`
    - `res_map` columns: `name`, `multi`, `flat`, `objective`, `constraint`, `grad_col`, `jac_col`
    - Mark variables as fixed when lower bound equals upper bound; set `jac_row`/`grad_row` to None for fixed elements
    - _Requirements: 17.1, 17.7_

  - [x] 2.2 Implement `_setup_partials` method on OptProblem class
    - Add `_setup_partials()` method that reads `var_map` and `res_map`
    - Stores private attributes: `_flat_partials_res_indices`, `_num_partials_responses`, `_grad_cols`, `_jac_cols`, `_num_objs_total`, `_num_cons_total`, `_free_flat_var_mask`, `_free_flat_var_positions`, `_num_flat_vars`, `_full_to_free_var_index`
    - _Requirements: 17.2_

  - [x] 2.3 Add read-only properties and model_validator to OptProblem
    - Add read-only properties: `var_map`, `res_map`, `flat_partials_res_indices`, `num_partials_responses`, `grad_cols`, `jac_cols`, `num_objs_total`, `num_cons_total`, `free_flat_var_mask`, `free_flat_var_positions`, `num_flat_vars`, `full_to_free_var_index`
    - Add `model_validator(mode="after")` named `_init_maps_and_partials` that calls `build_maps` then `_setup_partials`
    - Wrap errors in `ValueError` with problem name context
    - Ensure existing `check_problem` validator still executes first
    - _Requirements: 17.3, 17.4, 17.5, 17.6, 17.8_

  - [x] 2.4 Write property tests for OptProblem build_maps and _setup_partials
    - **Property 8: build_maps structural correctness**
    - **Property 9: Fixed variable exclusion in var_map**
    - **Property 10: OptProblem maps initialized after construction**
    - **Property 11: OptProblem backward compatibility**
    - **Validates: Requirements 17.1, 17.2, 17.4, 17.7, 17.8**

- [x] 3. Create evaluators package structure and migrate abstract base class
  - [x] 3.1 Create `src/standard_evaluator/evaluators/` package with `__init__.py`
    - Create the `evaluators/` directory and a minimal `__init__.py` (will be populated later)
    - Create `evaluators/decorators/` directory with `__init__.py`
    - Create `evaluators/test/` directory with `__init__.py`
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

  - [x] 3.2 Migrate `abstract_evaluator.py` into `src/standard_evaluator/evaluators/`
    - Copy from `boeing_standard_evaluator/src/boeing_standard_evaluator/evaluators/abstract_evaluator.py`
    - Replace all `boeing_standard_evaluator` imports with `standard_evaluator` equivalents per the design import mapping table
    - Verify public interface: `__call__`, `eval_np`, `eval_list`, `default_site`, `initial_guess`, all properties, `_evaluate` abstract method
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.3 Write unit tests for Evaluator base class imports and interface
    - Verify `Evaluator` is importable from `standard_evaluator.evaluators`
    - Verify public methods and properties exist on the class
    - Verify no `boeing_standard_evaluator` in import chain
    - _Requirements: 2.5, 2.6_

- [x] 4. Migrate decorators subfolder
  - [x] 4.1 Migrate `cache.py` and `site_logger.py` into `src/standard_evaluator/evaluators/decorators/`
    - Copy `cache.py` from `boeing_standard_evaluator` and update all imports
    - Copy `site_logger.py` from `boeing_standard_evaluator` and update all imports
    - Update `decorators/__init__.py` to export `cacher` and `site_logger`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [x] 5. Migrate concrete evaluators
  - [x] 5.1 Migrate PyEvaluator (`evaluator.py`)
    - Copy `evaluator.py` into `src/standard_evaluator/evaluators/evaluator.py`
    - Replace all `boeing_standard_evaluator` imports with `standard_evaluator`
    - Ensure it inherits from migrated `Evaluator` base class
    - Verify TypeError is raised for non-callable `func` argument
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6_

  - [x] 5.2 Migrate NumpyEvaluator (`numpy_evaluator.py`)
    - Copy `numpy_evaluator.py` into `src/standard_evaluator/evaluators/numpy_evaluator.py`
    - Replace all `boeing_standard_evaluator` imports, including utility references (`unroll_data_frame_numpy`, `unroll_names_using_variables`, `roll_data_frame_using_variables`)
    - Ensure it inherits from migrated `Evaluator`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_

  - [x] 5.3 Migrate ExecutableEvaluator (`executable_evaluator.py`)
    - Copy `executable_evaluator.py` into `src/standard_evaluator/evaluators/executable_evaluator.py`
    - Replace all `boeing_standard_evaluator` imports, preserve existing `standard_evaluator` imports (e.g., `ArrayVariable`)
    - Migrate any utility functions not yet available in `standard_evaluator`
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6_

  - [x] 5.4 Migrate ShiftScaleEvaluator (`shift_scale_evaluator.py`)
    - Copy `shift_scale_evaluator.py` into `src/standard_evaluator/evaluators/shift_scale_evaluator.py`
    - Replace all `boeing_standard_evaluator` imports including `shift_scale`, `problem_dict_utility`, `opt_problem`, `evaluator_info`, `conversions`
    - Verify TypeError for non-Evaluator `evaluate` argument
    - Verify ValueError when no interface args provided
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.6_

  - [x] 5.5 Migrate OpenMDAOEvaluator (`open_mdao_evaluator.py`)
    - Copy `open_mdao_evaluator.py` into `src/standard_evaluator/evaluators/open_mdao_evaluator.py`
    - Replace all `boeing_standard_evaluator` imports
    - Verify ValueError when `scan_model=False` and `use_defined_problem=False`
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6_

  - [x] 5.6 Migrate ExcelEvaluator and excel_utilities
    - Copy `excel_evaluator.py` into `src/standard_evaluator/evaluators/excel_evaluator.py`
    - Copy `excel_utilities.py` into `src/standard_evaluator/evaluators/excel_utilities.py`
    - Replace all `boeing_standard_evaluator` imports in both files
    - Ensure `SpreadsheetModel`, `MacroDefinition`, `VarType`, `get_interface_from_excel_named_ranges`, `create_variable_from_excel_address`, `range_shape` are importable
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 5.7 Migrate MatlabEvaluator (`matlab_evaluator.py`)
    - Copy `matlab_evaluator.py` into `src/standard_evaluator/evaluators/matlab_evaluator.py`
    - Replace all `boeing_standard_evaluator` imports
    - Preserve try/except pattern for optional `matlab.engine` import
    - _Requirements: 9.1, 9.2, 9.5_

  - [x] 5.8 Write property tests for PyEvaluator and constructor validation
    - **Property 2: PyEvaluator functional equivalence**
    - **Property 3: Evaluator constructor input validation**
    - **Validates: Requirements 3.4, 3.5, 6.5**

  - [x] 5.9 Write property tests for ShiftScaleEvaluator
    - **Property 5: ShiftScaleEvaluator transform equivalence**
    - **Property 6: ShiftScaleEvaluator Jacobian consistency**
    - **Validates: Requirements 6.4, 6.7**

- [x] 6. Checkpoint - Verify core evaluator imports
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Migrate TestEvaluator base class and benchmark evaluators
  - [x] 7.1 Migrate `test_evaluator.py` into `src/standard_evaluator/evaluators/test_evaluator.py`
    - Copy `test_evaluator.py` from `boeing_standard_evaluator`
    - Replace all `boeing_standard_evaluator` imports
    - Ensure it inherits from migrated `Evaluator` base class
    - Verify `known_solution` property, `_create_opt_problem` abstract method, and `__test__ = False` attribute retained
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 7.2 Migrate all 38 benchmark test evaluator files into `src/standard_evaluator/evaluators/test/`
    - Copy all 38 files listed in Requirement 12.1
    - Do NOT copy `costa_five.py`, `moo_costa_five.py`, or `vectorized_flight_aerodynamics_performance_model.py`
    - Replace all `boeing_standard_evaluator` imports and string references with `standard_evaluator`
    - _Requirements: 12.1, 12.2, 12.4_

  - [x] 7.3 Implement auto-discovery in `evaluators/test/__init__.py`
    - Use `importlib` and `inspect` to auto-discover all test evaluator classes
    - Populate `__all__` with discovered class names (excluding `TestEvaluator` base class)
    - Raise `ImportError` if any of the 38 classes cannot be discovered
    - _Requirements: 12.3, 12.6_

  - [x] 7.4 Write property tests for test evaluator independence
    - **Property 1: Test evaluator independence and correctness**
    - **Property 4: NumpyEvaluator numerical equivalence**
    - **Validates: Requirements 2.5, 4.5, 12.5, 13.5, 16.3**

- [x] 8. Update evaluators package __init__.py with full exports
  - [x] 8.1 Populate `evaluators/__init__.py` with all evaluator exports
    - Import and re-export: `Evaluator`, `PyEvaluator`, `NumpyEvaluator`, `ExecutableEvaluator`, `ShiftScaleEvaluator`, `OpenMDAOEvaluator`, `ExcelEvaluator`, `TestEvaluator`
    - Import and re-export excel_utilities public names: `SpreadsheetModel`, `MacroDefinition`, `VarType`, `get_interface_from_excel_named_ranges`, `create_variable_from_excel_address`
    - Handle `MatlabEvaluator` with try/except and set `can_use_matlab` flag
    - Import all test evaluator classes via `test.__all__`
    - Populate `__all__` with all exported names
    - _Requirements: 1.4, 1.5, 1.6, 9.3, 9.4_

  - [x] 8.2 Update `standard_evaluator/__init__.py` to expose evaluators subpackage
    - Import the `evaluators` subpackage
    - Add `"evaluators"` to `__all__`
    - Verify existing exports (`StandardBase`, `StandardEval`, `StandardGroup`, `EvaluatorInfo`, `OptProblem`) remain intact
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [x] 9. Checkpoint - Verify all imports and independence
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Migrate evaluator tests and fixtures
  - [x] 10.1 Copy all test files from `tests/evaluators/` in boeing_standard_evaluator to `tests/evaluators/` in standard_evaluator
    - Preserve directory structure including `test/` subdirectory
    - Replace all `boeing_standard_evaluator` import paths with `standard_evaluator` in every `.py` file
    - _Requirements: 14.1, 14.2, 14.5_

  - [x] 10.2 Copy test fixture files to standard_evaluator test directory
    - Copy `Range_example.xlsm`, `test_spreadsheet.xlsm`, `test_generic_airplane.aom`
    - Copy the entire `Executable Evaluator/` folder with contents
    - Place them in the corresponding location under `tests/evaluators/`
    - _Requirements: 14.3_

  - [x] 10.3 Write source independence property test
    - **Property 7: Source code independence from Boeing references**
    - Scan all `.py` files under `src/standard_evaluator/` for `boeing_standard_evaluator` references and Boeing proprietary notices
    - **Validates: Requirements 16.3, 16.4**

- [x] 11. Verify independence constraint
  - [x] 11.1 Verify `pyproject.toml` has no boeing_standard_evaluator dependency
    - Check `[project.dependencies]`, `[project.optional-dependencies]`, and `[build-system.requires]` sections
    - Verify no reference to `boeing_standard_evaluator` exists
    - _Requirements: 16.1_

  - [x] 11.2 Verify all source files are free of boeing_standard_evaluator references
    - Run grep/search across all `src/standard_evaluator/` Python files
    - Confirm zero occurrences of `boeing_standard_evaluator` as import or reference
    - Confirm zero occurrences of Boeing proprietary notice text
    - _Requirements: 16.2, 16.3, 16.4, 16.5_

- [x] 12. Final checkpoint - Run full test suite
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The migration is copy-based; the source library retains its files for backward compatibility
- MatlabEvaluator import is guarded by try/except; tests for it only run when `matlab.engine` is available
- All utility modules must be migrated before evaluator classes that depend on them

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1"] },
    { "id": 1, "tasks": ["2.1", "3.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.3", "4.1"] },
    { "id": 3, "tasks": ["2.4", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7"] },
    { "id": 4, "tasks": ["5.8", "5.9", "7.1"] },
    { "id": 5, "tasks": ["7.2"] },
    { "id": 6, "tasks": ["7.3", "7.4"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["8.2"] },
    { "id": 9, "tasks": ["10.1", "10.2"] },
    { "id": 10, "tasks": ["10.3", "11.1", "11.2"] }
  ]
}
```
