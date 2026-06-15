# Requirements Document

## Introduction

This feature migrates evaluator classes from the `boeing_standard_evaluator` library into `standard_evaluator`. The evaluator hierarchy (abstract base class, concrete evaluators, decorators, and analytic test functions) will be duplicated into `standard_evaluator` so that it becomes independently functional without depending on `boeing_standard_evaluator`. The existing classes in `standard_evaluator` (StandardBase, StandardEval, StandardGroup, EvaluatorInfo, OptProblem) remain unchanged.

## Glossary

- **Standard_Evaluator**: The target Python library at `c:\dev\standard-evaluator` with source in `src/standard_evaluator/`.
- **Boeing_Standard_Evaluator**: The source Python library at `c:\dev\boeing-standard-evaluator` with source in `src/boeing_standard_evaluator/`.
- **Evaluator**: The abstract base class defining the evaluator interface (from `abstract_evaluator.py`).
- **PyEvaluator**: A concrete evaluator that wraps a Python callable (from `evaluator.py`).
- **NumpyEvaluator**: A concrete evaluator that operates on NumPy arrays (from `numpy_evaluator.py`).
- **ExecutableEvaluator**: A concrete evaluator that wraps external executable processes (from `executable_evaluator.py`).
- **ShiftScaleEvaluator**: A concrete evaluator that applies shift and scale transformations (from `shift_scale_evaluator.py`).
- **OpenMDAOEvaluator**: A concrete evaluator that wraps OpenMDAO components (from `open_mdao_evaluator.py`).
- **ExcelEvaluator**: A concrete evaluator that interfaces with Excel workbooks (from `excel_evaluator.py` and `excel_utilities.py`).
- **MatlabEvaluator**: A concrete evaluator that interfaces with MATLAB via optional import (from `matlab_evaluator.py`).
- **TestEvaluator**: The abstract base class for analytic benchmark test evaluators (from `test_evaluator.py`).
- **Decorators_Subfolder**: The `evaluators/decorators/` package containing `cache.py` and `site_logger.py`.
- **Test_Subfolder**: The `evaluators/test/` package containing approximately 40 analytic benchmark function evaluators.
- **Migration**: The process of duplicating source code from Boeing_Standard_Evaluator into Standard_Evaluator.

## Requirements

### Requirement 1: Create Evaluators Package Structure

**User Story:** As a developer, I want the evaluators package structure created in standard_evaluator, so that I can organize migrated evaluator classes in the same subfolder layout as the source library.

#### Acceptance Criteria

1. THE Migration SHALL create an `evaluators/` package directory under `src/standard_evaluator/` with an `__init__.py` file that is syntactically valid Python.
2. THE Migration SHALL create an `evaluators/decorators/` package directory under `src/standard_evaluator/` with an `__init__.py` file that is syntactically valid Python.
3. THE Migration SHALL create an `evaluators/test/` package directory under `src/standard_evaluator/` with an `__init__.py` file that is syntactically valid Python.
4. WHEN the `evaluators/` package is imported, THE `evaluators/__init__.py` SHALL export every evaluator class defined in modules directly under `src/standard_evaluator/evaluators/` in its `__all__` list, matching the class names exactly as defined in those modules.
5. WHEN a Python interpreter imports `standard_evaluator.evaluators`, `standard_evaluator.evaluators.decorators`, or `standard_evaluator.evaluators.test`, THE import SHALL complete without raising an `ImportError` or `SyntaxError`.
6. WHEN the `evaluators/decorators/` or `evaluators/test/` subpackage is imported, THE respective `__init__.py` SHALL export the public classes and functions defined in that subpackage's modules in its `__all__` list.

### Requirement 2: Migrate Abstract Evaluator Base Class

**User Story:** As a developer, I want the Evaluator abstract base class available in standard_evaluator, so that all concrete evaluators can inherit from it without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL place the Evaluator base class source file at `src/standard_evaluator/evaluators/abstract_evaluator.py`.
2. THE copied Evaluator class SHALL expose the same public interface as the source, including: all public methods (`__call__`, `eval_np`, `eval_list`, `default_site`, `initial_guess`), all public properties (`problem`, `opt_problem`, `interface`, `variables`, `responses`, `inputs`, `outputs`, `nind`, `ndep`, `name`, `comp_cost`), the abstract method `_evaluate`, and the same constructor parameter signature.
3. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` module paths.
4. IF a dependency module referenced by abstract_evaluator.py does not yet exist in `standard_evaluator`, THEN THE Migration SHALL create a stub or migrate that module so that the import resolves without error and the migrated class remains fully functional with all its public methods and interface intact.
5. WHEN the Evaluator class is instantiated by a subclass and its `__call__` method is invoked with a valid DataFrame, THE Evaluator SHALL execute without importing any module from `boeing_standard_evaluator`.
6. WHEN a developer runs the test suite for the migrated Evaluator class, THE tests SHALL pass with zero references to `boeing_standard_evaluator` in the import chain.

### Requirement 3: Migrate PyEvaluator

**User Story:** As a developer, I want the PyEvaluator class available in standard_evaluator, so that I can wrap Python callables as evaluators without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `evaluator.py` into `src/standard_evaluator/evaluators/evaluator.py`.
2. THE copied PyEvaluator class SHALL inherit from the migrated Evaluator base class.
3. THE copied file SHALL update all internal import paths from `boeing_standard_evaluator` to `standard_evaluator`.
4. WHEN PyEvaluator is called with a valid DataFrame containing columns matching the evaluator's declared inputs, THE PyEvaluator SHALL produce the same outputs as the Boeing_Standard_Evaluator version.
5. IF the `func` argument passed to PyEvaluator is not callable and is not None, THEN THE PyEvaluator SHALL raise a TypeError immediately during initialization, preventing the PyEvaluator instance from being created.
6. WHEN PyEvaluator is imported from `standard_evaluator.evaluators`, THE import SHALL resolve without raising `ImportError`.

### Requirement 4: Migrate NumpyEvaluator

**User Story:** As a developer, I want the NumpyEvaluator class available in standard_evaluator, so that I can use NumPy-based evaluators without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `numpy_evaluator.py` into `src/standard_evaluator/evaluators/numpy_evaluator.py`.
2. THE copied NumpyEvaluator class SHALL inherit from the migrated Evaluator base class located at `standard_evaluator.evaluators.abstract_evaluator.Evaluator`.
3. THE copied file SHALL update all internal import paths from `boeing_standard_evaluator` to `standard_evaluator`, including imports of utility functions (`unroll_data_frame_numpy`, `unroll_names_using_variables`, `roll_data_frame_using_variables`).
4. THE migrated NumpyEvaluator SHALL preserve the same public API as the source, including the `_evaluate(sites: pd.DataFrame, **kwargs)` method, the abstract `eval_np(sites: NDArray[np.float64], names: list, **kwargs)` method, and the `dataframe_to_float_ndarray(df: pd.DataFrame)` method.
5. WHEN NumpyEvaluator is called with a DataFrame whose columns match the evaluator's declared inputs, THE NumpyEvaluator SHALL produce outputs numerically identical (within tolerance of 1e-12) to those produced by the Boeing_Standard_Evaluator version given the same input DataFrame.
6. IF NumpyEvaluator is imported from `standard_evaluator.evaluators`, THEN THE import SHALL resolve without raising `ImportError` or `ModuleNotFoundError`.

### Requirement 5: Migrate ExecutableEvaluator

**User Story:** As a developer, I want the ExecutableEvaluator class available in standard_evaluator, so that I can wrap external executables as evaluators without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `executable_evaluator.py` into `src/standard_evaluator/evaluators/executable_evaluator.py`.
2. THE copied ExecutableEvaluator class SHALL inherit from the migrated Evaluator base class in `standard_evaluator`.
3. THE copied file SHALL replace all `boeing_standard_evaluator` import references with their `standard_evaluator` equivalents, and SHALL preserve existing direct `standard_evaluator` imports (e.g., `ArrayVariable`) unchanged.
4. WHEN ExecutableEvaluator is instantiated with an `EvaluatorInfo` or `OptProblem` interface definition, a valid executable path, and a run directory, THE migrated ExecutableEvaluator SHALL produce output DataFrames with values matching the Boeing_Standard_Evaluator version within a tolerance of 1e-12 for identical input DataFrames.
5. IF a utility function referenced by ExecutableEvaluator does not yet exist in `standard_evaluator`, THEN THE Migration SHALL migrate that utility into `standard_evaluator` or provide an equivalent implementation; THE ExecutableEvaluator migration SHALL be blocked until all required utilities are successfully migrated or equivalent implementations are provided.
6. WHEN the migrated `ExecutableEvaluator` module is imported, THE module SHALL NOT import any symbol from the `boeing_standard_evaluator` package.

### Requirement 6: Migrate ShiftScaleEvaluator

**User Story:** As a developer, I want the ShiftScaleEvaluator class available in standard_evaluator, so that I can apply shift/scale transformations without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `shift_scale_evaluator.py` into `src/standard_evaluator/evaluators/shift_scale_evaluator.py`.
2. THE copied ShiftScaleEvaluator class SHALL inherit from the migrated Evaluator base class.
3. THE copied file SHALL update all internal import paths from `boeing_standard_evaluator` to `standard_evaluator`, including references to `utilities.shift_scale`, `opt_problem`, `evaluator_info`, `conversions`, and `utilities.problem_dict_utility`.
4. WHEN ShiftScaleEvaluator is instantiated with a valid Evaluator and OptProblem containing shift/scale values, and then called with a DataFrame of input values, THE ShiftScaleEvaluator SHALL produce numerical outputs identical to the Boeing_Standard_Evaluator version within a tolerance of 1e-10.
5. IF the `evaluate` argument passed to ShiftScaleEvaluator is not an instance of Evaluator, THEN THE ShiftScaleEvaluator SHALL raise a TypeError immediately in the constructor.
6. IF none of `opt_problem`, `interface`, or `problem` arguments are provided to the constructor, THEN THE ShiftScaleEvaluator SHALL raise a ValueError immediately in the constructor, preventing instantiation.
7. WHEN the wrapped Evaluator provides a `jacobian` method, THE ShiftScaleEvaluator SHALL expose a `jacobian` attribute that returns shift-and-scale-adjusted Jacobian values consistent with finite-difference approximation within a tolerance of 1e-4.

### Requirement 7: Migrate OpenMDAOEvaluator

**User Story:** As a developer, I want the OpenMDAOEvaluator class available in standard_evaluator, so that I can wrap OpenMDAO components as evaluators without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `open_mdao_evaluator.py` into `src/standard_evaluator/evaluators/open_mdao_evaluator.py`.
2. THE copied OpenMDAOEvaluator class SHALL inherit from the Evaluator base class located in `standard_evaluator.evaluators.abstract_evaluator`.
3. THE copied file SHALL replace all import paths referencing `boeing_standard_evaluator` with the corresponding `standard_evaluator` paths, resulting in zero remaining references to `boeing_standard_evaluator` in the migrated file.
4. WHEN OpenMDAOEvaluator is called with a valid OpenMDAO problem that has been set up and had `run_model()` invoked at least once, THE OpenMDAOEvaluator SHALL produce outputs that match the Boeing_Standard_Evaluator version to within a floating-point tolerance of 1e-10 for each response value.
5. WHEN `OpenMDAOEvaluator` is imported from `standard_evaluator.evaluators`, THE import SHALL resolve without raising `ImportError` or `ModuleNotFoundError`.
6. IF `scan_model` is False and `use_defined_problem` is False, THEN THE OpenMDAOEvaluator SHALL raise a `ValueError` immediately during instantiation (in `__init__`) with a message indicating that both flags cannot be False simultaneously.

### Requirement 8: Migrate ExcelEvaluator

**User Story:** As a developer, I want the ExcelEvaluator class and its utilities available in standard_evaluator, so that I can interface with Excel workbooks without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `excel_evaluator.py` into `src/standard_evaluator/evaluators/excel_evaluator.py`.
2. THE Migration SHALL copy `excel_utilities.py` into `src/standard_evaluator/evaluators/excel_utilities.py`.
3. THE copied files SHALL replace every `boeing_standard_evaluator` import with its `standard_evaluator` equivalent, covering at minimum the imports of `Evaluator`, `OptProblem`, `EvaluatorInfo`, `excel_utilities` submodule references, and `MacroDefinition`.
4. WHEN ExcelEvaluator is instantiated with a valid SpreadsheetModel, THE ExcelEvaluator SHALL instantiate successfully without importing any module from `boeing_standard_evaluator`, and all public names exported by `excel_utilities.py` (SpreadsheetModel, MacroDefinition, VarType, get_explicit_definitions, get_interface_from_excel_named_ranges, create_variable_from_excel_address, range_shape) SHALL be importable from `standard_evaluator.evaluators.excel_utilities`.
5. IF any dependent module required by the copied files is not resolvable within `standard_evaluator`, THEN the migration SHALL fail with an ImportError at import time rather than at runtime.

### Requirement 9: Migrate MatlabEvaluator

**User Story:** As a developer, I want the MatlabEvaluator class available in standard_evaluator as an optional import, so that I can interface with MATLAB when the engine is installed without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `matlab_evaluator.py` into `src/standard_evaluator/evaluators/matlab_evaluator.py`.
2. THE copied file SHALL update all internal import paths from `boeing_standard_evaluator` to `standard_evaluator`.
3. IF the `matlab.engine` package is not installed, THEN THE `evaluators/__init__.py` SHALL skip importing MatlabEvaluator without raising an error and SHALL set a module-level `can_use_matlab` flag to False.
4. IF the `matlab.engine` package is installed, THEN THE `evaluators/__init__.py` SHALL include MatlabEvaluator in the exported classes and SHALL set `can_use_matlab` to True.
5. WHEN the `matlab.engine` package is installed and MatlabEvaluator is instantiated with a valid function path and name, THE MatlabEvaluator SHALL function without importing any module from `boeing_standard_evaluator`.

### Requirement 10: Migrate TestEvaluator Base Class

**User Story:** As a developer, I want the TestEvaluator abstract base class available in standard_evaluator, so that analytic benchmark evaluators can inherit from it without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `test_evaluator.py` into `src/standard_evaluator/evaluators/test_evaluator.py`, creating the `evaluators` subdirectory and its `__init__.py` if they do not already exist.
2. THE copied TestEvaluator class SHALL inherit from the migrated Evaluator base class located within the `standard_evaluator` package.
3. THE copied file SHALL replace all import references from `boeing_standard_evaluator` to `standard_evaluator`, covering at minimum the Evaluator base class, OptProblem, and utility module imports.
4. THE TestEvaluator class SHALL retain its public `known_solution` property returning a `pd.DataFrame`, its abstract method `_create_opt_problem`, and its `__test__ = False` class attribute to prevent pytest discovery.
5. WHEN a subclass of the migrated TestEvaluator implements `_create_opt_problem` and is instantiated, THE system SHALL resolve all imports from `standard_evaluator` without raising ImportError.

### Requirement 11: Migrate Decorators Subfolder

**User Story:** As a developer, I want the cache and site_logger decorators available in standard_evaluator, so that evaluators can use caching and logging without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `decorators/cache.py` from `src/boeing_standard_evaluator/evaluators/decorators/cache.py` into `src/standard_evaluator/evaluators/decorators/cache.py` and update all import paths referencing `boeing_standard_evaluator` to reference `standard_evaluator`.
2. THE Migration SHALL copy `decorators/site_logger.py` from `src/boeing_standard_evaluator/evaluators/decorators/site_logger.py` into `src/standard_evaluator/evaluators/decorators/site_logger.py` and update all import paths referencing `boeing_standard_evaluator` to reference `standard_evaluator`.
3. THE Migration SHALL create a `src/standard_evaluator/evaluators/decorators/__init__.py` file that enables importing `cacher` and `site_logger` from the `standard_evaluator.evaluators.decorators` package.
4. WHEN the `cacher` decorator is imported from `standard_evaluator.evaluators.decorators.cache`, THE decorator SHALL be instantiable and callable without importing any module from `boeing_standard_evaluator`.
5. WHEN the `site_logger` decorator is imported from `standard_evaluator.evaluators.decorators.site_logger`, THE decorator SHALL be instantiable and callable without importing any module from `boeing_standard_evaluator`.
6. IF `cache.py` or `site_logger.py` references utility functions or classes that have not yet been migrated to `standard_evaluator`, THEN THE Migration SHALL document those dependencies as known limitations and allow the migration to complete, deferring full decorator usability until infrastructure dependencies are resolved.

### Requirement 12: Migrate Test Evaluators Subfolder

**User Story:** As a developer, I want the publicly-cited analytic benchmark test evaluators available in standard_evaluator, so that I can use them for testing optimization algorithms without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy the following 38 test evaluator Python files from `src/boeing_standard_evaluator/evaluators/test/` into `src/standard_evaluator/evaluators/test/`: borehole_multi_fi_base.py, borehole_multi_fi_hi.py, borehole_multi_fi_lo.py, c2_dtlz2.py, cantilevered_beam_continuous.py, cantilevered_beam_with_fixed_variables.py, cantilevered_beam.py, constrained_betts.py, cosine_tensor_product.py, das_knee.py, das_truss.py, disconnect.py, exponential_multi_fi_base.py, exponential_multi_fi_hi.py, exponential_multi_fi_lo.py, exponential_tensor_product.py, extended_rosenbrock.py, forrester_multi_fi_base.py, forrester_multi_fi_hi.py, forrester_multi_fi_lo.py, helical_valley.py, hs100.py, hs118.py, hs38.py, hs47.py, hyperbolic_tangent_tensor_product.py, optlib_test.py, powell_singular.py, rosenbrock.py, simple_multi_fi_base.py, simple_multi_fi_hi.py, simple_multi_fi_lo.py, simple_multi_fi_mid.py, sphere.py, tp37.py, trigonometric.py, two_bar_truss.py, wrkbk_prb_1.py. THE Migration SHALL require successful copying of all 38 files to be considered complete.
2. THE Migration SHALL NOT copy costa_five.py, moo_costa_five.py, or vectorized_flight_aerodynamics_performance_model.py.
3. THE copied `evaluators/test/__init__.py` SHALL use `importlib` and `inspect` to auto-discover all Python modules in the directory (excluding `__init__.py`) and export every class found in those modules except the `TestEvaluator` base class, populating `__all__` with the discovered class names.
4. THE copied test evaluator files SHALL replace all `import boeing_standard_evaluator` and `from boeing_standard_evaluator` statements with their `standard_evaluator` equivalents, and SHALL replace any inline references to `boeing_standard_evaluator` in string literals with `standard_evaluator`.
5. WHEN any of the 38 migrated test evaluator classes is instantiated and its `_evaluate` method is called with a valid input DataFrame, THE test evaluator SHALL return results in the `sites` DataFrame without importing or referencing any module from `boeing_standard_evaluator` at runtime.
6. WHEN `from standard_evaluator.evaluators.test import *` is executed, THE import SHALL make all 38 test evaluator classes available by their class names; IF any of the 38 classes cannot be discovered or imported, THEN THE import SHALL raise an `ImportError` rather than silently making only a subset available.

### Requirement 13: Migrate Supporting Utility Dependencies

**User Story:** As a developer, I want any utility functions required by the evaluators to be available in standard_evaluator, so that the evaluators do not depend on boeing_standard_evaluator utility modules.

#### Acceptance Criteria

1. WHEN an evaluator file references utility functions from `boeing_standard_evaluator.utilities`, THE Migration SHALL either copy the required utility functions into Standard_Evaluator or refactor the import to use equivalent existing functionality in Standard_Evaluator.
2. THE Migration SHALL produce a complete list of all utility modules referenced by the migrated evaluators, including at minimum `mapping`, `option_merge`, `problem_dict_utility`, `shift_scale`, `se_arrays`, and `utility`.
3. THE copied utility functions SHALL update all internal import paths from `boeing_standard_evaluator` to `standard_evaluator` only after the utility functions have been copied; zero remaining references to `boeing_standard_evaluator` SHALL exist in the migrated utility code after copy is complete.
4. IF a migrated utility function depends on other utility functions within `boeing_standard_evaluator.utilities`, THEN THE Migration SHALL also migrate those transitive dependencies so that no indirect runtime dependency on Boeing_Standard_Evaluator remains.
5. THE Standard_Evaluator SHALL remain independently functional without any runtime dependency on Boeing_Standard_Evaluator, verified by importing and invoking each migrated evaluator in an environment where Boeing_Standard_Evaluator is not installed.
6. IF a utility function is used by both migrated evaluators and non-migrated code in Boeing_Standard_Evaluator, THEN THE Migration SHALL copy the function to Standard_Evaluator rather than removing it from Boeing_Standard_Evaluator.

### Requirement 14: Migrate Evaluator Tests

**User Story:** As a developer, I want all evaluator tests copied to standard_evaluator, so that I can verify the migrated evaluators function correctly.

#### Acceptance Criteria

1. THE Migration SHALL copy all `.py` test files from `tests/evaluators/` (including the `test/` subdirectory) in Boeing_Standard_Evaluator into `tests/evaluators/` in Standard_Evaluator, preserving the directory structure.
2. THE copied test files SHALL replace all import paths referencing `boeing_standard_evaluator` with `standard_evaluator` across every copied `.py` file.
3. THE Migration SHALL copy the following test fixture files required by the tests: `Range_example.xlsm`, `test_spreadsheet.xlsm`, `test_generic_airplane.aom`, and the entire `Executable Evaluator/` folder with its contents.
4. WHEN the evaluator test suite is executed via `pytest tests/evaluators/` in the Standard_Evaluator project, THE tests SHALL pass with zero failures and zero import errors referencing `boeing_standard_evaluator`. THE Migration SHALL require actual test execution to be considered successful.
5. IF a copied test file contains a reference to `boeing_standard_evaluator` after migration, THEN THE Migration SHALL be considered failed for that file.

### Requirement 15: Update Standard_Evaluator Package Exports

**User Story:** As a developer, I want the migrated evaluator classes accessible from the top-level standard_evaluator package, so that I can import them conveniently.

#### Acceptance Criteria

1. THE `standard_evaluator` package `__init__.py` SHALL import the `evaluators` subpackage such that `standard_evaluator.evaluators` is accessible as a module attribute.
2. WHEN a user executes `from standard_evaluator.evaluators import Evaluator`, THE import SHALL resolve without raising an `ImportError`.
3. WHEN a user executes `from standard_evaluator.evaluators import PyEvaluator`, THE import SHALL resolve without raising an `ImportError`.
4. WHEN a user executes `from standard_evaluator import StandardBase, StandardEval, StandardGroup, EvaluatorInfo, OptProblem`, THE import SHALL resolve without raising an `ImportError`, confirming that existing exports remain intact.
5. THE `standard_evaluator` package `__all__` list SHALL include the string `"evaluators"` so that the subpackage is included in wildcard imports.

### Requirement 16: Independence Constraint

**User Story:** As a developer, I want standard_evaluator to have no runtime dependency on boeing_standard_evaluator, so that the two libraries can be installed and used independently.

#### Acceptance Criteria

1. THE Standard_Evaluator `pyproject.toml` SHALL NOT list `boeing_standard_evaluator` in its `[project.dependencies]`, `[project.optional-dependencies]`, or `[build-system.requires]` sections.
2. WHEN Standard_Evaluator is installed in an environment without Boeing_Standard_Evaluator, THE `import standard_evaluator` statement SHALL complete without raising `ImportError` or `ModuleNotFoundError`. Individual submodules under `src/standard_evaluator/` MAY fail to import independently without causing the top-level package import to fail.
3. THE Python source files under `src/standard_evaluator/` SHALL NOT contain any import statement referencing `boeing_standard_evaluator`.
4. THE Python source files under `src/standard_evaluator/` SHALL NOT contain any text stating "Boeing Proprietary", "Boeing Confidential", "All Rights Reserved by Boeing", or any notice attributing proprietary ownership to Boeing.
5. WHEN Standard_Evaluator is installed in an environment without Boeing_Standard_Evaluator, THE package's public classes and functions SHALL be callable without raising errors caused by missing `boeing_standard_evaluator` references.

### Requirement 17: Update OptProblem with Build Maps and Partials

**User Story:** As a developer, I want the OptProblem class in standard_evaluator updated with build_maps, _setup_partials, and computed property accessors, so that the evaluators can perform finite-difference partial computations without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Standard_Evaluator OptProblem class SHALL include a `build_maps` method that returns a tuple of two DataFrames `(var_map, res_map)` where `var_map` contains columns (`name`, `multi`, `flat`, `fixed`, `jac_row`, `grad_row`) with one row per flattened variable element, and `res_map` contains columns (`name`, `multi`, `flat`, `objective`, `constraint`, `grad_col`, `jac_col`) with one row per flattened response element.
2. THE Standard_Evaluator OptProblem class SHALL include a `_setup_partials` method that reads from `var_map` and `res_map` and stores the following private attributes: `_flat_partials_res_indices` (1-D int array of response indices flagged as objective or constraint), `_num_partials_responses` (int count of selected responses), `_grad_cols` and `_jac_cols` (1-D int arrays with -1 sentinel for non-applicable entries), `_num_objs_total` and `_num_cons_total` (int totals), `_free_flat_var_mask` (1-D bool array), `_free_flat_var_positions` (1-D int array of free element indices), `_num_flat_vars` (int count of free variables), and `_full_to_free_var_index` (1-D int array mapping full index to free index or -1 for fixed).
3. THE Standard_Evaluator OptProblem class SHALL expose the following read-only properties returning the corresponding private attributes: var_map, res_map, flat_partials_res_indices, num_partials_responses, grad_cols, jac_cols, num_objs_total, num_cons_total, free_flat_var_mask, free_flat_var_positions, num_flat_vars, full_to_free_var_index.
4. WHEN the OptProblem model completes Pydantic initialization, THE Standard_Evaluator OptProblem class SHALL invoke `build_maps` followed by `_setup_partials` via a Pydantic `model_validator(mode="after")`, storing the resulting DataFrames in `_var_map` and `_res_map` private attributes before `_setup_partials` executes.
5. IF `build_maps` raises an exception during initialization, THEN THE Standard_Evaluator OptProblem class SHALL raise a `ValueError` with a message indicating the problem name and the underlying error.
6. IF `_setup_partials` raises an exception during initialization, THEN THE Standard_Evaluator OptProblem class SHALL raise a `ValueError` with a message indicating the problem name and the underlying error, separate from any `build_maps` error.
7. THE Standard_Evaluator OptProblem class SHALL mark a flattened variable element as fixed (setting `fixed` to True in `var_map`) when that variable's lower bound equals its upper bound, and SHALL exclude fixed elements from `jac_row`/`grad_row` assignment by setting those cells to None.
8. THE updated OptProblem class SHALL remain backward-compatible such that existing code which instantiates OptProblem without referencing `build_maps`, `_setup_partials`, or the new properties continues to function without modification, and the base class `check_problem` model_validator still executes.
