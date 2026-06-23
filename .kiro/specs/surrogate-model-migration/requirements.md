# Requirements Document

## Introduction

This feature migrates the surrogate model classes from `boeing_standard_evaluator` into `standard_evaluator`. The migration includes the `SurrogateModel` abstract base class, the `PolynomialModel` with its monomial ordering utilities, and all six SMT Toolbox wrapper models (`AbstractSmtModel` base plus RBF, IDW, GENN, LS, RMTS, SOPA`). The evaluator hierarchy (`Evaluator` → `NumpyEvaluator`) has already been successfully migrated in a previous effort (see `evaluator-migration` spec). After this migration, `standard_evaluator` will be able to build, train, serialize, and evaluate surrogate models independently of `boeing_standard_evaluator`.

## Glossary

- **Standard_Evaluator**: The target Python library at `c:\dev\standard-evaluator` with source in `src/standard_evaluator/`.
- **Boeing_Standard_Evaluator**: The source Python library at `c:\dev\boeing-standard-evaluator` with source in `src/boeing_standard_evaluator/`.
- **SurrogateModel**: The abstract base class for all surrogate models (from `surrogate_models/abstract_model.py`), extending `NumpyEvaluator`.
- **PolynomialModel**: A concrete surrogate model implementing polynomial response surface approximation (from `surrogate_models/polynomial_model.py`).
- **PolynomialModelOptions**: The Pydantic `BaseModel` subclass defining options for `PolynomialModel` (degree, coefficient ordering, parameters).
- **CoefficientOrdering**: An `Enum` defining valid monomial ordering methods (`dec_grlex`, `asc_grlex`, `dec_grrevlex`, `asc_grrevlex`).
- **Monomial_Ordering_Utils**: Utility functions for generating monomial orderings (`grlex_ordering_to_deg`, `grrevlex_ordering_to_deg`, `next_grlex`, `next_grrevlex`).
- **AbstractSmtModel**: The abstract base class for all SMT Toolbox surrogate model wrappers (from `surrogate_models/smt_models/abstract_smt_model.py`).
- **AbstractSmtModelOptions**: The Pydantic `BaseModel` subclass defining base options common to all SMT models (print toggles, `use_xlimits`, `data_dir`, parameters).
- **RadialBasisFunctionModel**: SMT wrapper for Radial Basis Function interpolation.
- **InverseDistanceWeightingModel**: SMT wrapper for Inverse Distance Weighting interpolation.
- **GradientEnhancedNeuralNetworksModel**: SMT wrapper for Gradient-Enhanced Neural Networks.
- **LeastSquaresApproximationModel**: SMT wrapper for Least Squares Approximation.
- **RegularizedMinimalEnergyTensorProductBSplines**: SMT wrapper for RMTB/RMTS spline models.
- **SecondOrderPolynomialApproximationModel**: SMT wrapper for Second-Order Polynomial Approximation.
- **NumpyEvaluator**: The already-migrated evaluator base class that operates on NumPy arrays, located at `src/standard_evaluator/evaluators/numpy_evaluator.py`.
- **OptProblem**: The Pydantic model defining optimization problems, located at `src/standard_evaluator/problem.py`.
- **EvaluatorInfo**: The interface definition class in `standard_evaluator`.
- **SMT_Toolbox**: The `smt` Python package (Surrogate Modeling Toolbox), version ≥ 2.10.1.
- **Migration**: The process of duplicating source code from Boeing_Standard_Evaluator into Standard_Evaluator with import path updates.

## Requirements

### Requirement 1: Create Surrogate Models Package Structure

**User Story:** As a developer, I want the surrogate models package structure created in standard_evaluator, so that I can organize migrated surrogate model classes in the same subfolder layout as the source library.

#### Acceptance Criteria

1. THE Migration SHALL create a `surrogate_models/` package directory under `src/standard_evaluator/` with an `__init__.py` file that is syntactically valid Python.
2. THE Migration SHALL create a `surrogate_models/polynomial_model_utils/` package directory under `src/standard_evaluator/` with an `__init__.py` file that is syntactically valid Python.
3. THE Migration SHALL create a `surrogate_models/smt_models/` package directory under `src/standard_evaluator/` with an `__init__.py` file that is syntactically valid Python.
4. WHEN the `surrogate_models/` package is imported, THE `surrogate_models/__init__.py` SHALL export `SurrogateModel`, `PolynomialModel`, `PolynomialModelOptions`, `CoefficientOrdering`, and all six SMT concrete model classes in its `__all__` list.
5. WHEN a Python interpreter imports `standard_evaluator.surrogate_models`, THE import SHALL complete without raising an `ImportError` or `SyntaxError`, and the package SHALL be usable for subsequent class instantiation.
6. WHEN the main `standard_evaluator.surrogate_models` package imports successfully, THEN importing `standard_evaluator.surrogate_models.polynomial_model_utils` and `standard_evaluator.surrogate_models.smt_models` SHALL also complete without raising an `ImportError` or `SyntaxError`, and those subpackages SHALL be usable for subsequent operations.

### Requirement 2: Migrate SurrogateModel Abstract Base Class

**User Story:** As a developer, I want the SurrogateModel abstract base class available in standard_evaluator, so that all concrete surrogate models can inherit from it without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL place the SurrogateModel base class source file at `src/standard_evaluator/surrogate_models/abstract_model.py`.
2. THE copied SurrogateModel class SHALL inherit from the migrated `NumpyEvaluator` class located at `standard_evaluator.evaluators.numpy_evaluator`.
3. THE copied SurrogateModel class SHALL expose the same public interface as the source, including: all public properties (`xlb`, `xub`, `constant_variables`, `nonconstant_variables`, `nind`, `sites`, `sites_as_np`, `sites_input`, `sites_output`, `nsites`), all public methods (`__call__`, `update`, `to_dict`, `from_dict`, `check_consistency_of_sites`, `remove_constants`, `check_input_array`, `get_response_indices`), and the abstract methods (`eval_np`, `_def_update`, `_def_to_dict`, `_def_from_dict`).
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` module paths, including imports of `NumpyEvaluator`, `OptProblem`, `EvaluatorInfo`, and utility functions (`apply_types_from_evaluator_info`, `remove_duplicates`, `get_constant_vars`), and all referenced target modules SHALL exist and be importable in Standard_Evaluator before the migration is considered complete.
5. WHEN a subclass of SurrogateModel implements the abstract methods and is instantiated with a valid DataFrame and problem definition, THE SurrogateModel SHALL execute without importing any module from `boeing_standard_evaluator`.
6. IF a utility function referenced by `abstract_model.py` does not yet exist in `standard_evaluator`, THEN THE Migration SHALL migrate that utility so that the import resolves without error.

### Requirement 3: Migrate Polynomial Model Utilities

**User Story:** As a developer, I want the monomial ordering utility functions available in standard_evaluator, so that the PolynomialModel can generate degree exponents without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `monomial_ordering.py` into `src/standard_evaluator/surrogate_models/polynomial_model_utils/monomial_ordering.py`.
2. THE copied file SHALL contain the functions `next_grlex`, `grlex_ordering_to_deg`, `next_grrevlex`, and `grrevlex_ordering_to_deg` with the same function signatures and return types as the source.
3. WHEN `grlex_ordering_to_deg(nind=3, max_deg=2)` is called, THE function SHALL return a list of monomials in ascending graded lexicographic order identical to the Boeing_Standard_Evaluator version.
4. WHEN `grrevlex_ordering_to_deg(nind=3, max_deg=2)` is called, THE function SHALL return a list of monomials in ascending graded reverse lexicographic order identical to the Boeing_Standard_Evaluator version.
5. THE copied file SHALL contain zero references to `boeing_standard_evaluator`.

### Requirement 4: Migrate PolynomialModel

**User Story:** As a developer, I want the PolynomialModel class available in standard_evaluator, so that I can fit and evaluate polynomial response surface models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `polynomial_model.py` into `src/standard_evaluator/surrogate_models/polynomial_model.py`.
2. THE copied PolynomialModel class SHALL inherit from the migrated SurrogateModel base class located at `standard_evaluator.surrogate_models.abstract_model.SurrogateModel`.
3. THE copied file SHALL include the `CoefficientOrdering` enum, `PolynomialModelParameters` Pydantic model, `PolynomialModelOptions` Pydantic model, the `PolynomialModel` class, and the `convert_df_datatypes_to_list` helper function.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths, including imports of `SurrogateModel`, `grlex_ordering_to_deg`, `grrevlex_ordering_to_deg`, `remove_duplicates`, `OptProblem`, and utility functions.
5. WHEN PolynomialModel is instantiated with a valid DataFrame, an `OptProblem`, and a degree, and coefficient fitting succeeds, THE PolynomialModel SHALL produce predictions numerically identical (within tolerance of 1e-12) to the Boeing_Standard_Evaluator version for the same input data.
6. WHEN PolynomialModel.`jacobian(x)` is called with a valid input array, THE method SHALL return Jacobian values consistent with finite-difference approximation within a tolerance of 1e-6.
7. WHEN `PolynomialModel.to_dict()` is called, THE method SHALL return a dictionary that can be passed to `PolynomialModel.from_dict()` to reconstruct an equivalent model (round-trip serialization).
8. THE `_define_options()` classmethod SHALL return `PolynomialModelOptions` so that the Pydantic options system is preserved.

### Requirement 5: Migrate AbstractSmtModel Base Class

**User Story:** As a developer, I want the AbstractSmtModel class available in standard_evaluator, so that all SMT Toolbox surrogate model wrappers can inherit from it without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `abstract_smt_model.py` into `src/standard_evaluator/surrogate_models/smt_models/abstract_smt_model.py`.
2. THE copied AbstractSmtModel class SHALL inherit from the migrated SurrogateModel base class located at `standard_evaluator.surrogate_models.abstract_model.SurrogateModel`.
3. THE copied file SHALL include `AbstractSmtModelParameters`, `AbstractSmtModelOptions`, the `_options_to_smt_dict` helper function, and the `AbstractSmtModel` class.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. THE AbstractSmtModel SHALL preserve Dask-based parallel training via `_build_single_model` decorated with `@dask.delayed` and `dask.compute` in `_train`.
6. WHEN a subclass of AbstractSmtModel is instantiated with valid site data and an `OptProblem`, THE model SHALL train using Dask parallelism and produce predictions via `eval_np` without importing any module from `boeing_standard_evaluator`. THE migrated code SHALL NOT import `boeing_standard_evaluator` during class definition, instantiation, or evaluation regardless of whether input data is valid or invalid.
7. IF training fails despite valid inputs being provided, THEN the model SHALL raise an explicit exception rather than failing silently.
7. THE `_options_to_smt_dict` function SHALL convert Pydantic options to a dictionary suitable for passing as keyword arguments to SMT model constructors, excluding `parameters`, `use_xlimits`, and `data_dir` fields, and SHALL compute `xlimits` from `OptProblem` variable bounds when `use_xlimits` is True.
8. WHEN `AbstractSmtModel.to_dict()` is called, THE method SHALL serialize the model to a dictionary that can be passed to `AbstractSmtModel.from_dict()` to reconstruct an equivalent model (round-trip serialization).

### Requirement 6: Migrate Radial Basis Function Model

**User Story:** As a developer, I want the RadialBasisFunctionModel class available in standard_evaluator, so that I can use RBF surrogate models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `rbf_model_using_smt.py` into `src/standard_evaluator/surrogate_models/smt_models/rbf_model_using_smt.py`.
2. THE copied RadialBasisFunctionModel class SHALL inherit from the migrated AbstractSmtModel base class.
3. THE copied file SHALL include `RadialBasisFunctionModelOptions` with its fields (`d0`, `poly_degree`, `reg`, `max_print_depth`) and the `validate_poly_degree` field validator.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. WHEN RadialBasisFunctionModel is instantiated with valid site data and an `OptProblem`, THE model SHALL train and produce predictions numerically identical (within tolerance of 1e-10) to the Boeing_Standard_Evaluator version for the same input data and options.
6. WHEN RadialBasisFunctionModel is imported from `standard_evaluator.surrogate_models.smt_models`, THE import SHALL resolve without raising `ImportError`.

### Requirement 7: Migrate Inverse Distance Weighting Model

**User Story:** As a developer, I want the InverseDistanceWeightingModel class available in standard_evaluator, so that I can use IDW surrogate models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `idw_model_using_smt.py` into `src/standard_evaluator/surrogate_models/smt_models/idw_model_using_smt.py`.
2. THE copied InverseDistanceWeightingModel class SHALL inherit from the migrated AbstractSmtModel base class.
3. THE copied file SHALL include `InverseDistanceWeightingModelOptions` with its fields (`p`).
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. WHEN InverseDistanceWeightingModel is instantiated with valid site data and an `OptProblem`, THE model SHALL train and produce predictions numerically identical (within tolerance of 1e-10) to the Boeing_Standard_Evaluator version for the same input data and options.
6. WHEN InverseDistanceWeightingModel is imported from `standard_evaluator.surrogate_models.smt_models`, THE import SHALL resolve without raising `ImportError`.

### Requirement 8: Migrate Gradient-Enhanced Neural Networks Model

**User Story:** As a developer, I want the GradientEnhancedNeuralNetworksModel class available in standard_evaluator, so that I can use GENN surrogate models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `genn_model_using_smt.py` into `src/standard_evaluator/surrogate_models/smt_models/genn_model_using_smt.py`.
2. THE copied GradientEnhancedNeuralNetworksModel class SHALL inherit from the migrated AbstractSmtModel base class.
3. THE copied file SHALL include `GradientEnhancedNeuralNetworksModelOptions` with all its fields preserved from the source.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. WHEN GradientEnhancedNeuralNetworksModel is instantiated with valid site data and an `OptProblem`, THE model SHALL train and produce predictions without importing any module from `boeing_standard_evaluator`.
6. WHEN GradientEnhancedNeuralNetworksModel is imported from `standard_evaluator.surrogate_models.smt_models`, THE import SHALL resolve without raising `ImportError`.

### Requirement 9: Migrate Least Squares Approximation Model

**User Story:** As a developer, I want the LeastSquaresApproximationModel class available in standard_evaluator, so that I can use LS surrogate models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `ls_model_using_smt.py` into `src/standard_evaluator/surrogate_models/smt_models/ls_model_using_smt.py`.
2. THE copied LeastSquaresApproximationModel class SHALL inherit from the migrated AbstractSmtModel base class.
3. THE copied file SHALL include `LeastSquaresApproximationModelOptions` with all its fields preserved from the source.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. WHEN LeastSquaresApproximationModel is instantiated with valid site data and an `OptProblem`, THE model SHALL train and produce predictions without importing any module from `boeing_standard_evaluator`.
6. WHEN LeastSquaresApproximationModel is imported from `standard_evaluator.surrogate_models.smt_models`, THE import SHALL resolve without raising `ImportError`.

### Requirement 10: Migrate RMTS Model

**User Story:** As a developer, I want the RegularizedMinimalEnergyTensorProductBSplines class available in standard_evaluator, so that I can use RMTS surrogate models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `rmts_model_using_smt.py` into `src/standard_evaluator/surrogate_models/smt_models/rmts_model_using_smt.py`.
2. THE copied RegularizedMinimalEnergyTensorProductBSplines class SHALL inherit from the migrated AbstractSmtModel base class.
3. THE copied file SHALL include `RegularizedMinimalEnergyTensorProductBSplinesOptions` with all its fields preserved from the source.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. WHEN RegularizedMinimalEnergyTensorProductBSplines is instantiated with valid site data, an `OptProblem`, and any `use_xlimits` setting, THE model SHALL train and produce predictions without importing any module from `boeing_standard_evaluator`.
6. WHEN RegularizedMinimalEnergyTensorProductBSplines is imported from `standard_evaluator.surrogate_models.smt_models`, THE import SHALL resolve without raising `ImportError`.

### Requirement 11: Migrate Second-Order Polynomial Approximation Model

**User Story:** As a developer, I want the SecondOrderPolynomialApproximationModel class available in standard_evaluator, so that I can use SOPA surrogate models without depending on boeing_standard_evaluator.

#### Acceptance Criteria

1. THE Migration SHALL copy `sopa_model_using_smt.py` into `src/standard_evaluator/surrogate_models/smt_models/sopa_model_using_smt.py`.
2. THE copied SecondOrderPolynomialApproximationModel class SHALL inherit from the migrated AbstractSmtModel base class.
3. THE copied file SHALL include `SecondOrderPolynomialApproximationModelOptions` with all its fields preserved from the source.
4. THE copied file SHALL replace all `boeing_standard_evaluator` import paths with the corresponding `standard_evaluator` paths.
5. WHEN SecondOrderPolynomialApproximationModel is instantiated with valid site data and an `OptProblem`, THE model SHALL train and produce predictions without importing any module from `boeing_standard_evaluator`.
6. WHEN SecondOrderPolynomialApproximationModel is imported from `standard_evaluator.surrogate_models.smt_models`, THE import SHALL resolve without raising `ImportError`.

### Requirement 12: Migrate Supporting Utility Dependencies

**User Story:** As a developer, I want any utility functions required by the surrogate models to be available in standard_evaluator, so that the surrogate models do not depend on boeing_standard_evaluator utility modules.

#### Acceptance Criteria

1. WHEN a surrogate model file references utility functions from `boeing_standard_evaluator.utilities`, THE Migration SHALL either verify the function already exists in Standard_Evaluator or copy the required utility function into Standard_Evaluator.
2. THE Migration SHALL ensure the following utility functions are available in Standard_Evaluator: `apply_types_from_evaluator_info`, `remove_duplicates`, `get_constant_vars`, `legacy_to_opt_problem`, `get_opt_problem_constant_vars`, and `restrict_problem`.
3. THE copied utility functions SHALL update all internal import paths from `boeing_standard_evaluator` to `standard_evaluator` only when utility functions are actually copied; zero remaining references to `boeing_standard_evaluator` SHALL exist in the migrated utility code.
4. IF a migrated utility function depends on other utility functions within `boeing_standard_evaluator.utilities`, THEN THE Migration SHALL also migrate those transitive dependencies so that no indirect runtime dependency on Boeing_Standard_Evaluator remains.
5. THE `conversions` module function `evaluator_info_to_opt_problem` SHALL be available in Standard_Evaluator if referenced by any migrated surrogate model code.

### Requirement 13: Add SMT Dependency to Standard_Evaluator

**User Story:** As a developer, I want the `smt` package added as a dependency of standard_evaluator, so that the SMT-based surrogate models can access the underlying SMT library.

#### Acceptance Criteria

1. THE Standard_Evaluator `pyproject.toml` SHALL list `smt>=2.10.1` in its `[project.optional-dependencies]` under a group named `surrogate` (or equivalent appropriate group name).
2. WHEN Standard_Evaluator is installed with the `surrogate` optional dependency group (e.g., `pip install standard_evaluator[surrogate]`), THE `smt` package SHALL be available for import.
3. IF `smt` is not installed, THEN importing `standard_evaluator.surrogate_models.smt_models` SHALL raise an `ImportError` with a message indicating that the `smt` package is required and referencing the optional dependency group. THE error SHALL only be raised upon actual import of the `smt_models` module, not when importing the parent `standard_evaluator.surrogate_models` package.
4. THE Standard_Evaluator `pyproject.toml` SHALL also include `dask` in its dependencies (or confirm it already exists) since Dask-based parallel training is used by the SMT models.
5. THE Standard_Evaluator `pyproject.toml` SHALL also include `numdifftools` in the `test` optional dependency group if it is required by surrogate model tests.

### Requirement 14: Migrate Surrogate Model Tests

**User Story:** As a developer, I want all relevant surrogate model tests copied to standard_evaluator, so that I can verify the migrated surrogate models function correctly.

#### Acceptance Criteria

1. THE Migration SHALL copy the following test files from `tests/surrogate_models/` in Boeing_Standard_Evaluator into `tests/surrogate_models/` in Standard_Evaluator: `test_abstract_model.py`, `test_polynomial_model.py`.
2. THE Migration SHALL copy the following test files from `tests/surrogate_models/smt_models/` in Boeing_Standard_Evaluator into `tests/surrogate_models/smt_models/` in Standard_Evaluator: `conftest.py`, `test_abstract_smt_model_options.py`, `test_genn_model_using_smt.py`, `test_idw_model_using_smt.py`, `test_ls_model_using_smt.py`, `test_rbf_model_using_smt.py`, `test_rmts_model_using_smt.py`, `test_sopa_model_using_smt.py`, `test_smt_serialization.py`, `test_smt_wrapper_options.py`, `test_smt_wrapper_validation.py`, `test_options_to_smt_dict.py`.
3. THE Migration SHALL copy the `tests/surrogate_models/test_data/` directory and its contents if any test files reference fixture data from that directory.
4. THE copied test files SHALL replace all import paths referencing `boeing_standard_evaluator` with `standard_evaluator` across every copied `.py` file.
5. WHEN the surrogate model test suite is executed via `pytest tests/surrogate_models/` in the Standard_Evaluator project, THE tests SHALL pass with zero failures and zero import errors referencing `boeing_standard_evaluator`.
6. IF a copied test file contains a reference to `boeing_standard_evaluator` after all import path replacements have been applied, THEN THE Migration SHALL be considered failed for that file.

### Requirement 15: Update Standard_Evaluator Package Exports

**User Story:** As a developer, I want the migrated surrogate model classes accessible from the standard_evaluator package, so that I can import them conveniently.

#### Acceptance Criteria

1. THE `standard_evaluator` top-level `__init__.py` SHALL import the `surrogate_models` subpackage such that `standard_evaluator.surrogate_models` is accessible as a module attribute.
2. WHEN a user executes `from standard_evaluator.surrogate_models import SurrogateModel`, THE import SHALL resolve without raising an `ImportError`.
3. WHEN a user executes `from standard_evaluator.surrogate_models import PolynomialModel`, THE import SHALL resolve without raising an `ImportError`.
4. WHEN a user executes `from standard_evaluator.surrogate_models.smt_models import RadialBasisFunctionModel`, THE import SHALL resolve without raising an `ImportError` (assuming `smt` is installed).
5. THE `standard_evaluator` package `__all__` list SHALL include the string `"surrogate_models"` so that the subpackage is included in wildcard imports.

### Requirement 16: Preserve Pydantic Options System

**User Story:** As a developer, I want all surrogate models to use the Pydantic options system consistently, so that model configuration follows a uniform pattern across the library.

#### Acceptance Criteria

1. THE migrated SurrogateModel class SHALL accept an `options` parameter of type `BaseModel` in its constructor and store it as `self._options`.
2. EACH concrete surrogate model class SHALL implement a `_define_options()` classmethod that returns the Pydantic model class defining its options (e.g., `PolynomialModel._define_options()` returns `PolynomialModelOptions`).
3. THE migrated surrogate models SHALL support `lookup_option_value(field_name)` to retrieve individual option values from the stored Pydantic options instance.
4. WHEN a concrete model is instantiated without an explicit `options` argument, THE model SHALL construct a default instance of its options class (as returned by `_define_options()`).
5. THE `AbstractSmtModelOptions` class SHALL use `ConfigDict(arbitrary_types_allowed=True)` to support storing trained SMT model objects in `AbstractSmtModelParameters`.

### Requirement 17: Preserve Serialization Pattern

**User Story:** As a developer, I want the `to_dict()` / `from_dict()` serialization pattern preserved in the migrated models, so that surrogate models can be saved and restored from dictionaries.

#### Acceptance Criteria

1. THE migrated `SurrogateModel.to_dict()` method SHALL return a dictionary with keys `type`, `info`, `problem`, `version`, `name`, and `design explorer version`.
2. THE `version` value in `to_dict()` output SHALL reference `standard_evaluator` package version instead of `boeing_standard_evaluator`.
3. WHEN `SurrogateModel.from_dict(model_info)` is called with a dictionary produced by `to_dict()`, THE method SHALL reconstruct an equivalent model instance by dispatching to the correct subclass based on the `type` field.
4. WHEN `PolynomialModel.from_dict(model_info)` is called with a valid model dictionary, THE reconstructed model SHALL produce predictions numerically identical (within tolerance of 1e-12) to the original model for the same inputs.
5. WHEN `AbstractSmtModel.from_dict(model_info)` is called with a valid model dictionary, THE reconstructed model SHALL retrain from stored site data and produce predictions numerically close (within tolerance of 1e-10) to the original model.
6. IF `model_info` is not a dictionary, THEN `from_dict` SHALL raise a `TypeError`.
7. IF `model_info` is missing required keys, THEN `from_dict` SHALL raise a `KeyError`.

### Requirement 18: Independence Constraint

**User Story:** As a developer, I want standard_evaluator to have no runtime dependency on boeing_standard_evaluator after this migration, so that the two libraries can be installed and used independently.

#### Acceptance Criteria

1. THE Standard_Evaluator `pyproject.toml` SHALL NOT list `boeing_standard_evaluator` in its `[project.dependencies]`, `[project.optional-dependencies]`, or `[build-system.requires]` sections.
2. WHEN Standard_Evaluator is installed in an environment without Boeing_Standard_Evaluator, THE `import standard_evaluator` statement SHALL complete without raising `ImportError` or `ModuleNotFoundError`.
3. THE Python source files under `src/standard_evaluator/surrogate_models/` SHALL NOT contain any import statement referencing `boeing_standard_evaluator`. IF any such import exists, THEN `import standard_evaluator` SHALL fail, indicating the migration is incomplete.
4. THE Python source files under `src/standard_evaluator/surrogate_models/` SHALL NOT contain any text stating "Boeing Proprietary", "Boeing Confidential", "All Rights Reserved by Boeing", or any notice attributing proprietary ownership to Boeing.
5. WHEN Standard_Evaluator is installed with the `surrogate` optional dependency (and `smt` is available), THE surrogate model classes SHALL be instantiable and callable without raising errors caused by missing `boeing_standard_evaluator` references.
