# Implementation Plan: Surrogate Model Migration

## Overview

This plan migrates surrogate model classes from `boeing_standard_evaluator` into `standard_evaluator` using a copy-and-repoint strategy. Each task copies source files, updates import paths from `boeing_standard_evaluator` to `standard_evaluator`, and verifies the result compiles and passes tests. Tasks are ordered so that foundational components (package structure, utilities, base class) are implemented first, followed by concrete models, tests, and final integration.

## Tasks

- [x] 1. Set up package structure and dependency configuration
  - [x] 1.1 Create surrogate_models package directories and __init__.py files
    - Create `src/standard_evaluator/surrogate_models/__init__.py`
    - Create `src/standard_evaluator/surrogate_models/polynomial_model_utils/__init__.py`
    - Create `src/standard_evaluator/surrogate_models/smt_models/__init__.py`
    - The `surrogate_models/__init__.py` should initially import non-SMT classes and conditionally import SMT classes
    - The `smt_models/__init__.py` should wrap imports in try/except to raise a helpful `ImportError` when `smt` is not installed
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 13.3_

  - [x] 1.2 Update pyproject.toml with optional dependencies
    - Add `smt>=2.10.1` to a `surrogate` optional dependency group in `[project.optional-dependencies]`
    - Confirm `dask` is in dependencies (or add it)
    - Add `numdifftools` to the `test` optional dependency group if not already present
    - _Requirements: 13.1, 13.2, 13.4, 13.5_

  - [x] 1.3 Update standard_evaluator top-level __init__.py
    - Add `surrogate_models` to the top-level package exports and `__all__`
    - _Requirements: 15.1, 15.5_

- [x] 2. Migrate supporting utilities
  - [x] 2.1 Create opt_problem_utility.py with get_opt_problem_constant_vars
    - Create `src/standard_evaluator/utilities/opt_problem_utility.py`
    - Copy the `get_opt_problem_constant_vars` function from `boeing_standard_evaluator`
    - Replace all `boeing_standard_evaluator` import paths with `standard_evaluator`
    - _Requirements: 12.1, 12.2, 12.3_

  - [x] 2.2 Verify and migrate any missing utility functions
    - Confirm `apply_types_from_evaluator_info`, `remove_duplicates`, `get_constant_vars`, `legacy_to_opt_problem`, `restrict_problem` exist in `standard_evaluator.utilities`
    - If any are missing, copy them from `boeing_standard_evaluator` with import path updates
    - Ensure `evaluator_info_to_opt_problem` is available in `standard_evaluator.converters`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 3. Migrate SurrogateModel abstract base class
  - [x] 3.1 Copy and adapt abstract_model.py
    - Copy `surrogate_models/abstract_model.py` from `boeing_standard_evaluator` to `src/standard_evaluator/surrogate_models/abstract_model.py`
    - Replace all `boeing_standard_evaluator` imports with `standard_evaluator` equivalents (NumpyEvaluator, OptProblem, EvaluatorInfo, utilities, converters)
    - Replace `version("boeing_standard_evaluator")` with `version("standard_evaluator")`
    - Remove any Boeing proprietary notices
    - Verify the class inherits from `standard_evaluator.evaluators.NumpyEvaluator`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 16.1, 16.3, 16.4, 17.1, 17.2, 17.3, 17.6, 17.7, 18.3, 18.4_

- [x] 4. Migrate PolynomialModel and utilities
  - [x] 4.1 Copy and adapt monomial_ordering.py
    - Copy `surrogate_models/polynomial_model_utils/monomial_ordering.py` to `src/standard_evaluator/surrogate_models/polynomial_model_utils/monomial_ordering.py`
    - Verify zero references to `boeing_standard_evaluator`
    - Remove any Boeing proprietary notices
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 4.2 Write property test for monomial ordering correctness
    - **Property 1: Monomial ordering correctness**
    - Test that `grlex_ordering_to_deg` and `grrevlex_ordering_to_deg` produce lists that begin with the zero monomial, contain the correct count (binomial coefficient), have non-decreasing total degree, and contain no duplicates
    - Use Hypothesis strategies for `nind` (1-5) and `max_deg` (0-4)
    - **Validates: Requirements 3.3, 3.4**

  - [x] 4.3 Copy and adapt polynomial_model.py
    - Copy `surrogate_models/polynomial_model.py` to `src/standard_evaluator/surrogate_models/polynomial_model.py`
    - Replace all `boeing_standard_evaluator` imports: `SurrogateModel`, `grlex_ordering_to_deg`, `grrevlex_ordering_to_deg`, `remove_duplicates`, `OptProblem`, `get_opt_problem_constant_vars`
    - Remove any Boeing proprietary notices
    - Ensure `CoefficientOrdering`, `PolynomialModelParameters`, `PolynomialModelOptions`, `PolynomialModel`, and `convert_df_datatypes_to_list` are all present
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.8, 16.2_

  - [x] 4.4 Write property test for polynomial fit accuracy
    - **Property 2: Polynomial model least-squares fit accuracy**
    - Test that fitting a PolynomialModel and evaluating at training sites reproduces responses within tolerance 1e-10
    - Use Hypothesis to generate random site DataFrames with valid bounds and sufficient points
    - **Validates: Requirements 4.5**

  - [x] 4.5 Write property test for Jacobian consistency
    - **Property 3: Polynomial Jacobian consistency with finite differences**
    - Test that `jacobian(x)` agrees with central finite-difference approximation within tolerance 1e-6
    - Use `numdifftools` for finite-difference computation
    - **Validates: Requirements 4.6**

  - [x] 4.6 Write property test for PolynomialModel serialization round-trip
    - **Property 4: PolynomialModel serialization round-trip**
    - Test that `to_dict()` followed by `from_dict()` produces a model with numerically identical predictions (tolerance 1e-12)
    - **Validates: Requirements 4.7, 17.3, 17.4**

- [x] 5. Checkpoint - Ensure polynomial model tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Migrate AbstractSmtModel base class
  - [x] 6.1 Copy and adapt abstract_smt_model.py
    - Copy `surrogate_models/smt_models/abstract_smt_model.py` to `src/standard_evaluator/surrogate_models/smt_models/abstract_smt_model.py`
    - Replace all `boeing_standard_evaluator` imports with `standard_evaluator` equivalents
    - Remove any Boeing proprietary notices
    - Ensure `AbstractSmtModelParameters`, `AbstractSmtModelOptions`, `_options_to_smt_dict`, and `AbstractSmtModel` are present
    - Verify Dask parallel training (`@dask.delayed`, `dask.compute`) is preserved
    - Verify `ConfigDict(arbitrary_types_allowed=True)` is on AbstractSmtModelOptions
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.8, 16.2, 16.5_

  - [x] 6.2 Write property test for _options_to_smt_dict conversion
    - **Property 5: _options_to_smt_dict conversion correctness**
    - Test that the output dict does NOT contain `parameters`, `use_xlimits`, or `data_dir` keys
    - Test that `xlimits` is computed from OptProblem bounds when `use_xlimits` is True
    - Test that all other option fields are present with correct values
    - **Validates: Requirements 5.7**

- [x] 7. Migrate concrete SMT model wrappers
  - [x] 7.1 Copy and adapt rbf_model_using_smt.py
    - Copy to `src/standard_evaluator/surrogate_models/smt_models/rbf_model_using_smt.py`
    - Replace all `boeing_standard_evaluator` imports
    - Remove any Boeing proprietary notices
    - Ensure `RadialBasisFunctionModelOptions` with `d0`, `poly_degree`, `reg`, `max_print_depth` fields and `validate_poly_degree` validator are present
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

  - [x] 7.2 Copy and adapt idw_model_using_smt.py
    - Copy to `src/standard_evaluator/surrogate_models/smt_models/idw_model_using_smt.py`
    - Replace all `boeing_standard_evaluator` imports
    - Remove any Boeing proprietary notices
    - Ensure `InverseDistanceWeightingModelOptions` with `p` field is present
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

  - [x] 7.3 Copy and adapt genn_model_using_smt.py
    - Copy to `src/standard_evaluator/surrogate_models/smt_models/genn_model_using_smt.py`
    - Replace all `boeing_standard_evaluator` imports
    - Remove any Boeing proprietary notices
    - Ensure `GradientEnhancedNeuralNetworksModelOptions` with all fields is present
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.6_

  - [x] 7.4 Copy and adapt ls_model_using_smt.py
    - Copy to `src/standard_evaluator/surrogate_models/smt_models/ls_model_using_smt.py`
    - Replace all `boeing_standard_evaluator` imports
    - Remove any Boeing proprietary notices
    - Ensure `LeastSquaresApproximationModelOptions` is present
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_

  - [x] 7.5 Copy and adapt rmts_model_using_smt.py
    - Copy to `src/standard_evaluator/surrogate_models/smt_models/rmts_model_using_smt.py`
    - Replace all `boeing_standard_evaluator` imports
    - Remove any Boeing proprietary notices
    - Ensure `RegularizedMinimalEnergyTensorProductBSplinesOptions` with all fields (including `use_xlimits=True` default) is present
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.6_

  - [x] 7.6 Copy and adapt sopa_model_using_smt.py
    - Copy to `src/standard_evaluator/surrogate_models/smt_models/sopa_model_using_smt.py`
    - Replace all `boeing_standard_evaluator` imports
    - Remove any Boeing proprietary notices
    - Ensure `SecondOrderPolynomialApproximationModelOptions` is present
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.6_

  - [x] 7.7 Write property test for SMT model serialization round-trip
    - **Property 6: SMT model serialization round-trip**
    - Test that `to_dict()` followed by `from_dict()` produces predictions within tolerance 1e-10 for RBF, IDW, LS, RMTS, and SOPA models
    - **Validates: Requirements 5.8, 17.5**

  - [x] 7.8 Write property test for SMT model training produces valid predictions
    - **Property 7: SMT model training produces valid predictions**
    - Test that instantiating each concrete SMT model with valid training data completes without error and produces finite-valued predictions (no NaN or Inf)
    - Use Hypothesis to generate valid training DataFrames
    - **Validates: Requirements 6.5, 7.5, 9.5, 10.5, 11.5**

- [x] 8. Checkpoint - Ensure SMT model imports and basic instantiation work
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Migrate test suite
  - [x] 9.1 Copy and adapt test_abstract_model.py and test_polynomial_model.py
    - Copy `tests/surrogate_models/test_abstract_model.py` from `boeing_standard_evaluator` to `tests/surrogate_models/test_abstract_model.py` in `standard_evaluator`
    - Copy `tests/surrogate_models/test_polynomial_model.py` similarly
    - Replace all `boeing_standard_evaluator` imports with `standard_evaluator`
    - Copy `tests/surrogate_models/test_data/` directory and contents
    - _Requirements: 14.1, 14.3, 14.4, 14.5, 14.6_

  - [x] 9.2 Copy and adapt SMT model test files
    - Copy `tests/surrogate_models/smt_models/conftest.py` with import path updates
    - Copy `test_abstract_smt_model_options.py`, `test_genn_model_using_smt.py`, `test_idw_model_using_smt.py`, `test_ls_model_using_smt.py`, `test_rbf_model_using_smt.py`, `test_rmts_model_using_smt.py`, `test_sopa_model_using_smt.py` with import path updates
    - Copy `test_smt_serialization.py`, `test_smt_wrapper_options.py`, `test_smt_wrapper_validation.py`, `test_options_to_smt_dict.py` with import path updates
    - Remove any Boeing proprietary notices from all test files
    - _Requirements: 14.2, 14.4, 14.5, 14.6_

- [x] 10. Final integration and validation
  - [x] 10.1 Wire surrogate_models/__init__.py exports
    - Finalize the `__all__` list in `surrogate_models/__init__.py` to export `SurrogateModel`, `PolynomialModel`, `PolynomialModelOptions`, `CoefficientOrdering`, and all six SMT concrete model classes
    - Ensure the lazy/conditional import pattern for SMT classes works correctly
    - Verify `from standard_evaluator.surrogate_models import SurrogateModel` works
    - Verify `from standard_evaluator.surrogate_models import PolynomialModel` works
    - Verify `from standard_evaluator.surrogate_models.smt_models import RadialBasisFunctionModel` works (with smt installed)
    - _Requirements: 1.4, 15.2, 15.3, 15.4_

  - [x] 10.2 Verify independence constraint
    - Run a grep across all files in `src/standard_evaluator/surrogate_models/` confirming zero references to `boeing_standard_evaluator`
    - Run a grep confirming zero Boeing proprietary notices
    - Confirm `pyproject.toml` does not list `boeing_standard_evaluator` in any dependency section
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The copy-and-repoint strategy means minimal algorithmic changes — the main work is import path replacement and verifying nothing was missed
- SMT models require the `smt` package to be installed; tests for SMT models should be marked with `@pytest.mark.skipif` or run only when the `surrogate` extras are installed

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["3.1"] },
    { "id": 3, "tasks": ["4.1", "4.3"] },
    { "id": 4, "tasks": ["4.2", "4.4", "4.5", "4.6"] },
    { "id": 5, "tasks": ["6.1"] },
    { "id": 6, "tasks": ["6.2", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6"] },
    { "id": 7, "tasks": ["7.7", "7.8"] },
    { "id": 8, "tasks": ["9.1", "9.2"] },
    { "id": 9, "tasks": ["10.1", "10.2"] }
  ]
}
```
