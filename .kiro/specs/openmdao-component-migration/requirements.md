# Requirements Document

## Introduction

This document defines the requirements for migrating the `EvaluatorOpenMdaoComponent` class from the boeing-standard-evaluator (bse) library into the standard-evaluator (se) library. The component wraps a standard Evaluator so it can be used as an OpenMDAO ExplicitComponent, mapping evaluator variables to OpenMDAO inputs and evaluator responses to OpenMDAO outputs. The migration adapts the component from the legacy dict-based `problem` API to the pydantic `OptProblem` API used in se.

## Glossary

- **Component**: The `EvaluatorOpenMdaoComponent` class, an OpenMDAO `ExplicitComponent` that wraps a standard Evaluator.
- **Evaluator**: An instance of `standard_evaluator.evaluators.abstract_evaluator.Evaluator`, the abstract base class for all evaluators in se.
- **OptProblem**: A pydantic model (`standard_evaluator.problem.OptProblem`) defining the optimization problem with typed `variables` and `responses` lists.
- **Variable**: A pydantic model (`standard_evaluator.problem.FloatVariable` or similar) representing an input or output variable with `.name`, `.bounds`, `.default`, `.shift`, `.scale` attributes.
- **SurrogateModel**: A class in `standard_evaluator.surrogate_models.abstract_model` that extends Evaluator and provides `from_dict()` and `to_dict()` methods for serialization.
- **OpenMDAO_Input**: An OpenMDAO input variable declared via `add_input()` during component setup.
- **OpenMDAO_Output**: An OpenMDAO output variable declared via `add_output()` during component setup.

## Requirements

### Requirement 1: Component Module Location

**User Story:** As a developer, I want the EvaluatorOpenMdaoComponent to live in a `components/` subdirectory within the se package, so that the project structure mirrors the bse layout and keeps component logic separate from core evaluator logic.

#### Acceptance Criteria

1. THE Component SHALL be defined in the module `standard_evaluator.components.evaluator_om_component`.
2. THE `standard_evaluator.components` package SHALL export `EvaluatorOpenMdaoComponent` via its `__init__.py` such that `from standard_evaluator.components import EvaluatorOpenMdaoComponent` resolves without error. IF the component is not defined in the specified module, THEN the import SHALL fail.
3. WHEN `standard_evaluator.components` is imported, THE package SHALL include `EvaluatorOpenMdaoComponent` in its `__all__` list.

### Requirement 2: Constructor with Evaluator Instance

**User Story:** As a developer, I want to construct the component by passing an Evaluator instance directly, so that I can wrap any evaluator for use in OpenMDAO.

#### Acceptance Criteria

1. WHEN an Evaluator instance is provided as the first positional argument, THE Component SHALL store a deep copy of the Evaluator in its `eval` attribute such that subsequent mutations to the original Evaluator do not affect the stored copy.
2. WHEN an Evaluator instance is provided, THE Component SHALL detect any keyword argument that conflicts with an internal Component attribute (e.g., 'eval') and raise a clear error before calling the parent constructor. All non-conflicting keyword arguments SHALL be passed to the OpenMDAO ExplicitComponent constructor without modification.
3. THE Component SHALL accept any concrete subclass of `standard_evaluator.evaluators.abstract_evaluator.Evaluator` as the evaluator argument.
4. IF the first positional argument is provided and is not an instance of `standard_evaluator.evaluators.abstract_evaluator.Evaluator`, THEN THE Component SHALL raise a `TypeError` with a message indicating the expected type.

### Requirement 3: Constructor Fallback via SurrogateModel.from_dict

**User Story:** As a developer, I want the component to reconstruct an evaluator from a serialized dictionary when no evaluator instance is provided, so that models can be deserialized from stored configurations.

#### Acceptance Criteria

1. WHEN the `evaluator` argument is None AND kwargs contain an `evaluator_options` key with a dictionary value, THE Component SHALL create an Evaluator by calling `SurrogateModel.from_dict()` with the value of `evaluator_options` and store a deep copy of the resulting Evaluator.
2. WHEN the `evaluator` argument is None AND kwargs do not contain an `evaluator_options` key, THE Component SHALL raise an `AttributeError` with a message indicating that no evaluator and no evaluator_options were defined.
3. IF the `evaluator` argument is None AND `SurrogateModel.from_dict()` raises an exception due to an invalid `evaluator_options` value (e.g., not a dictionary, missing required keys, or unrecognized model type), THEN THE Component SHALL allow the exception to propagate to the caller without suppression.

### Requirement 4: Initialize Method for Serialization

**User Story:** As a developer, I want the component to store evaluator serialization data in OpenMDAO options when the evaluator supports `to_dict()`, so that the component can be reconstructed from saved OpenMDAO configurations.

#### Acceptance Criteria

1. WHEN the stored Evaluator has a `to_dict` method, THE Component SHALL declare an `evaluator_options` option of type `dict` specifically during the `initialize` lifecycle method (not at any other point in the lifecycle).
2. WHEN the stored Evaluator has a `to_dict` method, THE Component SHALL set the `evaluator_options` option value to the result of calling `to_dict()` on the Evaluator.
3. WHEN the stored Evaluator does not have a `to_dict` method, THE Component SHALL not declare the `evaluator_options` option.

### Requirement 5: Setup Maps Variables to OpenMDAO Inputs

**User Story:** As a developer, I want the component to automatically create OpenMDAO inputs from the evaluator's OptProblem variables, so that the variable definitions flow through without manual configuration.

#### Acceptance Criteria

1. WHEN setup is called, THE Component SHALL add one OpenMDAO input for each Variable in `evaluator.opt_problem.variables`.
2. WHEN setup is called, THE Component SHALL use the Variable `.name` attribute as the OpenMDAO input name.
3. IF the Variable `.default` attribute is not None, THEN THE Component SHALL use the Variable `.default` attribute as the OpenMDAO input default value, even if the default value falls outside the variable's defined bounds.
4. IF the Variable `.default` attribute is None, THEN THE Component SHALL calculate the default from the Variable bounds midpoint before using it as the OpenMDAO input default value.
5. IF the Variable has both a non-None `.default` value and bounds, THEN THE Component SHALL use the explicit `.default` value, ignoring bounds for default calculation.

### Requirement 6: Setup Maps Responses to OpenMDAO Outputs

**User Story:** As a developer, I want the component to automatically create OpenMDAO outputs from the evaluator's OptProblem responses with proper bounds and scaling, so that optimizer integration works correctly.

#### Acceptance Criteria

1. WHEN setup is called, THE Component SHALL call `add_output` once for each Variable in `evaluator.opt_problem.responses`, resulting in one OpenMDAO output per response Variable.
2. WHEN setup is called, THE Component SHALL use the response Variable `.name` attribute as the OpenMDAO output name passed to `add_output`.
3. WHEN a response Variable has bounds where both lower and upper values are finite (not `-inf` or `+inf`), THE Component SHALL set the OpenMDAO output `lower` parameter to the first element of the bounds tuple and the `upper` parameter to the second element of the bounds tuple.
4. IF a response Variable has bounds where either value is non-finite (`-inf` or `+inf`), THEN THE Component SHALL pass `None` for the corresponding `lower` or `upper` parameter of the OpenMDAO output.
5. WHEN a response Variable has shift and scale values, THE Component SHALL always compute `ref0 = -shift` and `ref = (1.0 / scale) + ref0` and pass these as the `ref0` and `ref` parameters of the OpenMDAO output, regardless of whether the values are defaults (shift=0.0, scale=1.0) or not.
6. IF a response Variable has a scale value of 0.0, THEN THE Component SHALL raise a `ValueError` with a message indicating which response has a scale value of 0.0.
7. THE Component SHALL set the OpenMDAO output default value (`val` parameter) to 0.0 for all response outputs.

### Requirement 7: Compute Evaluates via the Evaluator

**User Story:** As a developer, I want the compute method to convert OpenMDAO inputs into a DataFrame, call the evaluator, and extract responses back to OpenMDAO outputs, so that the evaluator executes within the OpenMDAO data flow.

#### Acceptance Criteria

1. WHEN compute is called, THE Component SHALL construct a single-row pandas DataFrame whose columns correspond to the evaluator's input names and whose values are the current OpenMDAO input values for those names.
2. WHEN compute is called, THE Component SHALL invoke the Evaluator as a callable, passing the constructed DataFrame as the argument.
3. WHEN compute is called, THE Component SHALL extract the scalar value at row index 0 for each response column from the evaluated DataFrame and assign it to the corresponding OpenMDAO output.
4. WHEN compute is called, THE Component SHALL first check for discrete inputs and outputs before performing any computation. IF discrete_inputs is not None, THEN THE Component SHALL raise a `ValueError` stating that discrete inputs are not supported.
5. WHEN compute is called, THE Component SHALL first check for discrete inputs and outputs before performing any computation. IF discrete_outputs is not None, THEN THE Component SHALL raise a `ValueError` stating that discrete outputs are not supported.
6. IF the Evaluator raises an exception during invocation, THEN THE Component SHALL propagate the exception to the caller without modification.

### Requirement 8: Integration with Test Evaluators

**User Story:** As a developer, I want to verify the component works end-to-end with se's existing test evaluators, so that the migration is validated against known expected outputs.

#### Acceptance Criteria

1. WHEN the Component wraps an HS100 evaluator from `standard_evaluator.evaluators.test` and `problem.run_model()` is called, THE Component SHALL produce output values that match, within a relative tolerance of 1e-10, the values obtained by invoking the HS100 evaluator directly with the same default input values.
2. WHEN the Component wraps any TestEvaluator from `standard_evaluator.evaluators.test`, THE Component SHALL complete `problem.setup()` and `problem.run_model()` without raising an exception.
3. WHEN the Component wraps any TestEvaluator from `standard_evaluator.evaluators.test` and `problem.run_model()` completes, THE Component SHALL produce output values for each response that match, within a relative tolerance of 1e-10, the values obtained by invoking the same evaluator directly with default input values.
