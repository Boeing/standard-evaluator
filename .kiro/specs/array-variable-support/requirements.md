# Requirements Document

## Introduction

Add support for `ArrayVariable` inputs and responses in both the OpenMDAO evaluator (`OpenMDAOEvaluator`) and the OpenMDAO component (`EvaluatorOpenMdaoComponent`). Currently these modules only handle scalar (`FloatVariable`) data — array-shaped variables cause a `TypeError`. This feature removes that restriction, enabling the library to wrap and expose OpenMDAO models that use array inputs and outputs.

## Glossary

- **OpenMDAO_Evaluator**: The `OpenMDAOEvaluator` class that wraps an OpenMDAO problem and exposes it through the Standard Evaluator API.
- **OM_Component**: The `EvaluatorOpenMdaoComponent` class that wraps a Standard Evaluator and exposes it as an OpenMDAO `ExplicitComponent`.
- **ArrayVariable**: A Pydantic model representing a NumPy array variable with per-element bounds, shift, and scale.
- **FloatVariable**: A Pydantic model representing a scalar floating-point variable.
- **Rolled_Representation**: A DataFrame format where array variables are stored as single cells containing NumPy arrays.
- **Shape**: A tuple of integers defining the dimensions of an array variable (NumPy shape convention).
- **OptProblem**: The Pydantic model that defines an optimization problem's variables, responses, objectives, and constraints.

## Requirements

### Requirement 1: OpenMDAO Evaluator Produces ArrayVariable from Scanned Models

**User Story:** As a developer, I want the OpenMDAO evaluator to produce `ArrayVariable` instances when scanning a model with array-shaped inputs or outputs, so that array data flows through the system without loss of structure.

#### Acceptance Criteria

1. WHEN the OpenMDAO_Evaluator scans a model and encounters an input with shape other than `(1,)`, THE OpenMDAO_Evaluator SHALL create an `ArrayVariable` with the detected shape and bounds set to `(-inf, +inf)` instead of raising a `TypeError`.
2. WHEN the OpenMDAO_Evaluator scans a model and encounters an output with shape other than `(1,)`, THE OpenMDAO_Evaluator SHALL create an `ArrayVariable` with the detected shape and bounds set to `(-inf, +inf)`.
3. WHEN an array-shaped input has OpenMDAO scaling metadata (ref, ref0, adder, scaler), THE OpenMDAO_Evaluator SHALL compute shift and scale using `om_utils.determine_adder_scaler(ref0, ref, adder, scaler)` and store the resulting adder in the `ArrayVariable` shift field and the resulting scaler in the scale field. Scaling metadata SHALL only be processed for array-shaped inputs (shape other than `(1,)`).
4. WHEN an array input or output has a units metadata field that is not None, THE OpenMDAO_Evaluator SHALL store the units string in the `ArrayVariable` units field.
5. WHEN a scalar input or output (shape `(1,)`) has a units metadata field that is not None, THE OpenMDAO_Evaluator SHALL store the units string in the `FloatVariable` units field.
6. WHEN an array-shaped input (shape other than `(1,)`) has a `val` entry in its metadata, THE OpenMDAO_Evaluator SHALL store the full NumPy array as the `ArrayVariable` default.
7. IF an array input does not have a `val` entry in its metadata, THEN THE OpenMDAO_Evaluator SHALL set the `ArrayVariable` default to a NumPy zeros array matching the detected shape.
8. WHEN an array input lacks scaling metadata (ref, ref0, adder, and scaler are all None or absent), THE OpenMDAO_Evaluator SHALL set the `ArrayVariable` shift to 0.0 and scale to 1.0.

### Requirement 2: OpenMDAO Evaluator Maps Array Design Variables and Responses

**User Story:** As a developer, I want `_map_elements` to produce `ArrayVariable` instances for multi-element design variables and responses, so that the optimization problem correctly reflects the array structure defined in OpenMDAO.

#### Acceptance Criteria

1. WHEN the OpenMDAO_Evaluator converts design variables from `get_design_vars()` and the `size` metadata of a variable is greater than 1, THE OpenMDAO_Evaluator SHALL create an `ArrayVariable` with `shape` set to the variable's metadata `shape` tuple, `bounds` set to the metadata `lower` and `upper` values, `scale` set to the metadata `scaler` value, and `shift` set to the metadata `adder` value.
2. WHEN the OpenMDAO_Evaluator converts responses from `get_responses()` and the `size` metadata of a response is greater than 1, THE OpenMDAO_Evaluator SHALL create an `ArrayVariable` with `shape` set to the response's metadata `shape` tuple, and `bounds` set to the metadata `lower` and `upper` values.
3. THE OpenMDAO_Evaluator SHALL continue to produce `FloatVariable` instances for variables and responses with `size` metadata equal to 1.
4. IF the metadata for a design variable or response is missing the `lower` bound, THEN THE OpenMDAO_Evaluator SHALL use negative infinity as the default lower bound. IF the metadata is missing the `upper` bound, THEN THE OpenMDAO_Evaluator SHALL use positive infinity as the default upper bound. Each missing bound SHALL be defaulted independently.
5. WHEN the metadata for a design variable or response contains a `units` field that is not None, THE OpenMDAO_Evaluator SHALL store the units string in the variable's `units` field for both `FloatVariable` and `ArrayVariable` instances.

### Requirement 3: OpenMDAO Evaluator Evaluates Array Variables

**User Story:** As a developer, I want the OpenMDAO evaluator to pass array data to and from the wrapped OpenMDAO model during evaluation, so that array variables are evaluated correctly.

#### Acceptance Criteria

1. WHEN the OpenMDAO_Evaluator evaluates a site containing an `ArrayVariable` input, THE OpenMDAO_Evaluator SHALL call `set_val` with the full NumPy array value from the DataFrame cell, preserving the array's shape as defined in the `ArrayVariable`.
2. WHEN the OpenMDAO_Evaluator evaluates a site containing a `FloatVariable` input, THE OpenMDAO_Evaluator SHALL call `set_val` with the scalar float value from the DataFrame cell.
3. WHEN the OpenMDAO_Evaluator extracts a response that is an `ArrayVariable`, THE OpenMDAO_Evaluator SHALL call `get_val` and store the returned NumPy array in the DataFrame cell without flattening or reshaping it.
4. WHEN the OpenMDAO_Evaluator extracts a response that is a `FloatVariable`, THE OpenMDAO_Evaluator SHALL call `get_val` and store the returned scalar float value in the DataFrame cell.
5. THE OpenMDAO_Evaluator SHALL evaluate all sites in the DataFrame by iterating over each row, setting all inputs (scalar and array), running the model, and extracting all outputs, producing values that match the wrapped OpenMDAO model's outputs for the given inputs.

### Requirement 4: OM Component Registers Array Inputs

**User Story:** As a developer, I want the OpenMDAO component to register array variables as shaped OpenMDAO inputs, so that the component integrates correctly into OpenMDAO models with array data.

#### Acceptance Criteria

1. WHEN the OM_Component sets up an input that is an `ArrayVariable` with a non-None `default`, THE OM_Component SHALL call `add_input` with `shape` set to the variable's `shape` tuple and `val` set to the variable's `default` NumPy array.
2. WHEN the OM_Component sets up an input that is an `ArrayVariable` with `units` set to a non-None string, THE OM_Component SHALL pass the `units` string to `add_input`.
3. IF the OM_Component sets up an input that is an `ArrayVariable` with `units` set to None, THEN THE OM_Component SHALL call `add_input` without the `units` keyword argument.
4. THE OM_Component SHALL continue to register `FloatVariable` inputs as scalar inputs with no `shape` argument.
5. WHEN the OM_Component sets up an input that is a `FloatVariable` with `units` set to a non-None string, THE OM_Component SHALL pass the `units` string to `add_input`.
6. IF the OM_Component sets up an input that is a `FloatVariable` with `units` set to None, THEN THE OM_Component SHALL call `add_input` without the `units` keyword argument.
7. WHEN the OM_Component sets up an input that is an `ArrayVariable` with `default` set to None, THE OM_Component SHALL compute the default value as the element-wise midpoint of the variable's lower and upper bounds and pass the resulting NumPy array as `val` to `add_input`.

### Requirement 5: OM Component Registers Array Outputs

**User Story:** As a developer, I want the OpenMDAO component to register array responses as shaped OpenMDAO outputs, so that downstream OpenMDAO components receive array data with correct metadata.

#### Acceptance Criteria

1. WHEN the OM_Component sets up an output that is an `ArrayVariable`, THE OM_Component SHALL call `add_output` with `shape` set to the response's shape and `val` set to a NumPy zeros array of that shape.
2. WHEN the `ArrayVariable` output has units defined (not None), THE OM_Component SHALL pass the units string to `add_output`.
3. WHEN a `FloatVariable` output has `units` set to a non-None string, THE OM_Component SHALL pass the `units` string to `add_output`.
4. IF a `FloatVariable` output has `units` set to None, THEN THE OM_Component SHALL call `add_output` without the `units` keyword argument.
5. WHEN the `ArrayVariable` output has a lower bound that is not all negative infinity and all elements share the same lower bound value, THE OM_Component SHALL pass that scalar value as `lower`. WHEN elements have different lower bound values, THE OM_Component SHALL pass the full array as `lower`. IF the lower bound is all negative infinity, THEN THE OM_Component SHALL explicitly not pass the `lower` keyword argument to `add_output`.
6. WHEN the `ArrayVariable` output has an upper bound that is not all positive infinity and all elements share the same upper bound value, THE OM_Component SHALL pass that scalar value as `upper`. WHEN elements have different upper bound values, THE OM_Component SHALL pass the full array as `upper`. IF the upper bound is all positive infinity, THEN THE OM_Component SHALL explicitly not pass the `upper` keyword argument to `add_output`.
7. WHEN the `ArrayVariable` output has non-default scaling (shift is not all zeros or scale is not all ones), THE OM_Component SHALL compute `ref0` from the shift array and `ref` from `shift + scale` and pass them to `add_output`. IF shift is all zeros and scale is all ones, THEN THE OM_Component SHALL not pass `ref0` or `ref` to `add_output`.
8. IF the `ArrayVariable` output has any scale element equal to zero, THEN THE OM_Component SHALL raise a `ValueError` during setup indicating that zero scale causes division by zero in OpenMDAO normalization.
9. THE OM_Component SHALL continue to register `FloatVariable` responses as scalar outputs with no shape argument.

### Requirement 6: OM Component Compute Handles Array Data

**User Story:** As a developer, I want the component's `compute` method to marshal array data between OpenMDAO and the evaluator, so that evaluation works end-to-end with array variables.

#### Acceptance Criteria

1. WHEN the OM_Component computes with `ArrayVariable` inputs, THE OM_Component SHALL place the NumPy array from `inputs[name]` into the DataFrame cell for that variable, preserving the original shape.
2. WHEN the OM_Component computes and the evaluator produces `ArrayVariable` responses, THE OM_Component SHALL extract the NumPy array from the DataFrame cell and assign it to `outputs[name]`, preserving the shape declared in the `ArrayVariable`.
3. WHEN the OM_Component computes with `FloatVariable` inputs, THE OM_Component SHALL place the scalar float value from `inputs[name]` into the DataFrame cell for that variable.
4. FOR ALL OptProblems containing a mix of `FloatVariable` and `ArrayVariable` variables and responses, THE OM_Component SHALL marshal each variable according to its type (scalar for `FloatVariable`, full NumPy array for `ArrayVariable`) in both input and output directions during a single `compute` call.

### Requirement 7: Backward Compatibility

**User Story:** As a developer, I want existing scalar-only evaluators and components to continue working unchanged, so that this feature does not break existing integrations.

#### Acceptance Criteria

1. THE OpenMDAO_Evaluator SHALL produce an OptProblem with identical variable types, bounds, defaults, and scaling, and SHALL produce identical evaluated DataFrame output values, for models that contain only scalar `(1,)` variables and responses as it did before this feature.
2. THE OM_Component SHALL register identical OpenMDAO inputs and outputs (names, shapes, units, bounds, scaling) and SHALL produce identical computed output values for evaluators whose OptProblem contains only `FloatVariable` instances as it did before this feature.
3. WHEN the OpenMDAO_Evaluator scans a model containing a mix of scalar and array variables, THE OpenMDAO_Evaluator SHALL produce `FloatVariable` for variables and responses with shape `(1,)` and `ArrayVariable` for variables and responses with shape other than `(1,)`.
4. THE OpenMDAO_Evaluator and OM_Component SHALL not require any new mandatory constructor parameters or method arguments for scalar-only usage, and SHALL not remove or rename any existing public methods or attributes.
