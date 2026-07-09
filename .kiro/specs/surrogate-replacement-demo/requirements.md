# Requirements Document

## Introduction

This feature delivers a comprehensive Jupyter notebook demonstrating the full workflow for replacing a computationally expensive component in an OpenMDAO assembly with a surrogate model, along with two bug fixes in the `om_converter.py` module that currently block the workflow from executing. The demo uses ExecComp-based components (no external dependencies) and showcases the standard_evaluator library's ability to capture assembly interfaces, build surrogates, and reconstruct modified assemblies.

## Glossary

- **Assembly**: A multi-component OpenMDAO Group representing an analysis system composed of interconnected sub-components
- **Surrogate_Model**: A polynomial approximation trained on data from an expensive evaluator, used to replace that evaluator at lower cost
- **Interface_Description**: A `JoinedInfo` Pydantic instance that serializes the structure, inputs, outputs, promotions, and linkages of an OpenMDAO assembly
- **OM_Converter**: The `om_converter.py` module that provides bidirectional conversion between OpenMDAO assemblies and JoinedInfo/JSON representations
- **Evaluator**: An object implementing the standard_evaluator `Evaluator` API that accepts a DataFrame of inputs and produces response columns
- **OpenMDAOEvaluator**: An evaluator subclass that wraps an OpenMDAO Problem so it can be called through the standard evaluator interface
- **EvaluatorOpenMdaoComponent**: An OpenMDAO ExplicitComponent that wraps any standard_evaluator Evaluator for use inside an OpenMDAO assembly
- **Connection_Metadata**: The tuple data stored in OpenMDAO's `_manual_connections` dictionary describing internal connect() calls within a Group
- **Notebook**: The Jupyter notebook file `docs/source/demos/surrogate_replacement_workflow.ipynb`

## Requirements

### Requirement 1: Fix get_linkages IndexError for OpenMDAO 3.43

**User Story:** As a developer, I want `get_linkages()` to correctly handle OpenMDAO 3.43 connection metadata, so that I can capture the interface of groups that use internal `connect()` calls.

#### Acceptance Criteria

1. WHEN OpenMDAO 3.43 returns 2-tuple connection metadata in `_manual_connections`, THE OM_Converter SHALL extract the source name from the first element without accessing a third element
2. WHEN OpenMDAO returns connection metadata with more than 2 elements containing non-None indexing information, THE OM_Converter SHALL print a diagnostic message indicating indexing is used
3. WHEN `get_interface()` is called on a Group with internal `connect()` calls, THE OM_Converter SHALL return a valid GroupInfo instance with correct linkage tuples
4. THE OM_Converter SHALL preserve backward compatibility with connection metadata that contains only a source name string (no indexing)

### Requirement 2: Fix create_problem TypeError for default_shape

**User Story:** As a developer, I want `create_problem()` to correctly reconstruct OpenMDAO assemblies from serialized interface descriptions, so that I can rebuild assemblies that use ExecComp components with shape options.

#### Acceptance Criteria

1. WHEN the serialized openmdao_options contain a `default_shape` value stored as a list, THE OM_Converter SHALL convert the value to a tuple before passing it to OpenMDAO's ExecComp constructor
2. WHEN any option value in the serialized openmdao_options dictionary is a list that OpenMDAO expects as a tuple, THE OM_Converter SHALL convert the list to a tuple
3. WHEN `create_problem()` is called with a JoinedInfo containing ExecComp components with shape options, THE OM_Converter SHALL instantiate the ExecComp without raising a TypeError
4. FOR ALL valid JoinedInfo instances produced by `get_interface()`, calling `create_problem()` on that instance SHALL produce a functioning OpenMDAO Problem (round-trip property)

### Requirement 3: Create Multi-Component Assembly

**User Story:** As a user learning the library, I want the notebook to create a realistic multi-component assembly with an expensive aero group, so that I can understand how component replacement works in practice.

#### Acceptance Criteria

1. THE Notebook SHALL define an assembly containing at least three component groups: an aerodynamics group, a structures group, and a performance group
2. THE Notebook SHALL use OpenMDAO ExecComp components exclusively (no external tool dependencies)
3. THE Notebook SHALL use promotions at the top-level assembly to connect components
4. THE Notebook SHALL include internal `connect()` calls within the aerodynamics sub-group to demonstrate Bug 1 fix
5. THE Notebook SHALL define outputs from non-aero components (weight, stress) to demonstrate that local surrogate replacement preserves those pathways

### Requirement 4: Capture and Extract Assembly Interface

**User Story:** As a user, I want the notebook to demonstrate capturing the full assembly interface and extracting the expensive sub-group as a standalone problem, so that I can learn how to isolate components for surrogate training.

#### Acceptance Criteria

1. WHEN the assembly is set up, THE Notebook SHALL call `se.get_interface()` on the assembly model to obtain a GroupInfo description
2. THE Notebook SHALL display the assembly structure using `se.show_structure()`
3. THE Notebook SHALL extract the aero sub-group from the assembly as a standalone OpenMDAO Problem using `se.create_problem()`
4. THE Notebook SHALL wrap the extracted aero Problem as an OpenMDAOEvaluator with `scan_model=True` and `use_defined_problem=False`

### Requirement 5: Train Local Surrogate Model

**User Story:** As a user, I want the notebook to demonstrate generating training data and building a polynomial surrogate, so that I can learn how to replace expensive components with cheap approximations.

#### Acceptance Criteria

1. THE Notebook SHALL generate training sites using `np.random.uniform` within the variable bounds of the aero evaluator
2. THE Notebook SHALL evaluate training sites using the aero OpenMDAOEvaluator
3. THE Notebook SHALL build a PolynomialModel with degree=2 (fixed, not configurable) from the training data
4. THE Notebook SHALL wrap the surrogate as an OpenMDAO component using EvaluatorOpenMdaoComponent
5. THE Notebook SHALL wrap the surrogate component in an OpenMDAO Problem and then in an OpenMDAOEvaluator

### Requirement 6: Replace Component in Assembly

**User Story:** As a user, I want the notebook to demonstrate replacing the expensive component's interface description with the surrogate's interface description, so that I can learn the full replacement workflow.

#### Acceptance Criteria

1. THE Notebook SHALL capture the surrogate evaluator's interface using `se.get_interface()`
2. THE Notebook SHALL replace the aero component entry in the original assembly's GroupInfo with the surrogate's EvaluatorInfo
3. THE Notebook SHALL instantiate the modified assembly using `se.create_problem()` to demonstrate the round-trip reconstruction
4. THE Notebook SHALL wrap the modified assembly as an OpenMDAOEvaluator

### Requirement 7: Build Global Surrogate

**User Story:** As a user, I want the notebook to demonstrate building a global surrogate that maps the full assembly inputs directly to outputs, so that I can compare local versus global replacement strategies.

#### Acceptance Criteria

1. THE Notebook SHALL wrap the original assembly as an OpenMDAOEvaluator
2. WHEN the global surrogate will be built, THE Notebook SHALL generate training data by evaluating the original assembly evaluator
3. THE Notebook SHALL build a PolynomialModel with degree=2 (fixed, not configurable) mapping full assembly inputs to all assembly outputs only after training data has been successfully generated
4. THE Notebook SHALL present the global surrogate as a third evaluation approach alongside the original and locally-replaced assemblies

### Requirement 8: Compare All Three Evaluators

**User Story:** As a user, I want the notebook to generate test data and compare the original assembly, locally-replaced assembly, and global surrogate, so that I can understand the accuracy tradeoffs of each approach.

#### Acceptance Criteria

1. THE Notebook SHALL generate new test sites using `np.random.uniform` (distinct from training sites)
2. THE Notebook SHALL evaluate all three evaluators (original, local-surrogate-replaced, global surrogate) on the same test sites
3. THE Notebook SHALL compute error metrics using the original assembly as the baseline truth
4. THE Notebook SHALL display an error summary table showing mean absolute error and max absolute error per response per evaluator
5. THE Notebook SHALL display scatter plots of predicted vs actual values for each response
6. THE Notebook SHALL display percent error box plots for each response grouped by evaluator

### Requirement 9: Notebook Import and Execution Conventions

**User Story:** As a developer, I want the notebook to follow project conventions, so that it integrates cleanly with the documentation build and existing codebase.

#### Acceptance Criteria

1. THE Notebook SHALL use `import standard_evaluator as se` as the primary import
2. THE Notebook SHALL use `import openmdao.api as om` for OpenMDAO imports
3. THE Notebook SHALL use `numpy` and `pandas` for data manipulation and `matplotlib` for visualization
4. THE Notebook SHALL be executable end-to-end without errors when the two bug fixes are applied
5. THE Notebook SHALL include markdown cells explaining each workflow step
