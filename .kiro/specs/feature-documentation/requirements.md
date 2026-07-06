# Requirements Document

## Introduction

The Standard Evaluator library has added several major features (evaluator hierarchy, surrogate models, OpenMDAO component, array variables, benchmark test problems, simplified interface) that currently lack documentation. This feature delivers comprehensive tutorial-style Jupyter notebook documentation for all new capabilities, following the existing MyST-NB Sphinx documentation pattern established in the `docs/source/demos/` folder.

## Glossary

- **Documentation_System**: The Sphinx-based documentation build using MyST-NB that renders Jupyter notebooks as HTML pages in the project documentation site.
- **Demo_Notebook**: A self-contained Jupyter notebook (`.ipynb`) located in `docs/source/demos/` that serves as both executable tutorial and rendered documentation page.
- **Evaluator**: An abstract class defining the common API for wrapping analysis codes; accepts a pandas DataFrame of inputs and adds response columns in-place.
- **NumpyEvaluator**: A concrete Evaluator subclass that wraps a user-defined NumPy function.
- **Surrogate_Model**: A trained mathematical approximation of an Evaluator, supporting training from data, prediction, serialization, and Dask-parallel training.
- **EvaluatorOpenMdaoComponent**: An OpenMDAO ExplicitComponent that wraps any Evaluator, mapping variables to inputs and responses to outputs.
- **ArrayVariable**: A variable type representing multi-dimensional NumPy arrays with shape, bounds, shift, and scale.
- **Benchmark_Evaluator**: A test evaluator implementing a known mathematical optimization problem, used for testing and demonstration.
- **Demos_Index**: The Sphinx toctree file (`docs/source/demos/index.rst`) that lists all demo notebooks for inclusion in the documentation build.
- **SE**: The standard import alias for `standard_evaluator` (`import standard_evaluator as se`).

## Requirements

### Requirement 1: Evaluator Hierarchy Documentation

**User Story:** As a developer new to the Standard Evaluator library, I want a tutorial notebook explaining the evaluator hierarchy and how to create and use evaluators, so that I can understand the core abstraction and start wrapping my own analysis codes.

#### Acceptance Criteria

1. WHEN a user opens the evaluator hierarchy notebook, THE Demo_Notebook SHALL contain a markdown overview explaining the abstract Evaluator class, its subclasses (NumpyEvaluator, ExecutableEvaluator, ShiftScaleEvaluator, OpenMdaoEvaluator, ExcelEvaluator, MatlabEvaluator), and the decorator pattern (caching, logging).
2. THE Demo_Notebook SHALL include a code example that creates a NumpyEvaluator from one of the benchmark test evaluators, builds an input DataFrame with at least 2 rows of input sites, calls the evaluator using the `__call__` method, and displays the resulting DataFrame showing both the original input columns and the appended response columns.
3. THE Demo_Notebook SHALL include a working code example demonstrating the `eval_np` and `eval_list` developer convenience methods, with a markdown explanation that these methods exclude fixed variables from the input and unroll ArrayVariables into individual scalar columns (only the `__call__` method preserves array variables as arrays).
4. THE Demo_Notebook SHALL include a working code example that applies the caching decorator to an evaluator by passing a `cache` parameter, evaluates at least one site, and then calls `get_cache()` to display the cached results.
5. THE Demo_Notebook SHALL include a working code example that applies the logging decorator to an evaluator by passing `logging=True`, evaluates at least one site, and then calls `get_log()` to display the logged evaluation history.
6. THE Demo_Notebook SHALL use the `import standard_evaluator as se` convention for all imports.
7. THE Demo_Notebook SHALL be executable without errors in the project virtual environment using only the base required dependencies (no optional extras such as smt, aviary, or excel).

### Requirement 2: Surrogate Model Documentation

**User Story:** As a developer who wants to build surrogate models, I want a tutorial notebook explaining how to train, evaluate, and serialize surrogate models, so that I can approximate expensive evaluators with fast mathematical models.

#### Acceptance Criteria

1. WHEN a user opens the surrogate model notebook, THE Demo_Notebook SHALL contain a markdown overview that lists the SurrogateModel base class, PolynomialModel, and the SMT-based models (RBF, IDW, GENN, LS, RMTS, SOPA), stating each model's purpose and their inheritance hierarchy.
2. THE Demo_Notebook SHALL include a working code example that trains a PolynomialModel on at least 10 sample sites generated from a test evaluator and predicts responses at a separate set of at least 3 new input sites, printing or displaying both predicted and true values.
3. THE Demo_Notebook SHALL include a working code example that serializes a trained model using `to_dict`, reconstructs it using `from_dict`, and verifies that predictions from the reconstructed model match the original model's predictions on the same input sites.
4. THE Demo_Notebook SHALL include a working code example that trains an SMT-based model (RBF) on at least 10 sample sites and displays a numerical accuracy metric (such as RMSE or R²) comparing the model's predictions to the true evaluator output on a separate set of test sites.
5. THE Demo_Notebook SHALL include a markdown section explaining the options system for surrogate models, accompanied by a code example that creates a model with a non-default option value (e.g., polynomial degree or RBF regularization parameter) and shows the effect on predictions.
6. THE Demo_Notebook SHALL include a markdown note indicating that SMT-based models require the `standard-evaluator[smt]` optional dependency installed via `pip install standard-evaluator[smt]`.
7. THE Demo_Notebook SHALL exclude long-running SMT model training examples (such as those requiring extended fitting times) and SHALL be executable without errors and complete within 120 seconds when the `smt` optional dependency is installed.

### Requirement 3: EvaluatorOpenMdaoComponent Documentation

**User Story:** As an OpenMDAO user, I want a tutorial notebook showing how to wrap any Evaluator as an OpenMDAO component, so that I can integrate Standard Evaluator analyses into OpenMDAO models.

#### Acceptance Criteria

1. WHEN a user opens the OpenMDAO component notebook, THE Demo_Notebook SHALL contain at least one markdown cell explaining the purpose of EvaluatorOpenMdaoComponent, the mapping from OptProblem variables to OpenMDAO inputs, and the mapping from OptProblem responses to OpenMDAO outputs.
2. THE Demo_Notebook SHALL include a code example that defines an OptProblem with at least two variables and one response, creates a NumpyEvaluator, wraps it in an EvaluatorOpenMdaoComponent, adds the component to an OpenMDAO Problem, runs the model, and prints the computed output values.
3. THE Demo_Notebook SHALL include a code example that retrieves and displays the OpenMDAO input metadata (default values, bounds) and output metadata (bounds, scaling) of the EvaluatorOpenMdaoComponent to show how OptProblem definitions map to OpenMDAO variable properties.
4. THE Demo_Notebook SHALL include a code example that calls `to_dict` on the evaluator to obtain serialized options, constructs a new EvaluatorOpenMdaoComponent by passing those options as the `evaluator_options` keyword argument, runs the reconstructed component in an OpenMDAO Problem, and prints output values matching those of the original component.
5. THE Demo_Notebook SHALL be executable without errors in the project virtual environment using only the required dependencies defined in the project configuration.

### Requirement 4: Array Variable Documentation

**User Story:** As a developer working with multi-dimensional inputs and outputs, I want a tutorial notebook explaining how to define and use ArrayVariables, so that I can model vector and matrix quantities in my evaluators.

#### Acceptance Criteria

1. WHEN a user opens the array variable notebook, THE Demo_Notebook SHALL contain a markdown overview explaining the ArrayVariable type, its fields (shape, bounds, shift, scale, default, units), its relationship to FloatVariable, and the distinction between rolled (single NumPy array in a DataFrame cell) and unrolled (individual scalar columns) representations.
2. THE Demo_Notebook SHALL include a working code example that creates at least one 1-D ArrayVariable (shape with a single dimension) and at least one 2-D ArrayVariable (shape with two dimensions), each with explicitly set bounds.
3. THE Demo_Notebook SHALL include a working code example that creates a NumpyEvaluator using ArrayVariable inputs and outputs, invokes it via `__call__` with rolled array data in a DataFrame, and invokes it via `eval_np` with unrolled data, displaying results from both calls.
4. THE Demo_Notebook SHALL include a working code example that instantiates an EvaluatorOpenMdaoComponent with an evaluator that uses ArrayVariable inputs and outputs, calls `setup()` on an OpenMDAO Problem containing the component, and verifies the component runs `compute` without errors.
5. THE Demo_Notebook SHALL include a markdown section explaining the name generation pattern for unrolled array variables, covering both 1-D arrays (e.g., `input[0]`, `input[1]`) and multi-dimensional arrays (e.g., `matrix[0,0]`, `matrix[0,1]`, `matrix[1,0]`).
6. THE Demo_Notebook SHALL contain all specified content (overview, code examples, explanations) and SHALL be executable without errors by running all cells sequentially in a fresh kernel using the project virtual environment with the standard `pip install -e .` dependencies and the `openmdao` package installed.

### Requirement 5: Benchmark Test Problems Documentation

**User Story:** As a researcher or developer testing optimization algorithms, I want a tutorial notebook listing and demonstrating the available benchmark test evaluators, so that I can select appropriate test problems for my work.

#### Acceptance Criteria

1. WHEN a user opens the benchmark problems notebook, THE Demo_Notebook SHALL contain a markdown overview listing all available benchmark test evaluators organized by category (unconstrained, constrained, multi-fidelity, feasibility-discovery), with each entry showing the evaluator class name and number of design variables.
2. THE Demo_Notebook SHALL include a working code example that instantiates at least three different benchmark evaluators (one unconstrained, one constrained, one feasibility-discovery), evaluates each at a minimum of 2 sample points provided as a pandas DataFrame, and displays the resulting output DataFrame including response columns.
3. THE Demo_Notebook SHALL include a markdown description of each of the five feasibility-discovery benchmarks (SmallCircleFeasibleRegion, DisconnectedFeasibleRegions, G6Problem, G7Problem, SpeedReducer), where each description states the number of design variables, the number of constraints, and the purpose of the problem (what feasibility challenge it represents).
4. THE Demo_Notebook SHALL include a working code example that accesses the OptProblem of a benchmark evaluator and prints its variables (names and bounds), responses (names), objectives, and constraints.
5. THE Demo_Notebook SHALL be executable without errors in the project virtual environment using only required dependencies listed in the project's pyproject.toml.
6. THE Demo_Notebook SHALL include a working code example that retrieves the known_solution property of at least one benchmark evaluator and displays the resulting DataFrame containing optimal input values and corresponding response values.

### Requirement 6: Evaluator Interface Documentation

**User Story:** As a developer, I want a tutorial notebook explaining how to define the interface for an evaluator using OptProblem and EvaluatorInfo, so that I can properly configure evaluators with typed variables, bounds, and metadata.

#### Acceptance Criteria

1. WHEN a user opens the evaluator interface notebook, THE Demo_Notebook SHALL contain a markdown overview explaining the two supported ways to define an evaluator interface: via `OptProblem` (for optimization problems with objectives and constraints) and via `EvaluatorInfo` (for general input/output descriptions).
2. THE Demo_Notebook SHALL include a code example that creates an OptProblem with at least 2 FloatVariable inputs (each specifying name and bounds), at least 1 FloatVariable response, and defines at least one objective and one constraint referencing valid response names, and executes without error.
3. THE Demo_Notebook SHALL include a code example that creates an EvaluatorInfo with at least 2 typed inputs and at least 1 typed output using Variable types from the supported union (FloatVariable, IntVariable, ArrayVariable, or CategoricalVariable), and uses it to instantiate a NumpyEvaluator by passing it as the `interface` parameter, executing without error.
4. THE Demo_Notebook SHALL include a code example that creates a NumpyEvaluator using an OptProblem passed as the `opt_problem` parameter, calls the evaluator with a pandas DataFrame containing valid input values, and accesses the `inputs` and `outputs` properties to retrieve the list of variable names and response names respectively.
5. THE Demo_Notebook SHALL include a markdown section explaining when to use OptProblem versus EvaluatorInfo, stating that OptProblem is used when objectives and constraints are defined for optimization, and EvaluatorInfo is used for general-purpose evaluators that only require input/output descriptions.
6. THE Demo_Notebook SHALL be executable without errors in the project virtual environment using only required dependencies.

### Requirement 7: Documentation Index Integration

**User Story:** As a documentation maintainer, I want all new demo notebooks registered in the Sphinx toctree, so that they appear in the built documentation site navigation.

#### Acceptance Criteria

1. WHEN the documentation is built, THE Demos_Index SHALL include one toctree entry for each of the following 6 new demo notebooks: evaluator hierarchy, surrogate models, OpenMDAO component, array variables, benchmark problems, and evaluator interface, such that each entry renders as a navigation link in the built HTML site.
2. THE Demos_Index SHALL list the new demo notebook entries after the existing entries (Problem_explanation, defining_options, group_creation_NASA, group_manipulation, group_reading) in the toctree directive, in the following order: evaluator hierarchy, surrogate models, OpenMDAO component, array variables, benchmark problems, evaluator interface.
3. WHEN the Sphinx build command is executed, THE Documentation_System SHALL complete with zero warnings and zero errors referencing any of the 6 new notebook files.
4. IF a new notebook file referenced in the toctree does not exist in the demos directory, THEN THE Documentation_System SHALL produce a Sphinx build warning identifying the missing file.

### Requirement 8: Notebook Consistency Standards

**User Story:** As a documentation maintainer, I want all new notebooks to follow a consistent style matching the existing demos, so that the documentation has a uniform appearance and quality.

#### Acceptance Criteria

1. THE Demo_Notebook SHALL begin with a level-1 markdown heading stating the notebook title.
2. THE Demo_Notebook SHALL include an Overview paragraph in the first markdown cell, directly below the level-1 heading, explaining what the notebook demonstrates and what the reader will learn.
3. THE Demo_Notebook SHALL include a markdown cell before each code cell that introduces a new concept, operation, or step, describing what the following code does or why it is needed.
4. THE Demo_Notebook SHALL include a markdown cell after any code cell whose output contains numeric results, warnings, or data structures that require interpretation to explain what the output represents.
5. THE Demo_Notebook SHALL use the `import standard_evaluator as se` convention as the primary import pattern.
6. IF a notebook requires an optional dependency, THEN THE Demo_Notebook SHALL state the required install command (e.g., `pip install standard-evaluator[smt]`) in a markdown cell before the first code cell that uses the dependency. IF a notebook uses an optional dependency without a preceding install command, THEN the notebook SHALL fail validation.
7. THE Demo_Notebook SHALL contain no consecutive code cells without an intervening markdown cell that provides context or transitions between them.

### Requirement 9: API Reference Documentation

**User Story:** As a developer integrating the Standard Evaluator library into my project, I want comprehensive API reference documentation for all public classes, so that I can understand the constructor signatures, methods, properties, and options of each component without reading the source code.

#### Acceptance Criteria

1. THE Documentation_System SHALL include an API reference section that documents all public classes in the evaluator hierarchy (Evaluator, NumpyEvaluator, ExecutableEvaluator, ShiftScaleEvaluator, OpenMdaoEvaluator, ExcelEvaluator, MatlabEvaluator), where each class page includes at minimum: a class description, constructor parameters with types, all public methods with signatures and parameter descriptions, and all public properties.
2. THE Documentation_System SHALL include an API reference section that documents all public classes in the surrogate model hierarchy (SurrogateModel, PolynomialModel, RBFModel, IDWModel, GENNModel, LSModel, RMTSModel, SOPAModel), where each class page includes at minimum: a class description, constructor parameters with types, all public methods with signatures and parameter descriptions, and all public properties.
3. THE Documentation_System SHALL include an API reference section that documents the EvaluatorOpenMdaoComponent class, where the page includes at minimum: a class description, constructor parameters with types, all public methods with signatures and parameter descriptions, and all public properties.
4. THE Documentation_System SHALL include an API reference section that documents the data model classes (OptProblem, EvaluatorInfo, GroupInfo, FloatVariable, IntVariable, ArrayVariable, CategoricalVariable), where each class page includes at minimum: a class description, all fields/attributes with types and descriptions, and all public methods.
5. THE Documentation_System SHALL include an API reference section that documents the decorator modules (caching, logging), where the page includes at minimum: a description of the decorator's purpose, function signature, accepted parameters, and return type.
6. WHEN the Sphinx documentation is built, THE Documentation_System SHALL auto-generate API reference pages from the source docstrings using Sphinx autodoc (autoclass directives) with the `autodocsumm` extension, producing one dedicated page per class that includes inherited members, method summaries, and attribute summaries. IF autodoc is disabled or unavailable, THEN the Sphinx build SHALL fail with an error indicating that autodoc is required for API reference generation.
7. THE Documentation_System SHALL build the API reference producing zero Sphinx warnings or errors of type "missing docstring" or "broken cross-reference" across all pages in the reference section.
