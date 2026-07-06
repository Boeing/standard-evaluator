# Implementation Plan: Feature Documentation

## Overview

Create comprehensive tutorial-style Jupyter notebook documentation and API reference pages for the Standard Evaluator library. This involves creating 6 new demo notebooks in `docs/source/demos/`, a new `docs/source/reference/` API section, and updating the demos index toctree. All notebooks use the `import standard_evaluator as se` convention and follow the alternating markdown/code cell pattern.

## Tasks

- [x] 1. Create the API reference directory and index
  - [x] 1.1 Create `docs/source/reference/index.rst` with toctree listing all API sub-pages (evaluators, surrogate_models, components, data_models, decorators)
    - The `docs/source/index.rst` already references `reference/index`
    - Create the directory and toctree file
    - _Requirements: 9.6_

  - [x] 1.2 Create `docs/source/reference/evaluators.rst` with autoclass directives for all evaluator classes
    - Include: Evaluator, NumpyEvaluator, ExecutableEvaluator, ShiftScaleEvaluator, OpenMDAOEvaluator, ExcelEvaluator, MatlabEvaluator
    - Use `autoclass` with `:members:`, `:show-inheritance:`, `:special-members: __init__, __call__`
    - _Requirements: 9.1_

  - [x] 1.3 Create `docs/source/reference/surrogate_models.rst` with autoclass directives for all surrogate model classes
    - Include: SurrogateModel, PolynomialModel, RBFModel (RadialBasisFunctionModel), IDWModel (InverseDistanceWeightingModel), GENNModel, LSModel, RMTSModel, SOPAModel
    - _Requirements: 9.2_

  - [x] 1.4 Create `docs/source/reference/components.rst` with autoclass directive for EvaluatorOpenMdaoComponent
    - _Requirements: 9.3_

  - [x] 1.5 Create `docs/source/reference/data_models.rst` with autoclass directives for data model classes
    - Include: OptProblem, EvaluatorInfo, GroupInfo, FloatVariable, IntVariable, ArrayVariable, CategoricalVariable
    - _Requirements: 9.4_

  - [x] 1.6 Create `docs/source/reference/decorators.rst` with automodule directives for decorator modules
    - Include: `standard_evaluator.evaluators.decorators.cache`, `standard_evaluator.evaluators.decorators.site_logger`
    - _Requirements: 9.5_

- [x] 2. Create the evaluator hierarchy demo notebook
  - [x] 2.1 Create `docs/source/demos/evaluator_hierarchy.ipynb` with markdown overview of Evaluator class hierarchy and code examples
    - Start with level-1 heading "Evaluator Hierarchy" and overview paragraph
    - Include markdown explaining abstract Evaluator class, subclasses (NumpyEvaluator, ExecutableEvaluator, ShiftScaleEvaluator, OpenMdaoEvaluator, ExcelEvaluator, MatlabEvaluator), and decorators
    - Code example: create a NumpyEvaluator from a benchmark test evaluator, build DataFrame with ≥2 rows, call via `__call__`, display results
    - Code example: demonstrate `eval_np` and `eval_list` with markdown explaining they exclude fixed variables and unroll ArrayVariables
    - Code example: apply caching decorator via `cache` parameter, evaluate, call `get_cache()`
    - Code example: apply logging decorator via `logging=True`, evaluate, call `get_log()`
    - Use `import standard_evaluator as se` convention throughout
    - Must run with base dependencies only (no smt, aviary, or excel extras)
    - Ensure alternating markdown/code cells with no consecutive code cells
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.7_

- [x] 3. Create the surrogate model demo notebook
  - [x] 3.1 Create `docs/source/demos/surrogate_models.ipynb` with surrogate model training, prediction, serialization, and options examples
    - Start with level-1 heading "Surrogate Models" and overview paragraph
    - Include markdown note about `pip install standard-evaluator[smt]` before any SMT import
    - Markdown overview listing SurrogateModel base, PolynomialModel, SMT models (RBF, IDW, GENN, LS, RMTS, SOPA) with inheritance hierarchy
    - Code example: train PolynomialModel on ≥10 sites from a test evaluator, predict at ≥3 new sites, display predicted vs true
    - Code example: serialize with `to_dict`, reconstruct with `from_dict`, verify predictions match
    - Code example: train RBF model on ≥10 sites, display accuracy metric (RMSE or R²)
    - Markdown section explaining options system + code example with non-default option value
    - Must complete within 120 seconds with smt installed
    - Ensure alternating markdown/code cells
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 4. Create the OpenMDAO component demo notebook
  - [x] 4.1 Create `docs/source/demos/openmdao_component.ipynb` with EvaluatorOpenMdaoComponent wrapping and variable mapping examples
    - Start with level-1 heading "OpenMDAO Component Integration" and overview paragraph
    - Markdown explaining purpose of EvaluatorOpenMdaoComponent, variable-to-input and response-to-output mapping
    - Code example: define OptProblem with ≥2 variables and 1 response, create NumpyEvaluator, wrap in EvaluatorOpenMdaoComponent, add to OpenMDAO Problem, run model, print outputs
    - Code example: retrieve and display OpenMDAO input/output metadata (defaults, bounds, scaling)
    - Code example: call `to_dict` on evaluator, construct new component via `evaluator_options` kwarg, run, verify outputs match
    - Must run with base dependencies (openmdao is in required dependencies)
    - Ensure alternating markdown/code cells
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.7_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Create the array variables demo notebook
  - [x] 6.1 Create `docs/source/demos/array_variables.ipynb` with ArrayVariable definition, rolled/unrolled usage, and OpenMDAO integration
    - Start with level-1 heading "Array Variables" and overview paragraph
    - Markdown overview explaining ArrayVariable fields (shape, bounds, shift, scale, default, units), relationship to FloatVariable, rolled vs unrolled representations
    - Code example: create 1-D ArrayVariable and 2-D ArrayVariable with explicit bounds
    - Code example: create NumpyEvaluator with ArrayVariable inputs/outputs, invoke via `__call__` (rolled) and `eval_np` (unrolled), display both
    - Code example: instantiate EvaluatorOpenMdaoComponent with array evaluator, setup OpenMDAO Problem, verify compute runs
    - Markdown section explaining unrolled name generation: 1-D (`input[0]`, `input[1]`) and multi-D (`matrix[0,0]`, `matrix[0,1]`)
    - Must run with base dependencies + openmdao
    - Ensure alternating markdown/code cells
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.7_

- [x] 7. Create the benchmark problems demo notebook
  - [x] 7.1 Create `docs/source/demos/benchmark_problems.ipynb` with benchmark evaluator catalog, evaluation, and known solutions
    - Start with level-1 heading "Benchmark Test Problems" and overview paragraph
    - Markdown listing all benchmark evaluators by category (unconstrained, constrained, multi-fidelity, feasibility-discovery) with class names and number of design variables
    - Code example: instantiate ≥3 benchmarks (one unconstrained, one constrained, one feasibility-discovery), evaluate at ≥2 sample points, display output DataFrames
    - Markdown describing the 5 feasibility-discovery benchmarks (SmallCircleFeasibleRegion, DisconnectedFeasibleRegions, G6Problem, G7Problem, SpeedReducer) with variable count, constraints, and purpose
    - Code example: access OptProblem of a benchmark, print variables (names/bounds), responses, objectives, constraints
    - Code example: retrieve `known_solution` property of a benchmark, display DataFrame
    - Must run with base dependencies only
    - Ensure alternating markdown/code cells
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.7_

- [x] 8. Create the evaluator interface demo notebook
  - [x] 8.1 Create `docs/source/demos/evaluator_interface.ipynb` with OptProblem and EvaluatorInfo usage patterns
    - Start with level-1 heading "Evaluator Interface" and overview paragraph
    - Markdown explaining two interface approaches: OptProblem (for optimization with objectives/constraints) vs EvaluatorInfo (general I/O descriptions)
    - Code example: create OptProblem with ≥2 FloatVariable inputs (name + bounds), ≥1 response, one objective, one constraint
    - Code example: create EvaluatorInfo with ≥2 typed inputs and ≥1 output using Variable types, instantiate NumpyEvaluator via `interface` parameter
    - Code example: create NumpyEvaluator with OptProblem via `opt_problem` parameter, call evaluator, access `inputs` and `outputs` properties
    - Markdown section on when to use OptProblem vs EvaluatorInfo
    - Must run with base dependencies only
    - Ensure alternating markdown/code cells
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.7_

- [x] 9. Update the demos index toctree
  - [x] 9.1 Update `docs/source/demos/index.rst` to add 6 new notebook entries after existing entries
    - Add entries in order: evaluator_hierarchy, surrogate_models, openmdao_component, array_variables, benchmark_problems, evaluator_interface
    - Place after existing entries (Problem_explanation, defining_options, group_creation_NASA, group_manipulation, group_reading)
    - _Requirements: 7.1, 7.2_

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Write notebook structure validation tests
  - [x] 11.1 Write property test for notebook structure invariant
    - **Property 1: Notebook Structure Invariant**
    - **Validates: Requirements 8.1, 8.2, 8.7**
    - Test that all 6 notebooks start with markdown cell containing level-1 heading, have overview paragraph, and contain no consecutive code cells

  - [x] 11.2 Write integration test for Sphinx build completeness
    - **Property 2: Build Completeness**
    - **Validates: Requirements 7.3, 9.7**
    - Run `sphinx-build -W` and verify zero warnings/errors for new notebooks and reference pages

  - [x] 11.3 Write integration test for notebook execution correctness
    - **Property 3: Notebook Execution Correctness**
    - **Validates: Requirements 1.7, 2.7, 3.5, 4.6, 5.5, 6.6**
    - Execute each notebook in a fresh kernel using `nbconvert` ExecutePreprocessor and verify zero exceptions

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- The API reference section (task 1) is independent of the demo notebooks (tasks 2–8) and can be built in parallel
- All notebooks use `import standard_evaluator as se` as the primary import pattern
- Only notebook 2 (surrogate_models) requires the `smt` optional dependency; all others use base dependencies
- The `reference/` directory does not yet exist and must be created as part of task 1
- Notebook content should use benchmark test evaluators from `se.evaluators.test` for examples

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "1.5", "1.6", "4.1"] },
    { "id": 2, "tasks": ["6.1", "7.1", "8.1"] },
    { "id": 3, "tasks": ["9.1"] },
    { "id": 4, "tasks": ["11.1", "11.2", "11.3"] }
  ]
}
```
