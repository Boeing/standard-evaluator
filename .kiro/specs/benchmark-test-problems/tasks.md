# Implementation Plan: Benchmark Test Problems

## Overview

Add five analytical benchmark test problems as `TestEvaluator` subclasses to the Standard Evaluator library. Each evaluator implements a well-defined mathematical optimization/feasibility problem following the established patterns in the codebase (e.g., `constrained_betts.py`). The implementation proceeds evaluator-by-evaluator, from simplest to most complex, with unit tests and property tests alongside each, finishing with library integration.

## Tasks

- [x] 1. Implement SmallCircleFeasibleRegion evaluator
  - [x] 1.1 Create `src/standard_evaluator/evaluators/test/small_circle_feasible_region.py`
    - Define `SmallCircleFeasibleRegionOptions` Pydantic BaseModel with fields: `a` (float, default 0.5), `b` (float, default -0.3), `r` (float, default 0.15)
    - Implement `SmallCircleFeasibleRegion(TestEvaluator)` with `_define_options`, `_create_opt_problem`, `_evaluate`, and `_def_initial_guess`
    - `_create_opt_problem`: 2 variables x1, x2 ∈ [-1, 1]; 1 response g1 with bounds (-inf, 0]; objectives=[]; constraints=["g1"]
    - `_evaluate`: compute `g1 = (x1 − a)² + (x2 − b)² − r²` using `lookup_option_value`
    - `_def_initial_guess`: return `[0.0, 0.0]`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.2 Write unit tests for SmallCircleFeasibleRegion
    - Create `tests/evaluators/test/test_small_circle_feasible_region.py`
    - Test instantiation produces a `TestEvaluator` instance
    - Test problem structure (variable names, bounds, response names, objectives, constraints)
    - Test feasible point at circle center (0.5, -0.3): g1 = -r² = -0.0225 within 1e-8
    - Test infeasible point at origin (0.0, 0.0): g1 > 0
    - Test custom options override
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6_

  - [x] 1.3 Write property test for SmallCircleFeasibleRegion formula
    - **Property 1: SmallCircleFeasibleRegion formula correctness**
    - **Validates: Requirements 1.6**
    - Create property test in the unit test module using Hypothesis
    - Generate random (x1, x2) within [-1, 1] and valid (a, b, r) options
    - Independently compute expected g1 = (x1−a)² + (x2−b)² − r²
    - Assert evaluator output matches within 1e-10 relative tolerance

- [x] 2. Implement DisconnectedFeasibleRegions evaluator
  - [x] 2.1 Create `src/standard_evaluator/evaluators/test/disconnected_feasible_regions.py`
    - Define `DisconnectedFeasibleRegionsOptions` Pydantic BaseModel with fields: `centers`, `radii`, `smooth`, `tau`; include validator ensuring len(centers) == len(radii) and tau > 0
    - Implement `DisconnectedFeasibleRegions(TestEvaluator)` with `_define_options`, `_create_opt_problem`, `_evaluate`, `_def_initial_guess`
    - `_create_opt_problem`: 2 variables x1, x2 ∈ [-1, 1]; 1 response g1 bounds (-inf, 0]; objectives=[]; constraints=["g1"]
    - `_evaluate` (smooth=False): compute g1 = min over ℓ of [(x1−a_ℓ)² + (x2−b_ℓ)² − r_ℓ²]
    - `_evaluate` (smooth=True): compute g1 = −τ·log(Σ_ℓ exp(−d_ℓ/τ)) with log-sum-exp stability trick
    - `_def_initial_guess`: return `[0.0, 0.0]`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [x] 2.2 Write unit tests for DisconnectedFeasibleRegions
    - Create `tests/evaluators/test/test_disconnected_feasible_regions.py`
    - Test instantiation produces a `TestEvaluator` instance
    - Test problem structure (variables, responses, objectives, constraints)
    - Test feasible point at first island center (-0.5, -0.4): g1 ≤ 0 for both smooth=False and smooth=True
    - Test infeasible point at origin (0.0, 0.0): g1 > 0 for both formulations
    - Test Pydantic validation rejects mismatched centers/radii lengths
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

  - [x] 2.3 Write property test for DisconnectedFeasibleRegions nonsmooth formula
    - **Property 2: DisconnectedFeasibleRegions nonsmooth formula correctness**
    - **Validates: Requirements 2.6**
    - Generate random (x1, x2) within [-1, 1] and valid island configurations
    - Independently compute expected g1 = min(d_ℓ)
    - Assert evaluator output matches within 1e-10 relative tolerance

  - [x] 2.4 Write property test for DisconnectedFeasibleRegions smooth formula
    - **Property 3: DisconnectedFeasibleRegions smooth formula correctness**
    - **Validates: Requirements 2.7**
    - Generate random (x1, x2) within [-1, 1], valid island configs, and tau > 0
    - Independently compute expected g1 = −τ·log(Σ_ℓ exp(−d_ℓ/τ)) using log-sum-exp trick
    - Assert evaluator output matches within 1e-10 relative tolerance

- [x] 3. Implement G6Problem evaluator
  - [x] 3.1 Create `src/standard_evaluator/evaluators/test/g6_problem.py`
    - Implement `G6Problem(TestEvaluator)` with `_create_opt_problem`, `_evaluate`, `_def_initial_guess`, `_def_known_solution`
    - `_create_opt_problem`: x1 ∈ [13, 100], x2 ∈ [0, 100]; responses f, g1, g2; objectives=["f"]; constraints=["g1", "g2"]
    - `_evaluate`: compute f = (x1−10)³ + (x2−20)³, g1 = −(x1−5)² − (x2−5)² + 100, g2 = (x1−6)² + (x2−5)² − 82.81
    - `_def_initial_guess`: return `[20.0, 5.5]`
    - `_def_known_solution`: return pd.Series with x1=14.095, x2=0.84296
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 3.2 Write unit tests for G6Problem
    - Create `tests/evaluators/test/test_g6_problem.py`
    - Test instantiation produces a `TestEvaluator` instance
    - Test problem structure (variables, bounds, responses, objectives, constraints)
    - Test feasible point at known optimum (14.095, 0.84296): g1 ≤ 0 and g2 ≤ 0
    - Test infeasible point (20.0, 50.0): at least one g_j > 0
    - Test known_solution property returns correct values
    - _Requirements: 7.1, 7.2, 7.3, 7.6_

  - [x] 3.3 Write property test for G6Problem formula
    - **Property 4: G6Problem all-responses formula correctness**
    - **Validates: Requirements 3.4, 3.5, 3.6**
    - Generate random (x1, x2) within [13, 100] × [0, 100]
    - Independently compute expected f, g1, g2
    - Assert evaluator output matches within 1e-10 relative tolerance

- [x] 4. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement G7Problem evaluator
  - [x] 5.1 Create `src/standard_evaluator/evaluators/test/g7_problem.py`
    - Implement `G7Problem(TestEvaluator)` with `_create_opt_problem`, `_evaluate`, `_def_initial_guess`
    - `_create_opt_problem`: 10 variables x1–x10 each ∈ [-10, 10]; 9 responses (f, g1–g8); objectives=["f"]; constraints=["g1",...,"g8"]
    - `_evaluate`: compute f and g1–g8 per the G7 formulas
    - `_def_initial_guess`: return `[0.0] * 10`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 5.2 Write unit tests for G7Problem
    - Create `tests/evaluators/test/test_g7_problem.py`
    - Test instantiation produces a `TestEvaluator` instance
    - Test problem structure (10 variables, 9 responses, objectives, constraints)
    - Test feasible point from literature: all g1–g8 ≤ 0
    - Test infeasible point at origin (0,...,0): at least one g_j > 0
    - _Requirements: 7.1, 7.2, 7.3, 7.6_

  - [x] 5.3 Write property test for G7Problem formula
    - **Property 5: G7Problem all-responses formula correctness**
    - **Validates: Requirements 4.4, 4.5**
    - Generate random 10-tuple within [-10, 10]¹⁰
    - Independently compute expected f and g1–g8
    - Assert evaluator output matches within 1e-10 relative tolerance

- [x] 6. Implement SpeedReducer evaluator
  - [x] 6.1 Create `src/standard_evaluator/evaluators/test/speed_reducer.py`
    - Implement `SpeedReducer(TestEvaluator)` with `_create_opt_problem`, `_evaluate`, `_def_initial_guess`
    - `_create_opt_problem`: 7 variables with specified bounds; 12 responses (f, g1–g11); objectives=["f"]; constraints=["g1",...,"g11"]
    - `_evaluate`: compute f and g1–g11 per the speed reducer formulas
    - `_def_initial_guess`: return midpoints of variable bounds
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 6.2 Write unit tests for SpeedReducer
    - Create `tests/evaluators/test/test_speed_reducer.py`
    - Test instantiation produces a `TestEvaluator` instance
    - Test problem structure (7 variables with correct bounds, 12 responses, objectives, constraints)
    - Test feasible point (3.5, 0.7, 17, 7.3, 7.8, 3.35, 5.287): all g1–g11 ≤ 0
    - Test infeasible point (bounds midpoints or other analytically infeasible point): at least one g_j > 0
    - _Requirements: 7.1, 7.2, 7.3, 7.6_

  - [x] 6.3 Write property test for SpeedReducer formula
    - **Property 6: SpeedReducer all-responses formula correctness**
    - **Validates: Requirements 5.4, 5.5**
    - Generate random 7-tuple within the defined variable bounds
    - Independently compute expected f and g1–g11
    - Assert evaluator output matches within 1e-10 relative tolerance

- [x] 7. Library integration and finite-response property test
  - [x] 7.1 Update `src/standard_evaluator/evaluators/test/__init__.py` to change `_EXPECTED_COUNT` from 38 to 43
    - _Requirements: 6.1, 6.2_

  - [x] 7.2 Write property test for finite responses across all evaluators
    - **Property 7: All evaluators produce finite responses for any input**
    - **Validates: Requirements 6.3, 6.4**
    - Parametrize over all 5 new evaluator classes
    - Generate random inputs within variable bounds
    - Assert no exception raised and all response values are finite (not NaN, not ±inf)

- [x] 8. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific known points, problem structure, and integration
- All evaluators follow the pattern established by `constrained_betts.py` and other existing test evaluators
- The auto-discovery mechanism in `__init__.py` handles registration — no manual imports needed beyond updating the count

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1", "5.1", "6.1"] },
    { "id": 1, "tasks": ["2.1", "1.2", "1.3", "3.2", "3.3", "5.2", "5.3", "6.2", "6.3"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "7.1"] },
    { "id": 3, "tasks": ["7.2"] }
  ]
}
```
