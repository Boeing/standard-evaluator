# Requirements Document

## Introduction

This feature adds five analytical benchmark test problems to the Standard Evaluator (SE) library. These problems serve as feasibility-discovery benchmarks for constrained optimization algorithms. Each problem is implemented as a concrete subclass of `TestEvaluator`, following the existing patterns in `src/standard_evaluator/evaluators/test/`. The problems range from 2D single-constraint cases to a 7D engineering design problem with 11 constraints, providing a progression of difficulty for testing feasible-point search methods.

## Glossary

- **TestEvaluator**: The abstract base class in SE that all test evaluators inherit from. Provides the interface for defining optimization problems, evaluating responses, and reporting known solutions.
- **OptProblem**: The Pydantic model representing a fully defined optimization problem including variables, responses, objectives, and constraints.
- **Evaluator**: A callable object that accepts a DataFrame of input sites and populates it with computed response values.
- **Feasibility_Constraint**: An inequality constraint of the form g(x) ≤ 0. A design is feasible when all constraints are satisfied.
- **Design_Space**: The hypercube defined by variable bounds within which the optimization operates.
- **SE_Library**: The `standard_evaluator` Python package.
- **Soft_Min**: A smooth approximation to the minimum function using log-sum-exp: −τ·log[Σ exp(−d_ℓ/τ)].

## Requirements

### Requirement 1: Small Circular Feasible Region Evaluator

**User Story:** As a researcher, I want a test evaluator with a single small circular feasible region inside a larger design domain, so that I can benchmark algorithms that must locate low-volume feasible islands.

#### Acceptance Criteria

1. THE SmallCircleFeasibleRegion evaluator SHALL inherit from TestEvaluator and reside in `src/standard_evaluator/evaluators/test/`.
2. THE SmallCircleFeasibleRegion module SHALL define a `SmallCircleFeasibleRegionOptions` Pydantic BaseModel with fields: a (float, default 0.5), b (float, default −0.3), r (float, default 0.15).
3. THE SmallCircleFeasibleRegion class SHALL override `_define_options` as a classmethod returning `SmallCircleFeasibleRegionOptions`.
4. WHEN _create_opt_problem is called, THE SmallCircleFeasibleRegion evaluator SHALL create an OptProblem with 2 continuous variables named "x1" and "x2" each bounded to [-1, 1], and 1 response named "g1" with bounds (-inf, 0].
5. WHEN _create_opt_problem is called, THE SmallCircleFeasibleRegion evaluator SHALL set the objectives list to empty and the constraints list to ["g1"], making this a feasibility-only problem.
6. WHEN _evaluate is called with a DataFrame of sites, THE SmallCircleFeasibleRegion evaluator SHALL retrieve a, b, r via `lookup_option_value` and compute g1 = (x1−a)² + (x2−b)² − r² for each row, storing the result in the "g1" column of the sites DataFrame.
7. WHEN _create_opt_problem is called, THE SmallCircleFeasibleRegion evaluator SHALL set the initial guess to [0.0, 0.0] via `_def_initial_guess`.

### Requirement 2: Disconnected Feasible Regions Evaluator

**User Story:** As a researcher, I want a test evaluator with multiple disconnected circular feasible islands and an optional smooth soft-min variant, so that I can benchmark algorithms that must discover feasibility across separated regions.

#### Acceptance Criteria

1. THE DisconnectedFeasibleRegions evaluator SHALL inherit from TestEvaluator and reside in `src/standard_evaluator/evaluators/test/`.
2. THE DisconnectedFeasibleRegions module SHALL define a `DisconnectedFeasibleRegionsOptions` Pydantic BaseModel with fields: centers (list of (a, b) tuples, default [(-0.5,-0.4), (0.4,0.3), (-0.2,0.6)]), radii (list of floats, default [0.15, 0.12, 0.10]), smooth (bool, default False), tau (float, default 0.01, must be greater than 0). The model SHALL validate that centers and radii have the same length.
3. THE DisconnectedFeasibleRegions class SHALL override `_define_options` to return `DisconnectedFeasibleRegionsOptions`.
4. WHEN _create_opt_problem is called, THE DisconnectedFeasibleRegions evaluator SHALL define 2 continuous variables (x1, x2) with bounds [-1, 1] for each.
5. WHEN _create_opt_problem is called, THE DisconnectedFeasibleRegions evaluator SHALL define 1 response named "g1" with bounds [-inf, 0] representing the union-of-circles constraint.
6. WHEN _evaluate is called with smooth=False (default), THE DisconnectedFeasibleRegions evaluator SHALL retrieve island parameters via `lookup_option_value` and compute g1 = min over ℓ of [(x1−a_ℓ)² + (x2−b_ℓ)² − r_ℓ²] for each row.
7. WHEN _evaluate is called with smooth=True, THE DisconnectedFeasibleRegions evaluator SHALL retrieve tau via `lookup_option_value` and compute g1 = −τ·log[Σ_ℓ exp(−d_ℓ/τ)] where d_ℓ = (x1−a_ℓ)² + (x2−b_ℓ)² − r_ℓ².
8. THE DisconnectedFeasibleRegions evaluator SHALL define an empty objectives list and a constraints list containing "g1".
9. THE DisconnectedFeasibleRegions evaluator SHALL define `_def_initial_guess` returning [0.0, 0.0] as the default starting point for optimizers.

### Requirement 3: G6 Problem Evaluator

**User Story:** As a researcher, I want a test evaluator implementing the G6 constrained benchmark problem, so that I can compare feasibility-discovery algorithms against an established 2D benchmark from the literature.

#### Acceptance Criteria

1. THE G6Problem evaluator SHALL inherit from TestEvaluator and reside in `src/standard_evaluator/evaluators/test/`.
2. WHEN _create_opt_problem is called, THE G6Problem evaluator SHALL define 2 continuous variables: x1 with bounds [13, 100] and x2 with bounds [0, 100].
3. WHEN _create_opt_problem is called, THE G6Problem evaluator SHALL define 3 responses: "f" with bounds [-inf, inf], "g1" with bounds [-inf, 0], and "g2" with bounds [-inf, 0].
4. WHEN _evaluate is called, THE G6Problem evaluator SHALL compute f = (x1−10)³ + (x2−20)³ for each row.
5. WHEN _evaluate is called, THE G6Problem evaluator SHALL compute g1 = −(x1−5)² − (x2−5)² + 100 for each row.
6. WHEN _evaluate is called, THE G6Problem evaluator SHALL compute g2 = (x1−6)² + (x2−5)² − 82.81 for each row.
7. THE G6Problem evaluator SHALL define objectives as ["f"] and constraints as ["g1", "g2"].
8. WHEN _def_known_solution is called, THE G6Problem evaluator SHALL return a pd.Series with x1 = 14.095 and x2 = 0.84296, corresponding to the known optimal solution f ≈ -6961.81388 where both constraints are satisfied (g1 ≤ 0 and g2 ≤ 0).

### Requirement 4: G7 Problem Evaluator

**User Story:** As a researcher, I want a test evaluator implementing the G7 constrained benchmark problem, so that I can benchmark feasibility-discovery algorithms on a moderate-dimensional (10D) problem with eight nonlinear constraints.

#### Acceptance Criteria

1. THE G7Problem evaluator SHALL inherit from TestEvaluator and reside in `src/standard_evaluator/evaluators/test/`.
2. WHEN _create_opt_problem is called, THE G7Problem evaluator SHALL define 10 continuous variables (x1 through x10) each with bounds [-10, 10].
3. WHEN _create_opt_problem is called, THE G7Problem evaluator SHALL define 9 responses: "f" with bounds [-inf, inf], and "g1" through "g8" each with bounds [-inf, 0].
4. WHEN _evaluate is called, THE G7Problem evaluator SHALL compute the objective and store it in column "f" as: f = x1² + x2² + x1·x2 − 14·x1 − 16·x2 + (x3−10)² + 4·(x4−5)² + (x5−3)² + 2·(x6−1)² + 5·x7² + 7·(x8−11)² + 2·(x9−10)² + (x10−7)² + 45 for each row.
5. WHEN _evaluate is called, THE G7Problem evaluator SHALL compute all eight constraints and store them in columns "g1" through "g8" exactly as: g1 = −105 + 4·x1 + 5·x2 − 3·x7 + 9·x8; g2 = 10·x1 − 8·x2 − 17·x7 + 2·x8; g3 = −8·x1 + 2·x2 + 5·x9 − 2·x10 − 12; g4 = 3·(x1−2)² + 4·(x2−3)² + 2·x3² − 7·x4 − 120; g5 = 5·x1² + 8·x2 + (x3−6)² − 2·x4 − 40; g6 = x1² + 2·(x2−2)² − 2·x1·x2 + 14·x5 − 6·x6; g7 = 0.5·(x1−8)² + 2·(x2−4)² + 3·x5² − x6 − 30; g8 = −3·x1 + 6·x2 + 12·(x9−8)² − 7·x10.
6. THE G7Problem evaluator SHALL define objectives as ["f"] and constraints as ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"].

### Requirement 5: Speed Reducer Design Problem Evaluator

**User Story:** As a researcher, I want a test evaluator implementing the speed reducer design problem, so that I can benchmark feasibility-discovery algorithms on a realistic 7D engineering problem with 11 constraints.

#### Acceptance Criteria

1. THE SpeedReducer evaluator SHALL inherit from TestEvaluator and reside in `src/standard_evaluator/evaluators/test/`.
2. WHEN _create_opt_problem is called, THE SpeedReducer evaluator SHALL define 7 continuous variables named x1 through x7 with bounds: x1∈[2.6, 3.6], x2∈[0.7, 0.8], x3∈[17, 28], x4∈[7.3, 8.3], x5∈[7.8, 8.3], x6∈[2.9, 3.9], x7∈[5.0, 5.5].
3. WHEN _create_opt_problem is called, THE SpeedReducer evaluator SHALL define 12 responses: "f" with bounds [-inf, inf], and "g1" through "g11" each with bounds [-inf, 0].
4. WHEN _evaluate is called, THE SpeedReducer evaluator SHALL compute the objective and store it in column "f" as: f = 0.7854·x1·x2²·(3.3333·x3² + 14.9334·x3 − 43.0934) − 1.508·x1·(x6² + x7²) + 7.4777·(x6³ + x7³) + 0.7854·(x4·x6² + x5·x7²) for each row.
5. WHEN _evaluate is called, THE SpeedReducer evaluator SHALL compute all eleven constraints and store them in columns "g1" through "g11" exactly as: g1 = 27/(x1·x2²·x3) − 1; g2 = 397.5/(x1·x2²·x3²) − 1; g3 = 1.93·x4³/(x2·x3·x6⁴) − 1; g4 = 1.93·x5³/(x2·x3·x7⁴) − 1; g5 = sqrt[(745·x4/(x2·x3))² + 16.9×10⁶]/(110·x6³) − 1; g6 = sqrt[(745·x5/(x2·x3))² + 157.5×10⁶]/(85·x7³) − 1; g7 = x2·x3/40 − 1; g8 = 5·x2/x1 − 1; g9 = x1/(12·x2) − 1; g10 = (1.5·x6 + 1.9)/x4 − 1; g11 = (1.1·x7 + 1.9)/x5 − 1.
6. THE SpeedReducer evaluator SHALL define objectives as ["f"] and constraints as ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11"].
7. WHEN _evaluate is called with the known feasible point x1=3.5, x2=0.7, x3=17, x4=7.3, x5=7.8, x6=3.35, x7=5.287, THE SpeedReducer evaluator SHALL produce g1 through g11 all ≤ 0.

### Requirement 6: Library Integration

**User Story:** As a developer, I want all new benchmark evaluators to be automatically discoverable through the existing SE test evaluator registry, so that they integrate seamlessly with existing tooling and test infrastructure.

#### Acceptance Criteria

1. WHEN the SE_Library test evaluators package is imported, THE SE_Library SHALL automatically discover and expose all 5 new benchmark evaluator classes (SmallCircleFeasibleRegion, DisconnectedFeasibleRegions, G6Problem, G7Problem, SpeedReducer) via the existing `__init__.py` auto-discovery mechanism.
2. THE SE_Library `__init__.py` expected class count SHALL be updated from 38 to 43 to reflect the addition of the 5 new evaluator classes.
3. WHEN any new benchmark evaluator is instantiated and called with a DataFrame containing columns matching the evaluator's input variable names and at least 1 row, THE evaluator SHALL return the DataFrame with all response columns populated with values that are not NaN and not ±infinity.
4. IF a new benchmark evaluator receives input sites outside the defined variable bounds, THEN THE evaluator SHALL still compute and return response values without raising an exception (bounds enforcement is the caller's responsibility).

### Requirement 7: Test Coverage

**User Story:** As a developer, I want unit tests for each new benchmark evaluator, so that correctness of the mathematical implementations can be verified and maintained over time.

#### Acceptance Criteria

1. THE test suite SHALL include at least one test per new evaluator (SmallCircleFeasibleRegion, DisconnectedFeasibleRegions, G6Problem, G7Problem, SpeedReducer) verifying that instantiation succeeds without raising an exception and produces an object that is an instance of TestEvaluator.
2. THE test suite SHALL include at least one test per new evaluator verifying that evaluation at a known feasible point (analytically derived from the problem geometry) produces g_j ≤ 0 for all constraints, where floating-point equality is asserted using a tolerance of 1e-8.
3. THE test suite SHALL include at least one test per new evaluator verifying that evaluation at a known infeasible point (analytically derived from the problem geometry) produces g_j > 0 for at least one constraint.
4. WHEN the DisconnectedFeasibleRegions evaluator is tested, THE test suite SHALL verify both the nonsmooth (smooth=False) and the smooth (smooth=True) formulations by evaluating identical input points and asserting that each formulation returns finite numerical values and that a point at an island center yields g1 ≤ 0.
5. WHEN the SmallCircleFeasibleRegion evaluator is tested with a point at the circle center (a, b), THE test suite SHALL verify that g1 = −r² within a tolerance of 1e-8 (with default parameters: expected value −0.0225).
6. THE test suite SHALL reside in the `tests/evaluators/test/` directory, with one test module per new evaluator following the existing naming convention `test_<evaluator_module_name>.py`.
