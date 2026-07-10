# Requirements Document

## Introduction

The Standard Evaluator library contains 43 test evaluator classes implementing benchmark optimization problems. Each evaluator has rich metadata (mathematical description, citation, test_info dictionary with problem characteristics) that is currently not exposed in the documentation. This feature extends the existing `format-class.py` Sphinx extension with a `:print-test-info:` directive option and creates a new categorized reference page, making all test evaluator metadata browsable in the built documentation site.

## Glossary

- **FormatClass Extension**: The custom Sphinx autodoc extension (`docs/source/_ext/format-class.py`) that adds `:print-options:` and `:known-solution:` directive options to `autoclass`.
- **test_info**: A dictionary on each test evaluator instance containing structured metadata: `test_goal`, `problem_type`, `n_vars`, `n_continuous`, `n_discrete`, `n_constraints`, `n_equality_constraints`, `n_inequality_constraints`, `bounded_variables`.
- **opt_problem**: A property on test evaluators providing an `OptProblem` instance with `description` (mathematical formula or text) and `cite` (academic citation).
- **test_goal**: A string field in `test_info` categorizing the evaluator's purpose: `"optimization"`, `"multiple_objective_optimization"`, or `"feasibility"`.
- **Abstract Base Evaluator**: A multi-fidelity base class (e.g., `BoreholeMultiFiBase`) that cannot be instantiated without constructor arguments and should be excluded from documentation rendering.

## Requirements

### Requirement 1: Sphinx Extension — `:print-test-info:` Directive Option

**User Story:** As a documentation maintainer, I want a `:print-test-info:` directive option for the `autoclass` Sphinx directive so that test evaluator metadata (description, citation, and problem characteristics) is rendered automatically from the source code.

#### Acceptance Criteria

1. The `format-class.py` extension SHALL register a `:print-test-info:` boolean option in the `FormatClass.option_spec` dictionary.
2. WHEN the `:print-test-info:` option is present on an `autoclass` directive, THEN the extension SHALL attempt to instantiate the documented class with no arguments.
3. WHEN instantiation succeeds AND `opt_problem.description` is non-None and non-empty, THEN the extension SHALL render a "Problem Description" section in the RST output.
4. WHEN the description contains `$$`-delimited LaTeX content (starts with `$$` and ends with `$$`), THEN the extension SHALL render the content inside a `.. math::` directive with the `$$` delimiters removed.
5. WHEN the description does not contain `$$` delimiters, THEN the extension SHALL render it as plain RST paragraph text.
6. WHEN instantiation succeeds AND `opt_problem.cite` is non-None and non-empty, THEN the extension SHALL render a `.. admonition:: Reference` block containing the citation text.
7. WHEN instantiation succeeds, THEN the extension SHALL render a "Problem Metadata" section containing a field list with all 9 `test_info` dictionary keys: `test_goal`, `problem_type`, `n_vars`, `n_continuous`, `n_discrete`, `n_constraints`, `n_equality_constraints`, `n_inequality_constraints`, `bounded_variables`.
8. Boolean values in `test_info` SHALL be rendered as "Yes" or "No".
9. The `test_goal` and `problem_type` values SHALL be rendered with underscores replaced by spaces and title-cased (e.g., "multiple_objective_optimization" → "Multiple Objective Optimization").

Cross-references: Design Correctness Properties 3, 4

### Requirement 2: Error Handling and Graceful Failures

**User Story:** As a documentation maintainer, I want the `:print-test-info:` directive to handle edge cases gracefully so that the Sphinx build never fails due to abstract classes, missing attributes, or None fields.

#### Acceptance Criteria

1. IF the documented class raises `TypeError` on instantiation (e.g., abstract base classes or classes requiring constructor arguments), THEN the extension SHALL skip all test-info rendering silently without raising an exception.
2. IF `opt_problem.description` is None or an empty string, THEN the extension SHALL NOT render a "Problem Description" section.
3. IF `opt_problem.cite` is None or an empty string, THEN the extension SHALL NOT render a "Reference" admonition section.
4. IF the documented class does not have an `opt_problem` attribute, THEN the extension SHALL skip test-info rendering without error.
5. IF the description starts with `$$` but does NOT end with `$$`, THEN the extension SHALL treat the description as plain text (non-math rendering path).
6. WHEN a `TypeError` is caught during instantiation, THEN all other documentation content (docstring, method summary, other directive options) SHALL continue to render normally.

Cross-references: Design Correctness Properties 1, 2

### Requirement 3: Test Evaluators RST Reference Pages

**User Story:** As a library user, I want dedicated reference pages listing test evaluators categorized by their test goal so that I can quickly find an appropriate test function for my optimization problem without scrolling through an excessively long page.

#### Acceptance Criteria

1. A new directory `docs/source/reference/test_evaluators/` SHALL be created containing per-category RST pages and a landing page.
2. The landing page `docs/source/reference/test_evaluators/index.rst` SHALL contain a toctree linking to the three category pages: `optimization`, `multi_objective`, and `feasibility`.
3. The category pages SHALL be: `optimization.rst` (single-objective optimization evaluators), `multi_objective.rst` (multiple-objective optimization evaluators), and `feasibility.rst` (feasibility-discovery evaluators).
4. Each evaluator entry SHALL use the `autoclass` directive with `:print-test-info:`, `:known-solution:`, and `:show-inheritance:` options.
5. Abstract base classes that cannot be instantiated (e.g., `BoreholeMultiFiBase`, `ExponentialMultiFiBase`, `ForresterMultiFiBase`, `SimpleMultiFiBase`) SHALL be excluded from all pages.
6. Each category page SHALL include all concrete (non-abstract) test evaluator classes whose `test_info['test_goal']` matches the category.

Cross-references: Design Correctness Property 4

### Requirement 4: Reference Index Integration

**User Story:** As a library user, I want the test evaluators section to appear in the API reference navigation so that I can discover it through the standard documentation structure.

#### Acceptance Criteria

1. The file `docs/source/reference/index.rst` SHALL include a `test_evaluators/index` entry in its toctree directive.
2. The `test_evaluators/index` entry SHALL appear alongside the existing entries (evaluators, surrogate_models, components, data_models, decorators).

### Requirement 5: Backward Compatibility

**User Story:** As a documentation maintainer, I want the new `:print-test-info:` option to coexist with existing `:print-options:` and `:known-solution:` directives without interference so that existing documentation continues to build correctly.

#### Acceptance Criteria

1. WHEN `:print-options:` is used on a class directive (without `:print-test-info:`), THEN the rendered output SHALL be identical to the current behavior.
2. WHEN `:known-solution:` is used on a class directive (without `:print-test-info:`), THEN the rendered output SHALL be identical to the current behavior.
3. WHEN `:print-test-info:` is used alongside `:known-solution:` on the same directive, THEN both sections SHALL render their respective output without interfering with each other.
4. WHEN `:print-test-info:` is used alongside `:print-options:` on the same directive, THEN both sections SHALL render their respective output without interfering with each other.
5. The extension SHALL maintain the existing rendering order: class docstring → print-options content → print-test-info content → known-solution content → autosummary tables.

Cross-references: Design Correctness Property 5

### Requirement 6: RST Output Validity

**User Story:** As a documentation maintainer, I want all generated RST to be valid so that the Sphinx build completes without errors or warnings from the test evaluators page.

#### Acceptance Criteria

1. All RST content generated by `:print-test-info:` SHALL be syntactically valid reStructuredText.
2. Math directives SHALL be properly indented (3-space indent for content lines under `.. math::`).
3. Admonition content SHALL be properly indented (3-space indent for content lines under `.. admonition:: Reference`).
4. Field list entries SHALL use the format `:Label: value` with no leading whitespace.
5. WHEN the Sphinx build is run with the test_evaluators.rst page included, THEN there SHALL be no build errors attributable to the generated RST content.

Cross-references: Design Correctness Properties 3, 4
