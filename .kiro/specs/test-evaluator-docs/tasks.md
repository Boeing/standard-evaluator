# Implementation Plan: Test Evaluator Documentation

## Overview

This plan adds a `:print-test-info:` directive option to the existing `format-class.py` Sphinx extension, creates a categorized reference page for all 43 test evaluators, and integrates it into the documentation navigation. The extension renders mathematical descriptions, citations, and structured metadata (test_info dictionary) for each evaluator.

## Tasks

- [x] 1. Extend the format-class.py Sphinx extension
  - [x] 1.1 Register `:print-test-info:` boolean option in `FormatClass.option_spec`
    - Add `option_spec['print-test-info'] = bool_option` alongside existing `print-options` and `known-solution` options
    - _Requirements: 1.1, 5.1-5.4_
  - [x] 1.2 Add dispatch in `add_content()` — call `_print_test_info()` between print-options and known-solution
    - Insert `if 'print-test-info' in self.options: self._print_test_info()` after the `_print_options()` call and before the `_print_known_solution()` call
    - This preserves ordering: docstring → print-options → print-test-info → known-solution → autosummary
    - _Requirements: 5.5_
  - [x] 1.3 Implement `_print_test_info()` method
    - Try `inst = obj()`, catch `TypeError` and return silently (handles abstract bases)
    - Check `hasattr(inst, 'opt_problem')`, skip if missing
    - If `inst.opt_problem.description` is non-None and non-empty, call `self._render_description(description.strip(), source_name)`
    - If `inst.opt_problem.cite` is non-None and non-empty, call `self._render_citation(cite.strip(), source_name)`
    - Call `self._render_test_info_table(inst.test_info, source_name)`
    - _Requirements: 1.2, 2.1, 2.2, 2.3, 2.4, 2.6_
  - [x] 1.4 Implement `_render_description(self, description, source_name)` method
    - Add separator `|` and title "Problem Description" using `.. rst-class:: title` pattern
    - If description starts with `$$` AND ends with `$$`: strip delimiters, render inside `.. math::` directive with 3-space indentation for each line
    - If description starts with `$$` but does NOT end with `$$`: treat as plain text (non-math path)
    - If description is plain text (no `$$` delimiters): render as paragraph lines
    - _Requirements: 1.3, 1.4, 1.5, 2.5, 6.1, 6.2_
  - [x] 1.5 Implement `_render_citation(self, cite, source_name)` method
    - Render as `.. admonition:: Reference` with 3-space indented content lines
    - Handle multi-line citations by stripping and indenting each line
    - _Requirements: 1.6, 2.3, 6.1, 6.3_
  - [x] 1.6 Implement `_render_test_info_table(self, test_info, source_name)` method
    - Add title "Problem Metadata" using `.. rst-class:: title` pattern
    - Render RST field list with labels for all 9 keys: test_goal → "Test Goal", problem_type → "Problem Type", n_vars → "Variables", n_continuous → "Continuous Variables", n_discrete → "Discrete Variables", n_constraints → "Constraints", n_equality_constraints → "Equality Constraints", n_inequality_constraints → "Inequality Constraints", bounded_variables → "Bounded Variables"
    - Boolean values → "Yes"/"No"
    - `test_goal` and `problem_type` values → replace underscores with spaces, title-case
    - Integer values → plain number strings
    - _Requirements: 1.7, 1.8, 1.9, 6.1, 6.4_

- [x] 2. Create test evaluator reference pages (per-category)
  - [x] 2.1 Discover evaluator categories by running `.venv\Scripts\python.exe -c "..."` to get each evaluator's test_goal
    - Import all evaluators from `standard_evaluator.evaluators.test`
    - Instantiate each, read `test_info['test_goal']`, group by category
    - Identify the abstract bases to exclude: `BoreholeMultiFiBase`, `ExponentialMultiFiBase`, `ForresterMultiFiBase`, `SimpleMultiFiBase`
    - _Requirements: 3.5, 3.6_
  - [x] 2.2 Create `docs/source/reference/test_evaluators/index.rst` — landing page with overview and toctree
    - Title: "Test Evaluators"
    - Brief intro explaining these are benchmark test functions with known solutions
    - Include `TestEvaluator` base class at top with `.. autoclass:: standard_evaluator.evaluators.test_evaluator.TestEvaluator` and `:show-inheritance:`
    - Toctree linking to: `optimization`, `multi_objective`, `feasibility`
    - _Requirements: 3.1, 3.2_
  - [x] 2.3 Create `docs/source/reference/test_evaluators/optimization.rst` — single-objective optimization evaluators
    - Title: "Optimization Test Evaluators"
    - Each evaluator: `.. autoclass:: standard_evaluator.evaluators.test.ClassName` with `:print-test-info:`, `:known-solution:`, `:show-inheritance:`
    - Only include evaluators where `test_info['test_goal'] == 'optimization'`
    - _Requirements: 3.3, 3.4, 3.6_
  - [x] 2.4 Create `docs/source/reference/test_evaluators/multi_objective.rst` — multi-objective optimization evaluators
    - Title: "Multiple Objective Optimization Test Evaluators"
    - Each evaluator: `.. autoclass:: standard_evaluator.evaluators.test.ClassName` with `:print-test-info:`, `:known-solution:`, `:show-inheritance:`
    - Only include evaluators where `test_info['test_goal'] == 'multiple_objective_optimization'`
    - _Requirements: 3.3, 3.4, 3.6_
  - [x] 2.5 Create `docs/source/reference/test_evaluators/feasibility.rst` — feasibility-discovery evaluators
    - Title: "Feasibility Discovery Test Evaluators"
    - Each evaluator: `.. autoclass:: standard_evaluator.evaluators.test.ClassName` with `:print-test-info:`, `:known-solution:`, `:show-inheritance:`
    - Only include evaluators where `test_info['test_goal'] == 'feasibility'`
    - _Requirements: 3.3, 3.4, 3.6_

- [x] 3. Update reference/index.rst
  - [x] 3.1 Add `test_evaluators/index` entry to the toctree in `docs/source/reference/index.rst`
    - Place alongside existing entries (evaluators, surrogate_models, components, data_models, decorators)
    - _Requirements: 4.1, 4.2_

- [x] 4. Verify Sphinx build
  - [x] 4.1 Run `.venv\Scripts\python.exe -m sphinx docs/source docs/build -b html` targeting the test_evaluators page
    - Verify no build errors or warnings attributable to generated RST content
    - Fix any issues that arise (malformed RST, import errors, rendering problems)
    - _Requirements: 6.5_

## Task Dependency Graph

```json
{
  "waves": [
    {
      "wave": 0,
      "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "2.1"],
      "description": "Extend Sphinx extension and discover evaluator categories"
    },
    {
      "wave": 1,
      "tasks": ["2.2", "2.3", "2.4", "2.5", "3.1"],
      "description": "Create RST pages and update index (depends on category discovery)"
    },
    {
      "wave": 2,
      "tasks": ["4.1"],
      "description": "Verify Sphinx build (depends on all above)"
    }
  ]
}
```
