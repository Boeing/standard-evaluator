# Implementation Plan: Fix Test Evaluator Documentation Rendering

## Overview

This plan fixes the `_render_description` method in the Sphinx extension to correctly handle mixed text/math descriptions, then updates ~20+ source description strings and their corresponding test assertion files to use `$$` delimiters only around actual LaTeX math content.

## Tasks

- [x] 1. Rewrite `_render_description` in `format-class.py`
  - Replace the current `startswith('$$') and endswith('$$')` heuristic with a `split('$$')` approach
  - Odd-indexed segments (after split) are math → emit inside `.. math::` directive with 3-space indentation
  - Even-indexed segments are text → emit as plain RST paragraph lines
  - Skip empty/whitespace-only segments
  - Always emit "Problem Description" heading before content
  - Always emit trailing blank line after all segments
  - File: `docs/source/_ext/format-class.py`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 5.1, 5.2, 6.1, 6.2_

- [x] 2. Fix pure-text description files (remove `$$` wrappers entirely)
  - [x] 2.1 Fix `borehole_multi_fi_base.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update `tests/evaluators/test/test_borehole_multi_fi_base.py` expected_description to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.2 Fix `cantilevered_beam_continuous.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.3 Fix `cantilevered_beam_with_fixed_variables.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.4 Fix `disconnect.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.5 Fix `exponential_multi_fi_base.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.6 Fix `forrester_multi_fi_base.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.7 Fix `simple_multi_fi_base.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.8 Fix `two_bar_truss.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_
  - [x] 2.9 Fix `wrkbk_prb_1.py` and update test
    - Remove `$$` delimiters wrapping plain text description
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 3.1_

- [x] 3. Fix text-with-RST-directives files (remove `$$`, keep RST content)
  - [x] 3.1 Fix `extended_rosenbrock.py` and update test
    - Remove `$$` delimiters that wrap RST directives (e.g., `.. math::`, `.. note::`)
    - Keep the RST directive content intact outside `$$` blocks
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 2.4, 3.1_
  - [x] 3.2 Fix `powell_singular.py` and update test
    - Remove `$$` delimiters that wrap RST directives
    - Keep the RST directive content intact outside `$$` blocks
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 2.4, 3.1_
  - [x] 3.3 Fix `trigonometric.py` and update test
    - Remove `$$` delimiters that wrap RST directives
    - Keep the RST directive content intact outside `$$` blocks
    - Update corresponding test assertion file to match
    - _Requirements: 2.1, 2.4, 3.1_

- [x] 4. Checkpoint - Pure text and RST directive files
  - Run `.venv\Scripts\python.exe -m pytest tests/evaluators/test/ -v` to verify all updated tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Fix mixed math+text description files (fix `$$` boundaries)
  - [x] 5.1 Fix `constrained_betts.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.2 Fix `g6_problem.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.3 Fix `hs38.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.4 Fix `hs47.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.5 Fix `hs100.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.6 Fix `hs118.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.7 Fix `helical_valley.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.8 Fix `optlib_test.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.9 Fix `tp37.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.10 Fix `das_knee.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_
  - [x] 5.11 Fix `das_truss.py` and update test
    - Ensure `$$` delimiters wrap only LaTeX math portions; plain text between math blocks has no `$$`
    - Update corresponding test assertion file to match
    - _Requirements: 2.2, 2.3, 3.1_

- [x] 6. Final verification
  - Run `.venv\Scripts\python.exe -m pytest tests/evaluators/test/ -v` to confirm all test assertions pass
  - Run `.venv\Scripts\python.exe -m sphinx docs/source docs/build -b html` to confirm Sphinx builds without errors/warnings on test evaluator pages
  - Ensure all tests pass and Sphinx build is clean, ask the user if questions arise.
  - _Requirements: 4.1, 4.2, 5.1, 5.2, 5.3_

## Task Dependency Graph

```json
{
  "waves": [
    { "tasks": ["1"] },
    { "tasks": ["2", "3"] },
    { "tasks": ["4"] },
    { "tasks": ["5"] },
    { "tasks": ["6"] }
  ]
}
```

## Notes

- Source files are in `src/standard_evaluator/evaluators/test/`
- Test assertion files are in `tests/evaluators/test/`
- Each source file fix must be paired with its corresponding test file update so tests stay green
- The `_render_description` rewrite (task 1) must be done first since the new parser expects properly delimited descriptions
- Python executable: `.venv\Scripts\python.exe`
- Run tests: `.venv\Scripts\python.exe -m pytest tests/evaluators/test/ -v`
- Run sphinx: `.venv\Scripts\python.exe -m sphinx docs/source docs/build -b html`
