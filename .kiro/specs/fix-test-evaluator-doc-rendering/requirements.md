# Requirements Document

## Introduction

This document specifies the requirements for fixing the documentation rendering of test evaluator classes in the Standard Evaluator project. The custom Sphinx extension (`format-class.py`) currently fails to render mixed text/math content in `opt_problem.description` fields. The fix involves rewriting the `_render_description` method to correctly parse `$$`-delimited segments, updating ~24 source description strings to use `$$` only around actual LaTeX math, and updating corresponding test assertion files to match.

## Glossary

- **Renderer**: The `_render_description` method within the `FormatClass` Sphinx extension (`format-class.py`) that converts `opt_problem.description` strings into RST content.
- **Description_String**: The `opt_problem.description` field on a test evaluator instance, containing plain text, LaTeX math, or a mix of both.
- **Math_Segment**: A substring enclosed between a pair of `$$` delimiters that contains valid LaTeX math content.
- **Text_Segment**: A substring outside `$$` delimiter pairs that contains plain RST text (may include RST directives, Unicode characters, or plain English).
- **Source_File**: A Python file in `src/standard_evaluator/evaluators/test/` that defines a test evaluator class and sets `opt_problem.description`.
- **Test_Assertion_File**: A Python file in `tests/evaluators/test/` that asserts the value of `opt_problem.description` for a corresponding source file.
- **Sphinx_Build**: The documentation generation process using `sphinx-build` that produces HTML output from RST content.
- **Non_Test_Evaluator_Class**: Any evaluator class outside the `src/standard_evaluator/evaluators/test/` directory that may use the `:print-test-info:` or `:print-options:` directive.

## Requirements

### Requirement 1: Split-based description parsing

**User Story:** As a documentation maintainer, I want the Renderer to correctly parse descriptions containing mixed text and math content, so that both plain text and LaTeX formulas render correctly in the documentation.

#### Acceptance Criteria

1. WHEN a Description_String contains one or more `$$` delimiter pairs, THE Renderer SHALL split the string on `$$` boundaries and classify odd-indexed segments (after split) as Math_Segments and even-indexed segments as Text_Segments.
2. WHEN a Description_String contains no `$$` delimiters, THE Renderer SHALL render the entire content as a single Text_Segment (plain RST paragraph).
3. WHEN processing a Math_Segment, THE Renderer SHALL emit it inside a `.. math::` RST directive with each line indented by three spaces.
4. WHEN processing a Text_Segment, THE Renderer SHALL emit it as plain RST paragraph lines without additional indentation.
5. WHEN a segment (text or math) is empty or contains only whitespace after stripping, THE Renderer SHALL skip that segment and produce no output for it.
6. THE Renderer SHALL always emit the "Problem Description" heading before any segment content.
7. THE Renderer SHALL always emit a trailing blank line after all segments are rendered.

### Requirement 2: Source description string correctness

**User Story:** As a documentation maintainer, I want each Source_File's description to use `$$` delimiters only around actual LaTeX math content, so that the Renderer can correctly distinguish math from text.

#### Acceptance Criteria

1. WHEN a Source_File description contains only plain text (no LaTeX commands such as `\min`, `\frac`, `\begin`, `\quad`, etc.), THE Source_File SHALL NOT wrap that content in `$$` delimiters.
2. WHEN a Source_File description contains actual LaTeX math formulas, THE Source_File SHALL wrap only the math portions in `$$` delimiters.
3. WHEN a Source_File description contains both plain text and LaTeX math, THE Source_File SHALL use separate `$$` pairs around each math portion with plain text between them.
4. THE Source_File descriptions SHALL NOT contain RST directives (such as `.. math::` or `.. note::`) inside `$$` delimited blocks.

### Requirement 3: Test assertion file consistency

**User Story:** As a developer, I want the Test_Assertion_Files to match the updated Source_File descriptions, so that the test suite passes after the description fixes.

#### Acceptance Criteria

1. WHEN a Source_File description is modified, THE corresponding Test_Assertion_File SHALL update its `expected_description` string to match the new Source_File description exactly.
2. THE Test_Assertion_File assertions SHALL continue to compare stripped strings (using `.strip()` on both sides of the comparison).

### Requirement 4: Sphinx build success

**User Story:** As a documentation maintainer, I want the Sphinx_Build to complete without errors or warnings related to test evaluator pages, so that the documentation site can be published reliably.

#### Acceptance Criteria

1. WHEN the Sphinx_Build processes test evaluator class pages, THE Sphinx_Build SHALL complete without emitting errors related to math rendering or RST parsing on those pages.
2. WHEN the Sphinx_Build processes test evaluator class pages, THE Sphinx_Build SHALL complete without emitting warnings related to malformed math content or unexpected RST structure on those pages.

### Requirement 5: Backward compatibility

**User Story:** As a developer, I want the Renderer changes to not affect the rendering of Non_Test_Evaluator_Classes or other Sphinx directives, so that existing documentation remains correct.

#### Acceptance Criteria

1. THE Renderer SHALL produce identical RST output for descriptions that contain no `$$` delimiters as the previous implementation did.
2. THE Renderer SHALL produce identical RST output for descriptions that are a single pure math block (`$$..$$` with no intermediate text) as the previous implementation did.
3. WHEN the `:print-options:` or `:known-solution:` directives are used on Non_Test_Evaluator_Classes, THE FormatClass extension SHALL produce identical output as before the change.

### Requirement 6: Rendering correctness for known patterns

**User Story:** As an end user reading the documentation, I want each test evaluator page to display math formulas as rendered LaTeX and plain text as readable prose, so that the problem descriptions are understandable.

#### Acceptance Criteria

1. WHEN a Description_String starts with `$$` (math-first pattern), THE Renderer SHALL render the first non-empty segment as a math block.
2. WHEN a Description_String contains multiple `$$` pairs with text between them (mixed pattern), THE Renderer SHALL alternate between math blocks and text paragraphs in the order they appear.
3. WHEN a Description_String contains Unicode math symbols (such as `∈`, `≤`, `×`) outside `$$` delimiters, THE Renderer SHALL render them as plain text characters (not as LaTeX).
