"""Property test for notebook structure invariant.

**Validates: Requirements 8.1, 8.2, 8.7**

For any demo notebook produced by the feature-documentation spec, the notebook
SHALL begin with a markdown cell containing a level-1 heading, followed by an
overview paragraph, and SHALL contain no two consecutive code cells without an
intervening markdown cell.
"""

import json
from pathlib import Path

import pytest

# Path to the demos directory
_DEMOS_DIR = Path(__file__).parent.parent / "docs" / "source" / "demos"

# The 6 new demo notebooks produced by this feature
_NEW_NOTEBOOKS = [
    "evaluator_hierarchy.ipynb",
    "surrogate_models.ipynb",
    "openmdao_component.ipynb",
    "array_variables.ipynb",
    "benchmark_problems.ipynb",
    "evaluator_interface.ipynb",
]


def _load_notebook(filename: str) -> dict:
    """Load and parse a notebook JSON file."""
    with open(_DEMOS_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("notebook_name", _NEW_NOTEBOOKS)
def test_first_cell_is_markdown_with_level1_heading(notebook_name):
    """Each notebook starts with a markdown cell containing a level-1 heading.

    **Validates: Requirements 8.1**
    """
    nb = _load_notebook(notebook_name)
    cells = nb["cells"]

    assert len(cells) > 0, f"{notebook_name} has no cells"

    first_cell = cells[0]
    assert first_cell["cell_type"] == "markdown", (
        f"{notebook_name}: first cell is '{first_cell['cell_type']}', expected 'markdown'"
    )

    source = "".join(first_cell["source"]) if isinstance(first_cell["source"], list) else first_cell["source"]
    lines = source.split("\n")
    assert any(line.startswith("# ") for line in lines), (
        f"{notebook_name}: first markdown cell has no level-1 heading (no line starting with '# ')"
    )


@pytest.mark.parametrize("notebook_name", _NEW_NOTEBOOKS)
def test_first_cell_has_overview_paragraph(notebook_name):
    """Each notebook has an overview paragraph in the first markdown cell.

    **Validates: Requirements 8.2**
    """
    nb = _load_notebook(notebook_name)
    first_cell = nb["cells"][0]

    source = "".join(first_cell["source"]) if isinstance(first_cell["source"], list) else first_cell["source"]
    lines = source.strip().split("\n")

    # There must be text beyond the heading line
    non_heading_lines = [line for line in lines if line.strip() and not line.startswith("# ")]
    assert len(non_heading_lines) > 0, (
        f"{notebook_name}: first cell has no overview paragraph beyond the heading"
    )


@pytest.mark.parametrize("notebook_name", _NEW_NOTEBOOKS)
def test_no_consecutive_code_cells(notebook_name):
    """No two adjacent cells are both code cells.

    **Validates: Requirements 8.7**
    """
    nb = _load_notebook(notebook_name)
    cells = nb["cells"]

    for i in range(len(cells) - 1):
        if cells[i]["cell_type"] == "code" and cells[i + 1]["cell_type"] == "code":
            pytest.fail(
                f"{notebook_name}: consecutive code cells at positions {i} and {i + 1}"
            )
