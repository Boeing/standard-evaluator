"""Integration test for notebook execution correctness.

**Validates: Requirements 1.7, 2.7, 3.5, 4.6, 5.5, 6.6**

Execute each of the 6 new demo notebooks in a fresh kernel using nbconvert's
ExecutePreprocessor and verify zero exceptions are raised during execution.

Notebooks that require optional dependencies (e.g. smt) are skipped when those
dependencies are not installed.
"""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# Path to the demos directory relative to the project root
DEMOS_DIR = Path(__file__).parent.parent / "docs" / "source" / "demos"

# Notebooks that only need base dependencies
BASE_NOTEBOOKS = [
    "evaluator_hierarchy.ipynb",
    "openmdao_component.ipynb",
    "array_variables.ipynb",
    "benchmark_problems.ipynb",
    "evaluator_interface.ipynb",
]

# Notebooks requiring optional dependencies
SMT_NOTEBOOKS = [
    "surrogate_models.ipynb",
]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", BASE_NOTEBOOKS)
def test_notebook_executes_without_errors(notebook_name: str) -> None:
    """Execute a base-dependency demo notebook and verify no exceptions.

    **Validates: Requirements 1.7, 3.5, 4.6, 5.5, 6.6**

    Each notebook is executed sequentially in a fresh Python 3 kernel. The test
    fails if any cell raises a CellExecutionError during execution.
    """
    notebook_path = DEMOS_DIR / notebook_name
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    nb = nbformat.read(notebook_path, as_version=4)

    ep = ExecutePreprocessor(
        timeout=60,
        kernel_name="python3",
    )

    # Execute the notebook in the demos directory so relative file references work
    ep.preprocess(nb, {"metadata": {"path": str(DEMOS_DIR)}})


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", SMT_NOTEBOOKS)
def test_smt_notebook_executes_without_errors(notebook_name: str) -> None:
    """Execute an SMT-dependency demo notebook and verify no exceptions.

    **Validates: Requirements 2.7**

    Skipped when the smt package is not installed. Each notebook is executed
    sequentially in a fresh Python 3 kernel.
    """
    pytest.importorskip("smt", reason="smt not installed, skipping surrogate model notebook test")

    notebook_path = DEMOS_DIR / notebook_name
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    nb = nbformat.read(notebook_path, as_version=4)

    ep = ExecutePreprocessor(
        timeout=60,
        kernel_name="python3",
    )

    # Execute the notebook in the demos directory so relative file references work
    ep.preprocess(nb, {"metadata": {"path": str(DEMOS_DIR)}})
