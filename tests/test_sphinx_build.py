"""Integration test for Sphinx build completeness.

Feature: feature-documentation
Property 2: Build Completeness

**Validates: Requirements 7.3, 9.7**

Verifies that the Sphinx documentation build completes successfully and that
the new demo notebooks and API reference pages produce HTML output. The build
runs without the -W flag since pre-existing orphan notebooks in docs/source/demos/
(e.g. design_space_exploration.ipynb, group_demo.md) produce toctree warnings
that are unrelated to our changes.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Skip all tests in this module if sphinx is not installed
sphinx = pytest.importorskip("sphinx", reason="Sphinx is required for documentation build tests")

# Project root directory
_PROJECT_ROOT = Path(__file__).parent.parent

# Docs source and build directories
_DOCS_SOURCE = _PROJECT_ROOT / "docs" / "source"
_DOCS_BUILD = _PROJECT_ROOT / "docs" / "build" / "test_html"

# The 6 new demo notebooks that should generate HTML pages
_NEW_DEMO_PAGES = [
    "evaluator_hierarchy",
    "surrogate_models",
    "openmdao_component",
    "array_variables",
    "benchmark_problems",
    "evaluator_interface",
]

# API reference pages that should be generated
_REFERENCE_PAGES = [
    "evaluators",
    "surrogate_models",
    "components",
    "data_models",
    "decorators",
]


@pytest.mark.integration
@pytest.mark.slow
class TestSphinxBuildCompleteness:
    """Integration tests verifying Sphinx documentation builds without errors.

    **Validates: Requirements 7.3, 9.7**
    """

    def test_sphinx_build_succeeds(self):
        """Sphinx build completes successfully (return code 0).

        Runs sphinx-build without -W so that pre-existing orphan-document
        warnings from legacy notebooks do not cause a failure. We only
        verify that the build itself succeeds.

        **Validates: Requirements 7.3, 9.7**
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                str(_DOCS_SOURCE),
                str(_DOCS_BUILD),
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=600,  # 10-minute timeout for full build with notebook execution
            check=False,  # We check returncode manually for better error messages
        )

        assert result.returncode == 0, (
            f"Sphinx build failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}\n"
            f"STDERR:\n{result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr}"
        )

    def test_demo_html_files_generated(self):
        """Verify HTML files are generated for all 6 new demo notebooks.

        **Validates: Requirements 7.3**
        """
        # Only check if build output exists (test_sphinx_build_succeeds
        # must run first to produce output)
        demos_build_dir = _DOCS_BUILD / "demos"
        if not demos_build_dir.exists():
            pytest.skip(
                "Build output not found. Run test_sphinx_build_succeeds first."
            )

        missing_pages = []
        for page_name in _NEW_DEMO_PAGES:
            html_file = demos_build_dir / f"{page_name}.html"
            if not html_file.exists():
                missing_pages.append(page_name)

        assert not missing_pages, (
            f"Missing HTML files for demo pages: {missing_pages}. "
            f"Expected files in {demos_build_dir}"
        )

    def test_reference_html_files_generated(self):
        """Verify HTML files are generated for all API reference pages.

        **Validates: Requirements 9.7**
        """
        # Only check if build output exists
        reference_build_dir = _DOCS_BUILD / "reference"
        if not reference_build_dir.exists():
            pytest.skip(
                "Build output not found. Run test_sphinx_build_succeeds first."
            )

        missing_pages = []
        for page_name in _REFERENCE_PAGES:
            html_file = reference_build_dir / f"{page_name}.html"
            if not html_file.exists():
                missing_pages.append(page_name)

        assert not missing_pages, (
            f"Missing HTML files for reference pages: {missing_pages}. "
            f"Expected files in {reference_build_dir}"
        )
