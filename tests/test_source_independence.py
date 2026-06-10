"""Property 7: Source code independence from Boeing references.

Scans all .py files under src/standard_evaluator/ and asserts:
- Zero occurrences of the string 'boeing_standard_evaluator'
- Zero occurrences of Boeing proprietary notice text

This is an exhaustive scan-based test, not Hypothesis-based.

**Validates: Requirements 16.3, 16.4**
"""

import os
from pathlib import Path

import pytest


# Root of the source package to scan
SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "standard_evaluator"

# Patterns that must NOT appear in any source file
FORBIDDEN_IMPORT_STRING = "boeing_standard_evaluator"
FORBIDDEN_PROPRIETARY_PATTERNS = [
    "Boeing Proprietary",
    "Boeing Confidential",
    "All Rights Reserved by Boeing",
]


def _collect_python_files(root: Path) -> list[Path]:
    """Walk all .py files under the given root directory."""
    py_files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                py_files.append(Path(dirpath) / filename)
    return py_files


def _get_all_source_files() -> list[Path]:
    """Get all Python source files under src/standard_evaluator/."""
    assert SRC_DIR.exists(), f"Source directory not found: {SRC_DIR}"
    files = _collect_python_files(SRC_DIR)
    assert len(files) > 0, f"No .py files found under {SRC_DIR}"
    return files


class TestSourceIndependenceFromBoeingReferences:
    """Property 7: Source code independence from Boeing references.

    For any Python file under src/standard_evaluator/, the file content SHALL contain
    zero occurrences of 'boeing_standard_evaluator' as an import path or module reference,
    and zero occurrences of Boeing proprietary notice text.

    **Validates: Requirements 16.3, 16.4**
    """

    def test_no_boeing_standard_evaluator_references(self):
        """No .py file under src/standard_evaluator/ contains 'boeing_standard_evaluator'."""
        violations = []
        for py_file in _get_all_source_files():
            content = py_file.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            for line_num, line in enumerate(lines, start=1):
                if FORBIDDEN_IMPORT_STRING in line:
                    rel_path = py_file.relative_to(SRC_DIR)
                    violations.append(f"{rel_path}:{line_num}: {line.strip()}")

        assert violations == [], (
            f"Found {len(violations)} occurrence(s) of '{FORBIDDEN_IMPORT_STRING}' "
            f"in source files:\n" + "\n".join(violations)
        )

    def test_no_boeing_proprietary_notices(self):
        """No .py file under src/standard_evaluator/ contains Boeing proprietary text."""
        violations = []
        for py_file in _get_all_source_files():
            content = py_file.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            for line_num, line in enumerate(lines, start=1):
                for pattern in FORBIDDEN_PROPRIETARY_PATTERNS:
                    if pattern.lower() in line.lower():
                        rel_path = py_file.relative_to(SRC_DIR)
                        violations.append(
                            f"{rel_path}:{line_num} [{pattern}]: {line.strip()}"
                        )

        assert violations == [], (
            f"Found {len(violations)} Boeing proprietary notice(s) in source files:\n"
            + "\n".join(violations)
        )

    def test_source_directory_has_python_files(self):
        """Sanity check: src/standard_evaluator/ contains .py files to scan."""
        files = _get_all_source_files()
        assert len(files) > 10, (
            f"Expected many .py files under src/standard_evaluator/, found only {len(files)}"
        )
