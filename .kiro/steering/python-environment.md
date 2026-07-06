---
inclusion: auto
---

# Python Environment — MANDATORY RULES

These rules are NON-NEGOTIABLE. Every Python command MUST use the virtual environment executable. Violations cause failures because system Python does not have the project dependencies installed.

## Virtual Environment

- **Python executable**: `.venv\Scripts\python.exe`
- **Working directory**: `c:\dev\standard-evaluator`
- **Platform**: Windows (cmd shell)

## FORBIDDEN Commands (NEVER use these)

```
python ...          ← WRONG: uses system Python
python3 ...        ← WRONG: doesn't exist on Windows
py ...             ← WRONG: uses system Python
pytest ...         ← WRONG: uses system pytest (may not exist)
pip ...            ← WRONG: uses system pip
.venv\Scripts\activate  ← WRONG: activate doesn't persist across commands
```

## REQUIRED Commands (ALWAYS use these)

| Task | Command |
|------|---------|
| Run pytest | `.venv\Scripts\python.exe -m pytest tests/` |
| Run specific test file | `.venv\Scripts\python.exe -m pytest tests/test_foo.py -v` |
| Run pytest with markers | `.venv\Scripts\python.exe -m pytest -m "not slow" tests/` |
| Run a script | `.venv\Scripts\python.exe script.py` |
| Run a module | `.venv\Scripts\python.exe -m module_name` |
| Pip install | `.venv\Scripts\python.exe -m pip install ...` |
| Quick Python check | `.venv\Scripts\python.exe -c "import standard_evaluator"` |
| Run nbconvert | `.venv\Scripts\python.exe -m nbconvert ...` |
| Run sphinx | `.venv\Scripts\python.exe -m sphinx ...` |

## Pytest Conventions

- Test files live in `tests/` at the project root
- Test file names: `test_*.py`
- Use `pytest.mark.slow` for tests that take >10 seconds
- Use `pytest.mark.integration` for tests requiring external tools (sphinx, nbconvert)
- Use `pytest.importorskip("module")` for optional dependency tests
- Use `-x` flag to stop on first failure when debugging
- Use `-v` flag for verbose output when you need to see individual test names
- Do NOT use `--timeout` (plugin not installed)

## Notebook Execution

When executing Jupyter notebooks programmatically:
- Use `nbformat` + `nbclient` or `nbconvert.preprocessors.ExecutePreprocessor`
- Set `kernel_name="python3"` (this refers to the kernel spec name, not the executable)
- The kernel will use whatever Python is registered as the `python3` kernel (which is the venv Python)

## Import Convention

- Always use `import standard_evaluator as se` as the primary import in notebooks and examples
- For internal modules, use full paths: `from standard_evaluator.evaluators.numpy_evaluator import NumpyEvaluator`
- Read existing source code BEFORE writing code that calls library APIs — verify constructor signatures, method names, and parameter names match the actual implementation
