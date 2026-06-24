---
inclusion: auto
---

# Python Environment

## Virtual Environment

Always use the project virtual environment when running Python commands.

- **Python executable**: `.venv\Scripts\python.exe`
- **Working directory**: `c:\dev\standard-evaluator`

## Rules

- NEVER use `python`, `python3`, `py`, or any system Python directly.
- NEVER use `.venv\Scripts\activate` — use the full path to the executable instead.
- For running scripts: `.venv\Scripts\python.exe script.py`
- For running modules: `.venv\Scripts\python.exe -m module_name`
- For running pytest: `.venv\Scripts\python.exe -m pytest`
- For pip installs: `.venv\Scripts\python.exe -m pip install ...`

## Examples

```cmd
.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -c "import standard_evaluator; print('ok')"
```
