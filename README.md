# Standard Evaluator

<p align="center">
  <img src="docs/assets/logo.svg" alt="Standard Evaluator Logo" width="270">
</p>

Standard Evaluator is an open-source Python library (published on PyPI as `standard-evaluator`) that provides a common API for defining, wrapping, and composing analysis codes and surrogate models. It was initially developed under NASA Contract 80GRC023CA045, and has been expanded since then.

**Documentation**: https://boeing.github.io/standard-evaluator/

The library has three main purposes:

1. **Common Evaluator API** — Expose analysis capabilities and surrogate models through a unified interface. End-users interact via Pandas DataFrames; developers can use a simplified NumPy-focused interface (`eval_np`, `eval_list`).

2. **Integration Framework Bridge** — Expose all evaluators to integration frameworks like OpenMDAO. The architecture is designed to support additional integration frameworks in the future.

3. **Assembly Serialization** — Capture the structure of assemblies of analyses in Pydantic classes or JSON files, and rebuild those assemblies from the stored information. Currently supports OpenMDAO; designed for future multi-framework support.

Note that user of the library are responsible for installing the third party open source components and for complying with the terms and conditions of the respective open source licenses governing the third party open source components.

## Documentation

Full documentation with demos and API reference is available at:
https://boeing.github.io/standard-evaluator/

## Installation

`pip install standard-evaluator`

Optional extras:

```bash
pip install standard-evaluator[smt]      # Surrogate Modeling Toolbox models
pip install standard-evaluator[aviary]   # NASA Aviary integration
pip install standard-evaluator[test]     # Testing dependencies
```

Optionally clone the repo and install locally for development:

```bash
git clone <repo-url>
cd standard-evaluator
pip install -e .[test,smt]
```

## Project Structure

- `src/` — Source code of the standard evaluator library
- `docs/` — Documentation source (Sphinx + Jupyter notebooks)
- `tests/` — Unit and property-based tests

## Quick Start

```python
import standard_evaluator as se
from standard_evaluator.evaluators import OpenMDAOEvaluator

# Capture an OpenMDAO assembly interface
info = se.get_interface(prob.model)

# Set variable bounds for surrogate training
se.set_variable_bounds(info, {'x': (0, 10), 'y': (-5, 5)})

# Build an optimization problem and evaluator
opt_problem = se.build_opt_problem(info, prob)
evaluator = OpenMDAOEvaluator(prob, opt_problem=opt_problem)
```
