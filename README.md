# Standard Evaluator

A Python library providing a common API for defining, wrapping, and composing analysis codes and surrogate models. Originally developed under NASA Contract 80GRC023CA045.

## Documentation

Full documentation with demos and API reference is available at:
https://mattermost.web.boeing.com/devhub/pl/spseni331tg19epgei6eot1stw

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
