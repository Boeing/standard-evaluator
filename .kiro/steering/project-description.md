---
inclusion: auto
---

# Project: Standard Evaluator (SE)

## Overview

Standard Evaluator is an open-source Python library (published on PyPI as `standard-evaluator`) that provides a common API for defining, wrapping, and composing analysis codes and surrogate models. It was initially developed under NASA Contract 80GRC023CA045, and has been expanded since then.

**Documentation**: https://boeing.github.io/standard-evaluator/

The library has three main purposes:

1. **Common Evaluator API** — Expose analysis capabilities and surrogate models through a unified interface. End-users interact via Pandas DataFrames; developers can use a simplified NumPy-focused interface (`eval_np`, `eval_list`).

2. **Integration Framework Bridge** — Expose all evaluators to integration frameworks like OpenMDAO. The architecture is designed to support additional integration frameworks in the future.

3. **Assembly Serialization** — Capture the structure of assemblies of analyses in Pydantic classes or JSON files, and rebuild those assemblies from the stored information. Currently supports OpenMDAO; designed for future multi-framework support.

## Target Audience

The broader open-source community, particularly users of OpenMDAO and multidisciplinary design optimization (MDO) tools.

## Architecture

### Core Data Models (Pydantic)

| Class | Module | Purpose |
|-------|--------|---------|
| `OptProblem` | `problem.py` | Defines optimization problems: variables, responses, objectives, constraints |
| `EvaluatorInfo` | `evaluator.py` | Describes evaluator interfaces (inputs/outputs/metadata) |
| `GroupInfo` | `evaluator.py` | Represents groups of components with promotions and linkages |
| `EquationInfo` | `evaluator.py` | Represents equation-based components (wraps `om.ExecComp`) |
| `FloatVariable` | `problem.py` | Scalar float design variable with bounds, shift, scale |
| `IntVariable` | `problem.py` | Integer design variable |
| `ArrayVariable` | `problem.py` | NumPy array design variable |
| `CategoricalVariable` | `problem.py` | Categorical (discrete set) variable |

`Variable` is the union type: `FloatVariable | IntVariable | ArrayVariable | CategoricalVariable`

### Evaluator Hierarchy

```
Evaluator (ABC)                    # evaluators/abstract_evaluator.py
├── NumpyEvaluator                 # evaluators/numpy_evaluator.py — wraps a NumPy function
├── ExcelEvaluator                 # evaluators/excel_evaluator.py — wraps an Excel workbook (Windows)
├── ExecutableEvaluator            # evaluators/executable_evaluator.py — wraps a CLI executable
├── MatlabEvaluator                # evaluators/matlab_evaluator.py — wraps MATLAB
├── OpenMdaoEvaluator              # evaluators/open_mdao_evaluator.py — wraps an OpenMDAO component
├── ShiftScaleEvaluator            # evaluators/shift_scale_evaluator.py — applies shift/scale transform
└── (test evaluators)              # evaluators/test/ — benchmark problems (Rosenbrock, sphere, etc.)
```

All evaluators accept a `pd.DataFrame` of input sites and modify it in-place to add response columns. They also support:
- `eval_np(sites: np.ndarray) -> np.ndarray` for NumPy arrays
- `eval_list(sites: list) -> list` for list-of-lists
- Decorators for **caching** (SQLite-backed) and **logging**
- Finite-difference gradient computation via `_get_partials_by_central_difference`

### Surrogate Models

```
AbstractModel (ABC)                # surrogate_models/abstract_model.py
├── PolynomialModel                # surrogate_models/polynomial_model.py
└── SMT-based models               # surrogate_models/smt_models/
    ├── AbstractSMTModel
    ├── GENNModel, IDWModel, LSModel, RBFModel, RMTSModel, SOPAModel
```

SMT (Surrogate Modeling Toolbox) models are an optional dependency.

### OpenMDAO Integration

| Module | Role |
|--------|------|
| `standard_base.py` | Mixin (`StandardBase`) adding options management to OpenMDAO components |
| `standard_evaluator.py` | `StandardEval` — extends `StandardBase` with Aviary metadata lookup |
| `standard_group.py` | `StandardGroup` — group-level counterpart |
| `om_converter.py` | Bi-directional conversion: OpenMDAO ↔ JoinedInfo/JSON; assembly save/load |
| `components/evaluator_om_component.py` | OpenMDAO `ExplicitComponent` that wraps an `Evaluator` |

### Utilities

- `utilities/mapping.py` — Flatten/unflatten between rolled (array) and unrolled (scalar) representations
- `utilities/shift_scale.py` — Shift/scale transformations
- `utilities/opt_problem_utility.py` — OptProblem helper operations
- `utilities/se_arrays.py` — Array manipulation helpers
- `utilities/option_merge.py` — Merge Pydantic model instances

## Aviary Integration

Aviary (NASA's aircraft design tool) integration is **optional**. The library is actively moving toward making Aviary a fully optional dependency (see `optional-aviary-dependency` spec). When Aviary is installed:
- `StandardEval` uses Aviary's `_MetaData` for variable lookup
- `om_converter.py` can serialize/deserialize `AviaryValues` and `EngineDeck` objects

## Key Conventions

### Naming

- Package: `standard_evaluator` (underscore)
- PyPI: `standard-evaluator` (hyphen)
- Common import alias: `import standard_evaluator as se`
- Variable classes use PascalCase: `FloatVariable`, `ArrayVariable`, `OptProblem`
- Evaluator subclasses use PascalCase with `Evaluator` suffix

### Data Flow Pattern

1. Define an `OptProblem` or `EvaluatorInfo` describing inputs/outputs
2. Instantiate an `Evaluator` subclass with that problem definition
3. Create a `pd.DataFrame` with input columns
4. Call the evaluator (or use `eval_np`/`eval_list`) — responses are added to the DataFrame

### Options System

`OptionsDictionaryUnit` extends OpenMDAO's `OptionsDictionary` with units support. Components declare required options via `_define_options()` classmethod. Options can be tagged on inputs/outputs to track which options affect which variables.

## Testing

- **Framework**: pytest (with hypothesis for property-based testing)
- **Test location**: `tests/`
- **Run tests**: `.venv\Scripts\python.exe -m pytest tests/`
- **Test evaluators**: `evaluators/test/` contains benchmark optimization problems (Rosenbrock, sphere, cantilevered beam, etc.) used both for testing and as examples

## Build & Release

- Build system: setuptools with `setuptools_scm` for versioning (version derived from git tags)
- Version scheme: post-release
- CI: GitHub Actions (`python-package.yml` for tests, `publish-pypi.yml` for releases, `pages.yml` for docs)
- Docs: Sphinx with MyST-NB (Jupyter notebooks as documentation)

## Dependencies

### Required
numpy, pandas, pydantic (v2+), annotated-types, numpydantic, openpyxl, dask, networkx, openmdao, json-numpy, h5py

### Optional Groups
- `smt` — SMT surrogate modeling (smt>=2.10.1)
- `aviary` — NASA Aviary integration (aviary, dymos)
- `excel` — Excel evaluator on Windows (pywin32)
- `test` — Testing (pytest, hypothesis, numdifftools)

## Architectural Principles

1. **Framework independence** — Core data models (OptProblem, EvaluatorInfo) are pure Pydantic; OpenMDAO/Aviary are integration layers, not dependencies of the core.
2. **Serializable everything** — All problem definitions and assembly structures can be serialized to JSON and reconstructed.
3. **Evaluator composability** — Any evaluator can be wrapped, cached, logged, or composed into larger systems.
4. **Rolled/unrolled duality** — Array variables can be treated as single named arrays (rolled) or expanded into indexed scalars (unrolled) depending on context.
5. **Options-driven configurability** — Components declare their options formally; the system tracks which inputs/outputs depend on which options.
