# Requirements Document

## Introduction

The `standard-evaluator` package currently has a hard dependency on `aviary`, which is a large aerospace engineering package. Many users of `standard-evaluator` do not need aviary-specific functionality (custom encoder, metadata defaults, aviary values conversion). This feature makes `aviary` a fully optional dependency so that:

- Users who don't need aviary can install and use `standard-evaluator` without it
- Users who need aviary functionality install it via `pip install standard-evaluator[aviary]`
- Existing behavior is completely unchanged when aviary IS installed

## Glossary

- **Package**: The `standard-evaluator` Python distribution installed via pip
- **Aviary**: A third-party aerospace engineering library (`aviary`) that provides `AviaryValues`, `EngineDeck`, metadata, and enum types
- **Dymos**: A third-party trajectory optimization library (`dymos`) used alongside aviary for encoding Radau transcriptions and grid data
- **Optional_Extra**: A pip extras group declared in `pyproject.toml` under `[project.optional-dependencies]` enabling installation via `pip install package[extra_name]`
- **Lazy_Import_Guard**: A pattern that defers importing a module until it is actually used, raising a clear `ImportError` if the dependency is unavailable at execution time
- **AviaryEncoder**: A custom JSON encoder class in `aviary_encoder.py` that serializes aviary-specific types
- **StandardEval**: A class in `standard_evaluator.py` that uses aviary metadata as a default option value
- **om_converter**: A module providing functions (`convert_aviary`, `convert_engine_deck`) that reconstruct aviary objects from serialized JSON

## Requirements

### Requirement 1: Package imports without aviary installed

**User Story:** As a developer, I want to import the `standard-evaluator` package without aviary installed, so that I can use non-aviary functionality without installing a large, unneeded dependency.

#### Acceptance Criteria

1. WHEN aviary is not installed, THE Package SHALL import successfully via `import standard_evaluator`
2. WHEN aviary is not installed, THE Package SHALL expose all non-aviary symbols in `__all__` without raising an `ImportError`
3. WHEN aviary is installed, THE Package SHALL behave identically to the current implementation with all symbols available

### Requirement 2: Aviary and dymos declared as optional extra

**User Story:** As a developer, I want to install aviary and dymos support explicitly via `pip install standard-evaluator[aviary]`, so that I only pull in those dependencies when I need them.

#### Acceptance Criteria

1. THE Package SHALL declare aviary and dymos under `[project.optional-dependencies]` as an extras group named `aviary`
2. THE Package SHALL NOT list aviary or dymos in the `dependencies` array
3. WHEN a user runs `pip install standard-evaluator[aviary]`, THE Package SHALL install both aviary and dymos as dependencies

### Requirement 3: Clear error on aviary-dependent execution without aviary

**User Story:** As a developer, I want a clear and helpful error message when I attempt to use aviary-dependent functionality without aviary installed, so that I know exactly what to do to fix the problem.

#### Acceptance Criteria

1. WHEN aviary is not installed and a user instantiates AviaryEncoder, THE Package SHALL raise an `ImportError` with a message indicating that aviary is required and how to install it (e.g., `pip install standard-evaluator[aviary]`)
2. WHEN aviary is not installed and a user calls `convert_aviary()`, THE Package SHALL raise an `ImportError` with a message indicating that aviary is required and how to install it
3. WHEN aviary is not installed and a user calls `convert_engine_deck()`, THE Package SHALL raise an `ImportError` with a message indicating that aviary is required and how to install it
4. WHEN aviary is not installed and a user instantiates StandardEval, THE Package SHALL raise an `ImportError` with a message indicating that aviary is required and how to install it

### Requirement 4: Backward compatibility when aviary is installed

**User Story:** As an existing user who has aviary installed, I want all existing functionality to work exactly as before, so that this change does not break my workflows.

#### Acceptance Criteria

1. WHEN aviary is installed, THE AviaryEncoder SHALL serialize all aviary types (AviaryValues, EngineDeck, CoreAerodynamicsBuilder, CorePropulsionBuilder, aviary enums) identically to the current behavior
2. WHEN aviary is installed, THE StandardEval SHALL default its `metadata` option to `_MetaData` from aviary
3. WHEN aviary is installed, THE om_converter module SHALL provide `convert_aviary()` and `convert_engine_deck()` with identical behavior to the current implementation
4. WHEN aviary is installed, THE Package `__init__.py` SHALL export all current symbols without changes to their behavior

### Requirement 5: Consistent lazy import guard pattern

**User Story:** As a maintainer, I want a single consistent pattern for guarding aviary imports across all affected modules, so that the codebase is easy to understand and maintain.

#### Acceptance Criteria

1. THE Package SHALL use a single, consistent Lazy_Import_Guard pattern across `aviary_encoder.py`, `standard_evaluator.py`, and `om_converter.py`
2. THE Lazy_Import_Guard SHALL check for aviary and dymos availability at execution time, not at module import time
3. IF aviary or dymos import fails at execution time, THEN THE Lazy_Import_Guard SHALL raise an `ImportError` with a message containing the text `pip install standard-evaluator[aviary]`
