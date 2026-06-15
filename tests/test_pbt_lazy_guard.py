"""Property-based test for lazy guard ImportError on all guarded entry points.

Feature: optional-aviary-dependency
Property 1: Lazy guard raises with install instructions for all guarded entry points

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 5.2, 5.3**
"""

import sys
import importlib
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure the package is initially loaded (with aviary present in dev env)
import standard_evaluator
import standard_evaluator.aviary_encoder
import standard_evaluator.standard_evaluator
import standard_evaluator.om_converter


# Aviary/dymos module names to block
_AVIARY_MODULES_TO_BLOCK = [
    "aviary",
    "aviary.variable_info",
    "aviary.variable_info.enums",
    "aviary.variable_info.variable_meta_data",
    "aviary.utils",
    "aviary.utils.aviary_values",
    "aviary.subsystems",
    "aviary.subsystems.propulsion",
    "aviary.subsystems.propulsion.engine_deck",
    "aviary.subsystems.propulsion.propulsion_builder",
    "aviary.subsystems.aerodynamics",
    "aviary.subsystems.aerodynamics.aerodynamics_builder",
    "dymos",
    "dymos.transcriptions",
    "dymos.transcriptions.grid_data",
]

# The set of guarded entry points with their invocation details
_ENTRY_POINTS = [
    "AviaryEncoder",
    "StandardEval",
    "convert_aviary",
    "convert_engine_deck",
]


def _block_aviary_modules():
    """Return a dict mapping aviary/dymos module names to None for use with patch.dict."""
    return {mod: None for mod in _AVIARY_MODULES_TO_BLOCK}


def _reload_guarded_modules():
    """Reload only the modules that have aviary guards to pick up the mocked state.

    We reload from leaf to root so that the __init__.py sees the reloaded submodules.
    """
    modules_to_reload = [
        "standard_evaluator.aviary_encoder",
        "standard_evaluator.standard_evaluator",
        "standard_evaluator.om_converter",
        "standard_evaluator",
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


def _invoke_entry_point(entry_point_name: str):
    """Invoke a guarded entry point by name, triggering the lazy guard.

    For StandardEval, the guard is in initialize() (called after construction),
    matching the pattern where aviary is only needed at execution time.
    """
    if entry_point_name == "AviaryEncoder":
        from standard_evaluator.aviary_encoder import AviaryEncoder
        AviaryEncoder()
    elif entry_point_name == "StandardEval":
        from standard_evaluator.standard_evaluator import StandardEval
        instance = StandardEval()
        instance.initialize()
    elif entry_point_name == "convert_aviary":
        from standard_evaluator.om_converter import convert_aviary
        convert_aviary({})
    elif entry_point_name == "convert_engine_deck":
        from standard_evaluator.om_converter import convert_engine_deck
        convert_engine_deck({})


# Strategy: randomly sample from the set of guarded entry points
entry_point_strategy = st.sampled_from(_ENTRY_POINTS)


@given(entry_point=entry_point_strategy)
@settings(max_examples=100)
def test_lazy_guard_raises_import_error_with_install_instructions(entry_point):
    """Property 1: For any guarded entry point, when aviary is absent:
    - The containing module imports without error
    - Invoking the entry point raises ImportError
    - The error message contains 'pip install standard-evaluator[aviary]'

    Feature: optional-aviary-dependency
    Property 1: Lazy guard raises with install instructions for all guarded entry points

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 5.2, 5.3**
    """
    with patch.dict(sys.modules, _block_aviary_modules()):
        # Reload modules to pick up the mocked aviary absence
        _reload_guarded_modules()

        # Verify the containing module imports without error
        # (reload already succeeded if we get here)
        if entry_point in ("AviaryEncoder",):
            mod = sys.modules["standard_evaluator.aviary_encoder"]
        elif entry_point in ("StandardEval",):
            mod = sys.modules["standard_evaluator.standard_evaluator"]
        else:
            mod = sys.modules["standard_evaluator.om_converter"]
        assert mod is not None, (
            f"Module for {entry_point} failed to import when aviary is absent"
        )

        # Verify invoking the entry point raises ImportError
        raised = False
        error_message = ""
        try:
            _invoke_entry_point(entry_point)
        except ImportError as e:
            raised = True
            error_message = str(e)

        assert raised, (
            f"Expected ImportError when invoking {entry_point} without aviary, "
            f"but no ImportError was raised"
        )

        # Verify the error message contains the install instructions
        assert "pip install standard-evaluator[aviary]" in error_message, (
            f"ImportError for {entry_point} does not contain install instructions. "
            f"Got: {error_message!r}"
        )

    # Restore modules after exiting the patch context
    _reload_guarded_modules()
