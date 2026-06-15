"""Test that standard_evaluator imports succeed without aviary installed.

This test uses mock patching to simulate aviary/dymos being absent,
then verifies that `import standard_evaluator` succeeds and all
non-aviary symbols in `__all__` are accessible.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

import sys
import importlib
from unittest.mock import patch

import pytest

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


def _block_aviary_modules():
    """Return a dict mapping aviary/dymos module names to None for use with patch.dict."""
    return {mod: None for mod in _AVIARY_MODULES_TO_BLOCK}


def _reload_guarded_modules():
    """Reload only the modules that have aviary guards to pick up the mocked state.
    
    We reload from leaf to root so that the __init__.py sees the reloaded submodules.
    We do NOT remove/reload standard_base, utilities, problem, etc. since they have 
    no aviary dependency and removing them causes numpy reload issues.
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
    return sys.modules["standard_evaluator"]


@pytest.fixture(autouse=True)
def restore_modules_after_test():
    """Reload guarded modules after each test to restore _AVIARY_AVAILABLE = True."""
    yield
    _reload_guarded_modules()


class TestImportWithoutAviary:
    """Verify that importing standard_evaluator succeeds when aviary is absent."""

    def test_import_succeeds_without_aviary(self):
        """Requirement 1.1: import standard_evaluator succeeds when aviary is not installed."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            mod = _reload_guarded_modules()
            # If we get here, import succeeded
            assert mod is not None
            assert hasattr(mod, "__all__")

    def test_all_non_aviary_symbols_accessible(self):
        """Requirement 1.2: All non-aviary symbols in __all__ are accessible without error."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            mod = _reload_guarded_modules()
            
            # These are the symbols that require aviary at execution time
            # (they should still be importable as a class, just not usable)
            aviary_execution_symbols = {"AviaryEncoder"}
            
            for symbol_name in mod.__all__:
                # All symbols should be accessible as attributes
                assert hasattr(mod, symbol_name), (
                    f"Symbol '{symbol_name}' in __all__ is not accessible "
                    f"when aviary is not installed"
                )
                obj = getattr(mod, symbol_name)
                # Non-aviary symbols should not be None
                if symbol_name not in aviary_execution_symbols:
                    assert obj is not None, (
                        f"Symbol '{symbol_name}' is None when aviary is not installed"
                    )

    def test_aviary_encoder_class_importable(self):
        """AviaryEncoder class should be importable (but not instantiable) without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            mod = _reload_guarded_modules()
            
            # AviaryEncoder should be importable as a class
            assert hasattr(mod, "AviaryEncoder")
            encoder_cls = getattr(mod, "AviaryEncoder")
            assert encoder_cls is not None
            # It should be a class (not None sentinel)
            assert isinstance(encoder_cls, type)

    def test_all_symbols_available_with_aviary(self):
        """Requirement 1.3: All symbols available when aviary IS installed."""
        # Reload without any mocking - aviary should be available in dev env
        mod = _reload_guarded_modules()
        
        for symbol_name in mod.__all__:
            assert hasattr(mod, symbol_name), (
                f"Symbol '{symbol_name}' in __all__ is not accessible "
                f"when aviary IS installed"
            )
            obj = getattr(mod, symbol_name)
            assert obj is not None, (
                f"Symbol '{symbol_name}' is None when aviary IS installed"
            )
