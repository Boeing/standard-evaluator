"""Unit tests for guarded import behavior when aviary is absent.

These tests verify that aviary-dependent entry points raise clear ImportError
messages when aviary is not installed, while the package itself imports cleanly.

Uses unittest.mock.patch.dict to simulate aviary/dymos absence and
importlib.reload to pick up the mocked state.

**Validates: Requirements 1.1, 1.2, 3.1, 3.2, 3.3, 3.4, 5.3**
"""

import re
import sys
import importlib
from unittest.mock import patch

import pytest

# Ensure the package is initially loaded (with aviary present in dev env)
import standard_evaluator
import standard_evaluator.aviary_encoder
import standard_evaluator.standard_evaluator
import standard_evaluator.om_converter


# Aviary/dymos module names to block (set to None in sys.modules)
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

# Expected substring in all ImportError messages
_INSTALL_HINT = "pip install standard-evaluator[aviary]"

# Escaped version for use as a regex pattern
_INSTALL_HINT_RE = re.escape(_INSTALL_HINT)


def _block_aviary_modules():
    """Return a dict mapping aviary/dymos module names to None for patch.dict."""
    return {mod: None for mod in _AVIARY_MODULES_TO_BLOCK}


def _reload_guarded_modules():
    """Reload modules with aviary guards to pick up mocked state.

    Reloads leaf modules first, then __init__.py so it sees reloaded submodules.
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
    """Reload guarded modules after each test to restore _AVIARY_AVAILABLE = True.

    Tests in this file reload modules with aviary mocked away. Even though
    patch.dict restores sys.modules entries, the reloaded module objects retain
    stale state. This fixture ensures subsequent tests see correct state.
    """
    yield
    _reload_guarded_modules()


class TestImportSucceedsWithoutAviary:
    """Verify the package imports cleanly when aviary is absent."""

    def test_import_standard_evaluator_succeeds(self):
        """Requirement 1.1: import standard_evaluator succeeds without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            mod = _reload_guarded_modules()
            assert mod is not None
            assert hasattr(mod, "__all__")

    def test_all_non_aviary_symbols_accessible(self):
        """Requirement 1.2: All non-aviary symbols in __all__ are accessible without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            mod = _reload_guarded_modules()

            # Symbols that are classes requiring aviary at execution time
            # (importable as a class reference, but not instantiable)
            aviary_execution_symbols = {"AviaryEncoder"}

            for symbol_name in mod.__all__:
                assert hasattr(mod, symbol_name), (
                    f"Symbol '{symbol_name}' in __all__ is not accessible "
                    f"when aviary is not installed"
                )
                obj = getattr(mod, symbol_name)
                if symbol_name not in aviary_execution_symbols:
                    assert obj is not None, (
                        f"Symbol '{symbol_name}' is None when aviary is not installed"
                    )


class TestAviaryEncoderRaisesWithoutAviary:
    """Verify AviaryEncoder raises ImportError when aviary is absent."""

    def test_aviary_encoder_instantiation_raises_import_error(self):
        """Requirement 3.1: AviaryEncoder() raises ImportError without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.aviary_encoder import AviaryEncoder

            with pytest.raises(ImportError):
                AviaryEncoder()

    def test_aviary_encoder_error_message_contains_install_hint(self):
        """Requirement 5.3: AviaryEncoder error message contains install instructions."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.aviary_encoder import AviaryEncoder

            with pytest.raises(ImportError, match=_INSTALL_HINT_RE):
                AviaryEncoder()


class TestStandardEvalRaisesWithoutAviary:
    """Verify StandardEval.initialize() raises ImportError when aviary is absent.

    Note: StandardEval.__init__ does not call initialize() automatically.
    The guard fires when initialize() is called explicitly, which is the
    entry point for aviary-dependent behavior (accessing _MetaData).
    """

    def test_standard_eval_initialize_raises_import_error(self):
        """Requirement 3.4: StandardEval initialization raises ImportError without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.standard_evaluator import StandardEval

            instance = StandardEval()
            with pytest.raises(ImportError):
                instance.initialize()

    def test_standard_eval_error_message_contains_install_hint(self):
        """Requirement 5.3: StandardEval error message contains install instructions."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.standard_evaluator import StandardEval

            instance = StandardEval()
            with pytest.raises(ImportError, match=_INSTALL_HINT_RE):
                instance.initialize()


class TestConvertAviaryRaisesWithoutAviary:
    """Verify convert_aviary raises ImportError when aviary is absent."""

    def test_convert_aviary_raises_import_error(self):
        """Requirement 3.2: convert_aviary({}) raises ImportError without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.om_converter import convert_aviary

            with pytest.raises(ImportError):
                convert_aviary({})

    def test_convert_aviary_error_message_contains_install_hint(self):
        """Requirement 5.3: convert_aviary error message contains install instructions."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.om_converter import convert_aviary

            with pytest.raises(ImportError, match=_INSTALL_HINT_RE):
                convert_aviary({})


class TestConvertEngineDeckRaisesWithoutAviary:
    """Verify convert_engine_deck raises ImportError when aviary is absent."""

    def test_convert_engine_deck_raises_import_error(self):
        """Requirement 3.3: convert_engine_deck({}) raises ImportError without aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.om_converter import convert_engine_deck

            with pytest.raises(ImportError):
                convert_engine_deck({})

    def test_convert_engine_deck_error_message_contains_install_hint(self):
        """Requirement 5.3: convert_engine_deck error message contains install instructions."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            from standard_evaluator.om_converter import convert_engine_deck

            with pytest.raises(ImportError, match=_INSTALL_HINT_RE):
                convert_engine_deck({})


class TestAllErrorMessagesConsistent:
    """Verify all guarded entry points produce error messages with install hint.

    Requirement 5.3: All ImportError messages contain 'pip install standard-evaluator[aviary]'.
    """

    @pytest.mark.parametrize(
        "entry_point_label,import_path,callable_factory",
        [
            (
                "AviaryEncoder",
                "standard_evaluator.aviary_encoder",
                lambda mod: lambda: mod.AviaryEncoder(),
            ),
            (
                "StandardEval.initialize",
                "standard_evaluator.standard_evaluator",
                lambda mod: lambda: mod.StandardEval().initialize(),
            ),
            (
                "convert_aviary",
                "standard_evaluator.om_converter",
                lambda mod: lambda: mod.convert_aviary({}),
            ),
            (
                "convert_engine_deck",
                "standard_evaluator.om_converter",
                lambda mod: lambda: mod.convert_engine_deck({}),
            ),
        ],
    )
    def test_error_message_contains_install_hint(
        self, entry_point_label, import_path, callable_factory
    ):
        """All aviary-dependent entry points mention how to install aviary."""
        with patch.dict(sys.modules, _block_aviary_modules()):
            _reload_guarded_modules()
            mod = sys.modules[import_path]

            invoke = callable_factory(mod)
            with pytest.raises(ImportError) as exc_info:
                invoke()

            assert _INSTALL_HINT in str(exc_info.value), (
                f"{entry_point_label} ImportError message does not contain "
                f"'{_INSTALL_HINT}'. Got: {exc_info.value}"
            )
