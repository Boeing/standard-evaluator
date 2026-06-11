"""Property-based test for AviaryEncoder type-marker round trip.

Feature: optional-aviary-dependency, Property 2: AviaryEncoder type-marker round trip

**Validates: Requirements 4.1**

For any valid aviary-typed object (Enum values, sets, tuples, numpy arrays,
pathlib.WindowsPath), when aviary IS installed, AviaryEncoder().default(obj)
shall produce a dictionary containing a recognizable type-marker key that
uniquely identifies the original type.
"""

import importlib
import pathlib
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aviary.variable_info.enums import EquationsOfMotion, LegacyCode, ProblemType


def _reload_with_aviary():
    """Reload guarded modules to ensure _AVIARY_AVAILABLE is True.

    Other tests may have reloaded these modules with aviary mocked away.
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


@pytest.fixture(autouse=True)
def ensure_aviary_available():
    """Ensure guarded modules are reloaded with aviary available before each test."""
    _reload_with_aviary()
    yield
    _reload_with_aviary()


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Strategy: random aviary Enum values
_aviary_enums = st.sampled_from(
    list(ProblemType) + list(EquationsOfMotion) + list(LegacyCode)
)

# Strategy: sets of simple JSON-serializable values
_simple_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(min_size=0, max_size=20),
)
_sets = st.frozensets(_simple_values, min_size=0, max_size=10).map(set)

# Strategy: tuples of simple values
_tuples = st.lists(_simple_values, min_size=0, max_size=10).map(tuple)

# Strategy: numpy arrays with random shapes and dtypes
_numpy_dtypes = st.sampled_from([np.float64, np.float32, np.int32, np.int64])
_numpy_arrays = st.builds(
    lambda shape, dtype: np.zeros(shape, dtype=dtype),
    shape=st.tuples(
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=5),
    ),
    dtype=_numpy_dtypes,
)

# Strategy: WindowsPath with random path strings
_windows_paths = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-/\\.",
    ),
    min_size=1,
    max_size=50,
).map(pathlib.WindowsPath)


# Combined strategy: one of the supported types paired with expected marker key
@st.composite
def aviary_typed_object_with_marker(draw):
    """Generate a random aviary-typed object and its expected type-marker key."""
    choice = draw(st.integers(min_value=0, max_value=4))

    if choice == 0:
        obj = draw(_aviary_enums)
        return obj, "__enum__"
    elif choice == 1:
        obj = draw(_sets)
        return obj, "__set__"
    elif choice == 2:
        obj = draw(_tuples)
        return obj, "__tuple__"
    elif choice == 3:
        obj = draw(_numpy_arrays)
        return obj, "__numpy__"
    else:
        obj = draw(_windows_paths)
        return obj, "__pathlib.WindowsPath__"


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestAviaryEncoderTypeMarkerRoundTrip:
    """Property 2: AviaryEncoder type-marker round trip.

    For any valid aviary-typed object, AviaryEncoder().default(obj)
    returns a dict containing the expected type-marker key.
    """

    @given(data=aviary_typed_object_with_marker())
    @settings(max_examples=100)
    def test_default_returns_dict_with_type_marker(self, data):
        """AviaryEncoder.default() produces a dict with the correct type-marker key.

        **Validates: Requirements 4.1**
        """
        from standard_evaluator.aviary_encoder import AviaryEncoder

        obj, expected_key = data
        encoder = AviaryEncoder()
        result = encoder.default(obj)

        assert isinstance(result, dict), (
            f"Expected dict from AviaryEncoder.default({type(obj).__name__}), "
            f"got {type(result).__name__}"
        )
        assert expected_key in result, (
            f"Expected key '{expected_key}' in result for {type(obj).__name__}, "
            f"got keys: {list(result.keys())}"
        )
