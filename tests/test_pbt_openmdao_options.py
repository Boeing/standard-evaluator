"""Property-based test for create_openmdao_options list-to-tuple conversion.

Feature: surrogate-replacement-demo
Property 2: create_openmdao_options converts list-valued options to tuples

**Validates: Requirements 2.1, 2.2**

For any serialized openmdao_options dictionary where option values at
_dict[name]['val'] are lists (resulting from JSON serialization of tuples),
create_openmdao_options() shall return a dictionary where all such values
are tuples, preserving element order and content.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.om_converter import create_openmdao_options


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Strategy: simple JSON-serializable values that could appear in list elements
_simple_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(min_size=0, max_size=10),
    st.booleans(),
)

# Strategy: list values of varying lengths (simulating JSON-serialized tuples)
_list_values = st.lists(_simple_values, min_size=0, max_size=10)

# Strategy: non-list scalar values that should pass through unchanged
_non_list_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(min_size=1, max_size=20),
    st.booleans(),
)

# Strategy: option name (avoiding 'aviary_options' which has special handling)
_option_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
).filter(lambda x: x != "aviary_options")


@st.composite
def openmdao_options_with_lists(draw):
    """Generate an info_dict with '_dict' containing random options with list values.

    Returns:
        tuple: (info_dict, expected_names_with_lists) where expected_names_with_lists
               is the set of option names that had list values.
    """
    # Generate between 1 and 5 options with list values
    num_list_options = draw(st.integers(min_value=1, max_value=5))
    # Generate between 0 and 3 options with non-list values
    num_scalar_options = draw(st.integers(min_value=0, max_value=3))

    _dict = {}
    list_option_names = set()

    for i in range(num_list_options):
        name = draw(_option_names.filter(lambda x: x not in _dict))
        val = draw(_list_values)
        _dict[name] = {"val": val}
        list_option_names.add(name)

    for i in range(num_scalar_options):
        name = draw(_option_names.filter(lambda x: x not in _dict))
        val = draw(_non_list_values)
        _dict[name] = {"val": val}

    info_dict = {"_dict": _dict}
    return info_dict, list_option_names


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestCreateOpenmdaoOptionsListToTuple:
    """Property 2: create_openmdao_options converts list-valued options to tuples.

    For any serialized openmdao_options dictionary where option values are lists,
    create_openmdao_options() returns a dictionary where all list values are
    converted to tuples, preserving element order and content.
    """

    @given(data=openmdao_options_with_lists())
    @settings(max_examples=100, deadline=None)
    def test_list_values_converted_to_tuples(self, data):
        """All list-valued options are converted to tuples.

        **Validates: Requirements 2.1, 2.2**
        """
        info_dict, list_option_names = data

        result = create_openmdao_options(info_dict)

        # All options that had list values should now be tuples
        for name in list_option_names:
            assert name in result, (
                f"Option '{name}' missing from result"
            )
            assert isinstance(result[name], tuple), (
                f"Option '{name}' should be a tuple, got {type(result[name]).__name__}"
            )

    @given(data=openmdao_options_with_lists())
    @settings(max_examples=100, deadline=None)
    def test_list_to_tuple_preserves_order_and_content(self, data):
        """Converted tuples preserve element order and content from original lists.

        **Validates: Requirements 2.1, 2.2**
        """
        info_dict, list_option_names = data

        # Capture original list values before conversion (function mutates _dict)
        original_lists = {
            name: list(info_dict["_dict"][name]["val"])
            for name in list_option_names
        }

        result = create_openmdao_options(info_dict)

        for name, original_list in original_lists.items():
            expected_tuple = tuple(original_list)
            assert result[name] == expected_tuple, (
                f"Option '{name}': expected {expected_tuple}, got {result[name]}"
            )

    @given(data=openmdao_options_with_lists())
    @settings(max_examples=100, deadline=None)
    def test_non_list_values_unchanged(self, data):
        """Non-list values pass through without modification.

        **Validates: Requirements 2.1, 2.2**
        """
        info_dict, list_option_names = data

        # Capture original non-list values before conversion
        non_list_options = {
            name: info["val"]
            for name, info in info_dict["_dict"].items()
            if name not in list_option_names
        }

        result = create_openmdao_options(info_dict)

        for name, original_val in non_list_options.items():
            assert result[name] == original_val, (
                f"Non-list option '{name}': expected {original_val!r}, "
                f"got {result[name]!r}"
            )
