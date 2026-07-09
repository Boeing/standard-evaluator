"""Property-based test for get_linkages tuple handling.

Feature: surrogate-replacement-demo
Property 1: get_linkages correctly extracts linkages from any valid connection metadata

**Validates: Requirements 1.1, 1.4**
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.om_converter import get_linkages


# Strategy: generate valid source and target variable names (OpenMDAO style)
_om_name_strategy = st.from_regex(r"[a-z][a-z0-9_]{0,19}\.[a-z][a-z0-9_]{0,19}", fullmatch=True)

# Strategy: generate a 2-tuple connection value (OpenMDAO 3.43 format)
# Format: (source_name, metadata_dict)
_two_tuple_strategy = st.tuples(
    _om_name_strategy,
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers()),
)

# Strategy: generate a 3-tuple connection value (legacy format)
# Format: (source_name, src_indices_or_None, flat_src_indices_or_None)
_three_tuple_strategy = st.tuples(
    _om_name_strategy,
    st.none(),
    st.none(),
)

# Strategy: generate a single connection entry (either 2-tuple or 3-tuple)
_connection_value_strategy = st.one_of(_two_tuple_strategy, _three_tuple_strategy)

# Strategy: generate a _manual_connections dictionary
_manual_connections_strategy = st.dictionaries(
    keys=_om_name_strategy,
    values=_connection_value_strategy,
    min_size=0,
    max_size=10,
)


@given(manual_connections=_manual_connections_strategy)
@settings(max_examples=100, deadline=None)
def test_get_linkages_extracts_correct_pairs_from_any_valid_metadata(manual_connections):
    """Property 1: For any valid connection metadata dict with 2-tuple or 3-tuple entries,
    get_linkages() returns a list of (source, target) pairs where source equals value[0]
    and target equals the dictionary key, without raising IndexError.

    Feature: surrogate-replacement-demo
    Property 1: get_linkages correctly extracts linkages from any valid connection metadata

    **Validates: Requirements 1.1, 1.4**
    """
    # Create a mock om.Group with _manual_connections attribute
    mock_group = MagicMock()
    mock_group._manual_connections = manual_connections

    # Should not raise IndexError
    result = get_linkages(mock_group)

    # Verify result is a list of tuples
    assert isinstance(result, list)
    assert len(result) == len(manual_connections)

    # Verify each (source, target) pair is correct
    for (source, target), (key, value) in zip(result, manual_connections.items()):
        assert source == value[0], (
            f"Expected source={value[0]!r}, got source={source!r}"
        )
        assert target == key, (
            f"Expected target={key!r}, got target={target!r}"
        )
