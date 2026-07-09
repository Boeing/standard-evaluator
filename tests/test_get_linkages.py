"""Unit tests for get_linkages function.

Tests specific scenarios for connection metadata handling:
- 2-tuple metadata (OpenMDAO 3.43 format)
- 3-tuple metadata with None values (legacy format)
- Mixed 2-tuple and 3-tuple metadata

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from unittest.mock import MagicMock

from standard_evaluator.om_converter import get_linkages


class TestGetLinkagesTwoTupleMetadata:
    """Test get_linkages with OpenMDAO 3.43 format (2-tuple metadata)."""

    def test_single_2tuple_connection(self):
        """Req 1.1: Extract source from 2-tuple without accessing third element."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "lift_calc.q": ("pressure_calc.pressure", {"src_indices": None}),
        }

        result = get_linkages(mock_group)

        assert result == [("pressure_calc.pressure", "lift_calc.q")]

    def test_multiple_2tuple_connections(self):
        """Req 1.1: Handle multiple 2-tuple connections correctly."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "lift_calc.q": ("pressure_calc.pressure", {}),
            "drag_calc.q": ("pressure_calc.pressure", {}),
        }

        result = get_linkages(mock_group)

        assert result == [
            ("pressure_calc.pressure", "lift_calc.q"),
            ("pressure_calc.pressure", "drag_calc.q"),
        ]

    def test_2tuple_with_nonempty_metadata_dict(self):
        """Req 1.1: 2-tuple with populated metadata dict does not raise."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "comp_b.x": ("comp_a.y", {"src_indices": [0, 1], "flat": True}),
        }

        result = get_linkages(mock_group)

        assert result == [("comp_a.y", "comp_b.x")]


class TestGetLinkagesThreeTupleMetadata:
    """Test get_linkages with legacy format (3-tuple metadata)."""

    def test_3tuple_with_none_values(self):
        """Req 1.4: Backward compatibility with 3-tuple (None, None) indexing."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "lift_calc.q": ("pressure_calc.pressure", None, None),
        }

        result = get_linkages(mock_group)

        assert result == [("pressure_calc.pressure", "lift_calc.q")]

    def test_3tuple_with_nonnone_indexing_prints_diagnostic(self, capsys):
        """Req 1.2: Print diagnostic when indexing info is non-None."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "comp_b.x": ("comp_a.y", [0, 1, 2], None),
        }

        result = get_linkages(mock_group)

        assert result == [("comp_a.y", "comp_b.x")]
        captured = capsys.readouterr()
        assert "Indexing used" in captured.out

    def test_3tuple_with_both_nonnone_prints_diagnostic(self, capsys):
        """Req 1.2: Print diagnostic when both src_indices and flat_src_indices are non-None."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "comp_b.x": ("comp_a.y", [0, 1], True),
        }

        result = get_linkages(mock_group)

        assert result == [("comp_a.y", "comp_b.x")]
        captured = capsys.readouterr()
        assert "Indexing used" in captured.out

    def test_3tuple_with_both_none_no_diagnostic(self, capsys):
        """Req 1.2: No diagnostic printed when indexing values are both None."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "comp_b.x": ("comp_a.y", None, None),
        }

        result = get_linkages(mock_group)

        assert result == [("comp_a.y", "comp_b.x")]
        captured = capsys.readouterr()
        assert captured.out == ""


class TestGetLinkagesMixedMetadata:
    """Test get_linkages with mixed 2-tuple and 3-tuple metadata."""

    def test_mixed_2tuple_and_3tuple(self):
        """Req 1.4: Handle mixed metadata formats in same group."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "lift_calc.q": ("pressure_calc.pressure", {"src_indices": None}),
            "drag_calc.q": ("pressure_calc.pressure", None, None),
        }

        result = get_linkages(mock_group)

        assert result == [
            ("pressure_calc.pressure", "lift_calc.q"),
            ("pressure_calc.pressure", "drag_calc.q"),
        ]

    def test_mixed_with_diagnostic_on_3tuple_only(self, capsys):
        """Req 1.2, 1.4: Diagnostic only printed for 3-tuple with non-None indexing."""
        mock_group = MagicMock()
        mock_group._manual_connections = {
            "comp_b.x": ("comp_a.y", {}),
            "comp_c.z": ("comp_a.w", [0, 1], None),
        }

        result = get_linkages(mock_group)

        assert result == [
            ("comp_a.y", "comp_b.x"),
            ("comp_a.w", "comp_c.z"),
        ]
        captured = capsys.readouterr()
        assert "Indexing used" in captured.out

    def test_empty_connections(self):
        """Edge case: no connections returns empty list."""
        mock_group = MagicMock()
        mock_group._manual_connections = {}

        result = get_linkages(mock_group)

        assert result == []
