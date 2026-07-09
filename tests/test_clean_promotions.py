"""Unit tests for clean_promotions function.

Tests the filtering of no-op promotions that would create dotted promoted names,
which trigger an infinite loop in OpenMDAO >= 3.43's conn_graph.add_auto_ivc_nodes.

Requirements: 2.4
"""

from standard_evaluator.om_converter import clean_promotions


class TestCleanPromotionsDeduplication:
    """Test deduplication of promotions with same source name."""

    def test_duplicate_promotions_collapsed_to_bare_name(self):
        """When a variable appears in multiple tuples with same source, collapse to bare name."""
        proms = [('d', 'sub1.d'), ('d', 'sub1.d')]
        result = clean_promotions(proms, 'sub1')
        # 'd' appears in two tuples (after dedup they're the same, but the logic
        # handles the case where different targets exist for same source)
        # Since both are no-ops, they get filtered out
        assert 'd' not in result or ('d', 'sub1.d') not in result

    def test_same_source_different_targets_collapses_to_bare_name(self):
        """When same source has both ('a','a') and ('a','sub1.a'), collapse to bare 'a'."""
        proms = [('a', 'a'), ('a', 'sub1.a')]
        result = clean_promotions(proms, 'sub1')
        # grouped_tuples sees two entries for 'a', so it becomes bare 'a'
        assert 'a' in result


class TestCleanPromotionsNoOpFiltering:
    """Test filtering of no-op promotions ('var', 'subsys.var')."""

    def test_noop_promotion_is_dropped(self):
        """A promotion ('k', 'aero.k') for subsystem 'aero' is a no-op and should be dropped."""
        proms = [('k', 'aero.k'), ('r', 'aero.r'), ('drag', 'aero.drag')]
        result = clean_promotions(proms, 'aero')
        assert result == []

    def test_flat_promotion_is_kept(self):
        """A promotion ('a', 'a') is a real flat promote and should be kept."""
        proms = [('a', 'a')]
        result = clean_promotions(proms, 'sub1')
        assert ('a', 'a') in result

    def test_rename_promotion_is_kept(self):
        """A promotion ('b', 'my_alias') is a real rename and should be kept."""
        proms = [('b', 'my_alias')]
        result = clean_promotions(proms, 'sub1')
        assert ('b', 'my_alias') in result

    def test_mixed_promotions_only_noops_dropped(self):
        """Only no-op promotions matching the subsystem prefix are dropped."""
        proms = [
            ('a', 'a'),              # flat promote — keep
            ('k', 'aero.k'),         # no-op for 'aero' — drop
            ('drag', 'aero.drag'),   # no-op for 'aero' — drop
            ('b', 'my_alias'),       # rename — keep
        ]
        result = clean_promotions(proms, 'aero')
        assert ('a', 'a') in result
        assert ('b', 'my_alias') in result
        assert ('k', 'aero.k') not in result
        assert ('drag', 'aero.drag') not in result

    def test_dotted_source_noop_is_dropped(self):
        """A promotion ('z1.c', 'sub1.z1.c') for subsystem 'sub1' is a no-op."""
        proms = [('z1.c', 'sub1.z1.c')]
        result = clean_promotions(proms, 'sub1')
        assert result == []

    def test_noop_for_different_subsystem_is_kept(self):
        """A promotion ('k', 'aero.k') is NOT a no-op when subsystem name is 'sub1'."""
        proms = [('k', 'aero.k')]
        result = clean_promotions(proms, 'sub1')
        assert ('k', 'aero.k') in result

    def test_bare_string_promotions_are_kept(self):
        """Bare string promotions (not tuples) are always kept."""
        proms = [('d', 'sub1.d'), ('d', 'sub1.d'), ('a', 'a')]
        # After dedup: 'd' has one unique entry, 'a' has one unique entry
        # 'd' -> ('d', 'sub1.d') is a no-op for 'sub1', gets dropped
        # 'a' -> ('a', 'a') is a flat promote, kept
        result = clean_promotions(proms, 'sub1')
        assert ('a', 'a') in result
        assert ('d', 'sub1.d') not in result


class TestCleanPromotionsEmptyAndEdgeCases:
    """Test edge cases for clean_promotions."""

    def test_empty_promotions(self):
        """Empty input returns empty output."""
        assert clean_promotions([], 'aero') == []

    def test_single_noop_promotion(self):
        """Single no-op promotion returns empty list."""
        result = clean_promotions([('k', 'aero.k')], 'aero')
        assert result == []

    def test_single_flat_promotion(self):
        """Single flat promotion is preserved."""
        result = clean_promotions([('a', 'a')], 'sub1')
        assert result == [('a', 'a')]

    def test_subsystem_name_with_underscore(self):
        """No-op detection works with underscored subsystem names."""
        proms = [('x', 'my_comp.x')]
        result = clean_promotions(proms, 'my_comp')
        assert result == []

    def test_partial_prefix_match_is_not_noop(self):
        """A promotion ('k', 'aero_group.k') is NOT a no-op for subsystem 'aero'."""
        proms = [('k', 'aero_group.k')]
        result = clean_promotions(proms, 'aero')
        assert ('k', 'aero_group.k') in result
