"""
Unit tests for the FieldMatcher module.
Tests fuzzy matching of field names against the ORBIS data dictionary.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.field_matcher import FieldMatcher


@pytest.fixture
def matcher():
    """Create a FieldMatcher instance for testing."""
    return FieldMatcher()


class TestFieldMatcherBasic:
    """Basic field matching tests."""

    def test_exact_match(self, matcher):
        """Test exact field name matching."""
        results = matcher.match("bvd_id_number")
        assert len(results) > 0
        # Should find fields containing bvd_id_number
        assert any("bvd_id_number" in r[0] for r in results)

    def test_partial_match(self, matcher):
        """Test partial field name matching."""
        results = matcher.match("bvd_id")
        assert len(results) > 0
        # Should match bvd_id_number
        assert any("bvd" in r[0].lower() for r in results)

    def test_fuzzy_match_underscore(self, matcher):
        """Test matching with underscores replaced by spaces."""
        results = matcher.match("bvd id number")
        assert len(results) > 0

    def test_case_insensitive(self, matcher):
        """Test case insensitive matching."""
        results_lower = matcher.match("bvd_id_number")
        results_upper = matcher.match("BVD_ID_NUMBER")
        # Both should return results
        assert len(results_lower) > 0
        assert len(results_upper) > 0


class TestFieldMatcherTypos:
    """Test handling of typos and misspellings."""

    def test_typo_single_char(self, matcher):
        """Test matching with single character typo."""
        results = matcher.match("bvd_id_numbre")  # typo: numbre
        assert len(results) > 0
        # Score should still be decent
        if results:
            assert results[0][1] >= 70

    def test_typo_missing_underscore(self, matcher):
        """Test matching without underscores."""
        results = matcher.match("bvdidnumber")
        # May or may not match depending on threshold
        # Just check it doesn't crash

    def test_typo_extra_char(self, matcher):
        """Test matching with extra character."""
        results = matcher.match("bvd_id_numberr")
        assert len(results) > 0


class TestFieldMatcherMultiple:
    """Test matching multiple fields at once."""

    def test_match_multiple(self, matcher):
        """Test matching multiple queries."""
        results = matcher.match_multiple(["bvd_id", "email", "url"])
        assert len(results) == 3
        assert "bvd_id" in results
        assert "email" in results
        assert "url" in results

    def test_match_multiple_empty(self, matcher):
        """Test with empty list."""
        results = matcher.match_multiple([])
        assert len(results) == 0


class TestFieldMatcherTables:
    """Test table-related functionality."""

    def test_list_tables(self, matcher):
        """Test listing all tables."""
        tables = matcher.list_tables()
        assert len(tables) > 0
        # Should include known tables
        assert "acnc" in tables or len(tables) > 50

    def test_search_by_table(self, matcher):
        """Test searching fields by table name."""
        fields = matcher.search_by_table("acnc")
        if fields:  # Table may not exist in all datasets
            assert all(f.startswith("acnc.") for f in fields)

    def test_search_by_table_nonexistent(self, matcher):
        """Test searching non-existent table."""
        fields = matcher.search_by_table("nonexistent_table_xyz")
        assert len(fields) == 0


class TestFieldMatcherInfo:
    """Test field information retrieval."""

    def test_get_field_info(self, matcher):
        """Test getting field information."""
        # First find a valid field
        results = matcher.match("bvd_id_number", limit=1)
        if results:
            field_name = results[0][0]
            info = matcher.get_field_info(field_name)
            assert info is not None
            assert "field_name" in info
            assert "data_type" in info

    def test_get_field_info_nonexistent(self, matcher):
        """Test getting info for non-existent field."""
        info = matcher.get_field_info("nonexistent.field")
        assert info is None


class TestFieldMatcherScoring:
    """Test matching score behavior."""

    def test_score_range(self, matcher):
        """Test that scores are in valid range."""
        results = matcher.match("bvd", limit=10)
        for _, score, _ in results:
            assert 0 <= score <= 100

    def test_scores_descending(self, matcher):
        """Test that results are sorted by score descending."""
        results = matcher.match("email", limit=10)
        if len(results) > 1:
            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_threshold_filtering(self, matcher):
        """Test that low scores are filtered out."""
        results = matcher.match("xyz_abc_123", limit=10, threshold=90)
        # Very unlikely to find high-score matches for garbage input
        for _, score, _ in results:
            assert score >= 90
