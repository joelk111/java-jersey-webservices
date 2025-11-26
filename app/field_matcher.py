"""
Field matching module using fuzzy string matching.

This module provides functionality to match user-provided field names
to the canonical field names in the ORBIS data dictionary.
"""

import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    from rapidfuzz import fuzz, process
except ImportError:
    # Fallback to basic matching if rapidfuzz not installed
    fuzz = None
    process = None

from .config import config


class FieldMatcher:
    """
    Matches user-provided field names to ORBIS data dictionary fields
    using fuzzy string matching.
    """

    def __init__(self, field_dictionary_path: Optional[str] = None):
        """
        Initialize the field matcher.

        Args:
            field_dictionary_path: Path to the field dictionary file.
                                   Defaults to config.field_dictionary_path.
        """
        self.path = field_dictionary_path or config.field_dictionary_path
        self.fields: List[str] = []
        self.field_labels: Dict[str, str] = {}
        self.field_types: Dict[str, str] = {}
        self._load_fields()

    def _load_fields(self):
        """Load fields from the C-style header file."""
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Field dictionary not found: {self.path}")

        with open(path, 'r') as f:
            content = f.read()

        # Parse lines like: "table.field",  // Label | TYPE
        pattern = r'"([^"]+)",\s*//\s*([^|]*)\|\s*(\w+)'

        for match in re.finditer(pattern, content):
            field_name = match.group(1)
            label = match.group(2).strip()
            data_type = match.group(3).strip()

            self.fields.append(field_name)
            self.field_labels[field_name] = label
            self.field_types[field_name] = data_type

    def match(
        self,
        query: str,
        limit: int = 5,
        threshold: Optional[int] = None
    ) -> List[Tuple[str, int, str]]:
        """
        Find best matching fields for a query string.

        Args:
            query: The field name or description to match
            limit: Maximum number of results to return
            threshold: Minimum match score (0-100). Defaults to config.fuzzy_threshold.

        Returns:
            List of (field_name, score, label) tuples, sorted by score descending.
        """
        if threshold is None:
            threshold = config.fuzzy_threshold

        # Normalize query
        query_normalized = query.lower().replace(" ", "_").replace("-", "_")

        results = []

        for field in self.fields:
            label = self.field_labels.get(field, "")

            # Try multiple matching strategies
            scores = []

            # Extract just the field name (after the dot)
            field_name_only = field.split('.')[-1] if '.' in field else field

            if fuzz:
                # Use rapidfuzz for better matching
                scores.append(fuzz.ratio(query_normalized, field_name_only.lower()))
                scores.append(fuzz.ratio(query_normalized, field.lower()))
                scores.append(fuzz.partial_ratio(query_normalized, field.lower()))
                scores.append(fuzz.partial_ratio(query_normalized, label.lower()))
                scores.append(fuzz.token_set_ratio(query_normalized, f"{field} {label}".lower()))
            else:
                # Fallback to simple substring matching
                if query_normalized in field.lower():
                    scores.append(80)
                elif query_normalized in label.lower():
                    scores.append(70)
                else:
                    scores.append(0)

            best_score = max(scores)
            if best_score >= threshold:
                results.append((field, best_score, label))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def match_multiple(
        self,
        queries: List[str]
    ) -> Dict[str, List[Tuple[str, int, str]]]:
        """
        Match multiple field queries at once.

        Args:
            queries: List of field names or descriptions to match

        Returns:
            Dictionary mapping each query to its match results.
        """
        return {query: self.match(query) for query in queries}

    def get_field_info(self, field_name: str) -> Optional[Dict[str, str]]:
        """
        Get information about a specific field.

        Args:
            field_name: The fully qualified field name (table.field)

        Returns:
            Dictionary with field_name, label, and data_type, or None if not found.
        """
        if field_name in self.fields:
            return {
                "field_name": field_name,
                "label": self.field_labels.get(field_name, ""),
                "data_type": self.field_types.get(field_name, "STRING")
            }
        return None

    def search_by_table(self, table_name: str) -> List[str]:
        """
        Get all fields for a specific table.

        Args:
            table_name: The table name to search for

        Returns:
            List of field names belonging to that table.
        """
        table_prefix = f"{table_name}."
        return [f for f in self.fields if f.startswith(table_prefix)]

    def list_tables(self) -> List[str]:
        """
        Get list of all table names.

        Returns:
            Sorted list of unique table names.
        """
        tables = set()
        for field in self.fields:
            if '.' in field:
                tables.add(field.split('.')[0])
        return sorted(tables)

    def get_fields_by_type(self, data_type: str) -> List[str]:
        """
        Get all fields of a specific data type.

        Args:
            data_type: The data type to filter by (e.g., "STRING", "INT")

        Returns:
            List of field names with that data type.
        """
        return [
            field for field, dtype in self.field_types.items()
            if dtype.upper() == data_type.upper()
        ]
