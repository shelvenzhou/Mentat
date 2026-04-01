"""Metadata filtering for vector storage searches.

Provides a backend-agnostic filter representation that can be translated
to backend-specific query syntax (e.g. LanceDB SQL WHERE clauses).
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Union

# Only allow alphanumeric column names and underscores to prevent SQL injection
# via crafted field names in user-supplied filters.
_SAFE_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass
class MetadataFilter:
    """A single filter condition on a metadata field.

    Args:
        field: Column name (e.g. "source", "indexed_at", "file_type").
            Must match ``[a-zA-Z_][a-zA-Z0-9_]*`` to prevent SQL injection.
        op: Comparison operator. One of:
            "eq", "neq", "gt", "gte", "lt", "lte", "in", "like", "between".
        value: Comparison value. Scalar for most ops, list for "in" and "between".
    """

    field: str
    op: str
    value: Any

    def __post_init__(self):
        if not _SAFE_FIELD_RE.match(self.field):
            raise ValueError(
                f"Invalid filter field name '{self.field}': "
                f"must match [a-zA-Z_][a-zA-Z0-9_]* (no dots, spaces, or special chars)"
            )
        valid_ops = {"eq", "neq", "gt", "gte", "lt", "lte", "in", "like", "between"}
        if self.op not in valid_ops:
            raise ValueError(f"Invalid filter op '{self.op}', must be one of {valid_ops}")
        if self.op == "in" and not isinstance(self.value, (list, tuple, set)):
            raise ValueError(f"Filter op 'in' requires a list/tuple value, got {type(self.value)}")
        if self.op == "between":
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError("Filter op 'between' requires a 2-element list [low, high]")


@dataclass
class MetadataFilterSet:
    """Composite filter with AND/OR logic.

    Args:
        filters: List of MetadataFilter conditions.
        logic: How to combine filters. "AND" (all must match) or "OR" (any must match).
    """

    filters: List[MetadataFilter] = field(default_factory=list)
    logic: str = "AND"

    def __post_init__(self):
        if self.logic not in ("AND", "OR"):
            raise ValueError(f"Invalid logic '{self.logic}', must be 'AND' or 'OR'")

    def is_empty(self) -> bool:
        """Return True if no filters are defined."""
        return len(self.filters) == 0

    def to_lance_sql(self) -> str:
        """Convert to a LanceDB-compatible SQL WHERE clause string.

        Returns empty string if no filters are defined.
        """
        if not self.filters:
            return ""

        clauses = [_filter_to_lance_sql(f) for f in self.filters]
        joiner = f" {self.logic} "
        return joiner.join(clauses)


def _escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL string values."""
    return value.replace("'", "''")


def _filter_to_lance_sql(f: MetadataFilter) -> str:
    """Convert a single MetadataFilter to a LanceDB SQL clause."""
    field_name = f.field

    if f.op == "eq":
        if isinstance(f.value, str):
            return f"{field_name} = '{_escape_sql_string(f.value)}'"
        return f"{field_name} = {f.value}"

    elif f.op == "neq":
        if isinstance(f.value, str):
            return f"{field_name} != '{_escape_sql_string(f.value)}'"
        return f"{field_name} != {f.value}"

    elif f.op == "gt":
        return f"{field_name} > {f.value}"

    elif f.op == "gte":
        return f"{field_name} >= {f.value}"

    elif f.op == "lt":
        return f"{field_name} < {f.value}"

    elif f.op == "lte":
        return f"{field_name} <= {f.value}"

    elif f.op == "in":
        if all(isinstance(v, str) for v in f.value):
            vals = ", ".join(f"'{_escape_sql_string(v)}'" for v in f.value)
        else:
            vals = ", ".join(str(v) for v in f.value)
        return f"{field_name} IN ({vals})"

    elif f.op == "like":
        return f"{field_name} LIKE '{_escape_sql_string(f.value)}'"

    elif f.op == "between":
        return f"{field_name} BETWEEN {f.value[0]} AND {f.value[1]}"

    else:
        raise ValueError(f"Unsupported filter op: {f.op}")
