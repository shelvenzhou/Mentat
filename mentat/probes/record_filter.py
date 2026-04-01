"""In-memory record filtering with dot-path field resolution.

Mirrors ``storage.filters.MetadataFilter`` but operates on Python dicts
instead of generating SQL.  Designed for probe-level row filtering (JSONL
lines, CSV rows, etc.) where records are already parsed into dicts.

Dot-path resolution supports nested dicts and list-of-dicts traversal::

    _resolve_dot_path({"a": {"b": [{"c": 1}, {"c": 2}]}}, "a.b.c")
    # → [1, 2]

When a path segment encounters a list of dicts, it descends into each
element and collects matching values.  This enables paths like
``message.content.text`` where ``content`` is an array of content blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List


_MISSING = object()

_VALID_OPS = frozenset(
    {"eq", "neq", "in", "nin", "gt", "gte", "lt", "lte", "exists", "regex", "contains"}
)

# Path segments matching these patterns are rejected to prevent
# accidental (or malicious) access to Python internals.
_FORBIDDEN_PART_RE = re.compile(r"^__.*__$")


def _validate_path_parts(parts: list[str]) -> None:
    """Raise ValueError if any path segment looks like a Python dunder."""
    for part in parts:
        if _FORBIDDEN_PART_RE.match(part):
            raise ValueError(
                f"Dot-path segment '{part}' looks like a Python internal attribute "
                f"and is not allowed"
            )
        if not part:
            raise ValueError("Dot-path contains an empty segment")


def _resolve_dot_path(obj: Any, path: str) -> Any:
    """Resolve a dot-separated path against a nested dict.

    Returns ``_MISSING`` if any segment fails to resolve.

    When a path segment hits a list of dicts, it descends into each
    element and returns a flat list of resolved values.

    Raises ``ValueError`` for paths containing dunder segments.
    """
    parts = path.split(".")
    _validate_path_parts(parts)

    current: Any = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, _MISSING)
        elif isinstance(current, list):
            results = []
            for item in current:
                if isinstance(item, dict):
                    val = item.get(part, _MISSING)
                    if val is not _MISSING:
                        if isinstance(val, list):
                            results.extend(val)
                        else:
                            results.append(val)
            current = results if results else _MISSING
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _compare(op: str, val: Any, target: Any) -> bool:
    """Evaluate a single comparison."""
    if op == "eq":
        return val == target
    elif op == "neq":
        return val != target
    elif op == "in":
        return val in target
    elif op == "nin":
        return val not in target
    elif op == "gt":
        return val > target
    elif op == "gte":
        return val >= target
    elif op == "lt":
        return val < target
    elif op == "lte":
        return val <= target
    elif op == "regex":
        return bool(re.search(target, str(val)))
    elif op == "contains":
        if isinstance(val, str):
            return target in val
        if isinstance(val, list):
            return target in val
        return False
    else:
        raise ValueError(f"Unknown op: {op}")


def _eval_filter(f: RecordFilter, record: dict) -> bool:
    """Evaluate a single filter against a record."""
    val = _resolve_dot_path(record, f.field)

    # "exists" checks presence, not value
    if f.op == "exists":
        found = val is not _MISSING
        return found == bool(f.value if f.value is not None else True)

    if val is _MISSING:
        return False

    # When dot-path resolved to a list (from list-of-dicts traversal),
    # the filter matches if ANY element satisfies the condition.
    if isinstance(val, list) and f.op not in ("in", "nin", "contains"):
        return any(_compare(f.op, v, f.value) for v in val)

    return _compare(f.op, val, f.value)


@dataclass
class RecordFilter:
    """A single filter condition on a record field.

    Args:
        field: Dot-path into the record (e.g. ``"type"``, ``"message.role"``).
            Dunder segments (``__xxx__``) are rejected for safety.
        op: Comparison operator — one of:
            ``eq``, ``neq``, ``in``, ``nin``, ``gt``, ``gte``, ``lt``, ``lte``,
            ``exists``, ``regex``, ``contains``.
        value: Comparison value.  Scalar for most ops, list for ``in``/``nin``.
    """

    field: str
    op: str
    value: Any = None

    def __post_init__(self):
        if self.op not in _VALID_OPS:
            raise ValueError(
                f"Invalid filter op '{self.op}', must be one of {sorted(_VALID_OPS)}"
            )
        if self.op in ("in", "nin") and not isinstance(self.value, (list, tuple, set)):
            raise ValueError(
                f"Filter op '{self.op}' requires a list/tuple/set value, got {type(self.value)}"
            )
        # Validate path eagerly so config errors surface at construction time
        _validate_path_parts(self.field.split("."))

    def match(self, record: dict) -> bool:
        """Test whether *record* satisfies this filter."""
        return _eval_filter(self, record)


@dataclass
class RecordFilterSet:
    """Composite filter with AND/OR logic.

    Args:
        filters: List of :class:`RecordFilter` conditions.
        logic: ``"AND"`` (all must match) or ``"OR"`` (any must match).
    """

    filters: List[RecordFilter] = field(default_factory=list)
    logic: str = "AND"

    def __post_init__(self):
        if self.logic not in ("AND", "OR"):
            raise ValueError(f"Invalid logic '{self.logic}', must be 'AND' or 'OR'")

    def is_empty(self) -> bool:
        return len(self.filters) == 0

    def match(self, record: dict) -> bool:
        """Test whether *record* satisfies the filter set."""
        if not self.filters:
            return True
        results = [_eval_filter(f, record) for f in self.filters]
        if self.logic == "AND":
            return all(results)
        return any(results)

    @classmethod
    def from_dicts(cls, raw: list[dict], logic: str = "AND") -> "RecordFilterSet":
        """Build from a list of plain dicts (e.g. deserialized JSON config).

        Each dict must have ``field``, ``op``, and optionally ``value`` keys.
        """
        filters = [RecordFilter(**d) for d in raw]
        return cls(filters=filters, logic=logic)
