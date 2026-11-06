"""Validate public records and cache identities before model execution."""

from collections.abc import Mapping
from typing import Any

from .encoding import text_key


def validate_cache_key(value: str) -> None:
    """Require a canonical SHA-256 filename, never a relative or absolute path."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("Cache keys must be lowercase SHA-256 digests")


def validate_record(record: Mapping[str, Any]) -> None:
    """Check task semantics and exact token-to-cache binding without mutating input.

    Natural reading ignores any question field. Source corpus records may retain
    annotations, but only the documented model fields are passed to the network.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Each input must be a JSON object")
    words_key = text_key(record.get("words"))
    if record.get("task") not in ("NR", "IS"):
        raise ValueError(
            "task must be NR (natural reading) or IS (information seeking)"
        )
    if record.get("text_key") and record["text_key"] != words_key:
        raise ValueError("text_key does not match the supplied word sequence")
    if record["task"] == "IS":
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Information seeking requires a question visible before reading"
            )
        question_key = text_key(question.split())
        if record.get("question_key") and record["question_key"] != question_key:
            raise ValueError("question_key does not match the supplied question")
