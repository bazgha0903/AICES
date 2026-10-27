"""Small text helpers shared by all modules."""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text):
    """Splits text into sentences and drops tiny fragments (<3 chars)."""
    text = normalize_space(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if len(p) > 2]
