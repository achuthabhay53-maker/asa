"""Small helpers shared by agent modules that previously duplicated them."""
from __future__ import annotations

import json
import re
from typing import Any


def field_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an attribute-style object."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a Claude reply, tolerating Markdown fences and surrounding text."""
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Claude did not return a valid JSON object.")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from Claude.")
    return data
