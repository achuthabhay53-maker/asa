"""Helpers for schema-constrained Anthropic tool responses."""

from __future__ import annotations

from typing import Any


def model_input_schema(model: type[Any]) -> dict[str, Any]:
    """Build an Anthropic-compatible schema with local Pydantic references expanded."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(value: Any, schema_node: bool = False) -> Any:
        if isinstance(value, list):
            return [expand(item, schema_node) for item in value]
        if not isinstance(value, dict):
            return value

        reference = value.get("$ref")
        if reference:
            definition_name = reference.rsplit("/", 1)[-1]
            if definition_name not in definitions:
                raise ValueError(f"Unknown schema reference: {reference}")
            expanded = expand(definitions[definition_name], schema_node=True)
            overrides = {
                key: expand(item, schema_node=True)
                for key, item in value.items()
                if key != "$ref"
            }
            return {**expanded, **overrides}

        result = {}
        for key, item in value.items():
            if schema_node and key in {"title", "description", "default"}:
                continue
            if key == "properties" and isinstance(item, dict):
                result[key] = {
                    name: expand(property_schema, schema_node=True)
                    for name, property_schema in item.items()
                }
            elif key in {"items", "anyOf", "allOf", "oneOf"}:
                result[key] = expand(item, schema_node=True)
            else:
                result[key] = expand(item)
        if schema_node and result.get("type") == "object":
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
        return result

    return expand(schema, schema_node=True)


def extract_tool_input(response: Any, expected_name: str) -> dict[str, Any]:
    """Require one named Anthropic tool-use block and return its object input."""
    content = response.get("content", []) if isinstance(response, dict) else getattr(response, "content", [])
    tool_blocks = [
        block
        for block in content
        if _value(block, "type") == "tool_use"
    ]
    if len(tool_blocks) != 1:
        raise RuntimeError(
            f"Expected exactly one {expected_name} structured response; received {len(tool_blocks)}."
        )

    block = tool_blocks[0]
    name = _value(block, "name")
    if name != expected_name:
        raise RuntimeError(f"Expected structured response {expected_name!r}; received {name!r}.")

    payload = _value(block, "input")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Structured response {expected_name!r} input must be an object.")
    return payload


def _value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
