import json
from typing import Any, Dict


def safe_json_parse(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from text, handling common edge cases."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nText: {text}")
