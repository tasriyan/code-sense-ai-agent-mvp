import json
from typing import Dict, Any


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text response"""
    try:
        # Try to find JSON in text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return _get_empty_implementation()
    except:
        return _get_empty_implementation()


def _get_empty_implementation() -> Dict[str, Any]:
    """Return empty implementation structure"""
    return {
        "suggested_service": "Unknown",
        "suggested_files": [],
        "implementation_steps": ["Unable to generate implementation"],
        "business_rationale": "Could not analyze request",
        "integration_points": [],
        "code_examples": [],
        "confidence_score": 0.0
    }