from dataclasses import dataclass
from typing import List, Dict, Any

from classification.code_scanner import CodeFile


def _ensure_list(value: Any) -> List[str]:
    """Ensure value is a list of strings"""
    if isinstance(value, list):
        return [str(item) for item in value]
    elif isinstance(value, str):
        # If it's a string, try to split it or return as single item
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        else:
            return [value] if value else []
    else:
        return []

@dataclass
class ClassificationResult:
    """Represents the semantic classification result"""
    file_path: str
    project_name: str
    file_type: str
    business_purpose: str
    business_rules: List[str]
    business_triggers: List[str]
    business_data: List[str]
    integration_points: List[str]
    business_workflow: str
    technical_pattern: str
    code_snippet: str
    llm_provider: str
    classification_confidence: float

    @staticmethod
    def create_classification_result(code_file: CodeFile,
                                      classification: Dict[str, Any],
                                      provider_name: str):
        """Create ClassificationResult from LLM response"""
        return ClassificationResult(
            file_path=code_file.file_path,
            project_name=code_file.project_name,
            file_type=code_file.file_type,
            business_purpose=classification.get('business_purpose', ''),
            business_rules=classification.get('business_rules', []),
            business_triggers=classification.get('business_triggers', []),
            business_data=classification.get('business_data', []),
            integration_points=classification.get('integration_points', []),
            business_workflow=classification.get('business_workflow', ''),
            technical_pattern=classification.get('technical_pattern', ''),
            code_snippet=code_file.content[:500],  # First 500 chars
            llm_provider=provider_name,
            classification_confidence=classification.get('confidence', 0.0)
        )

    @staticmethod
    def get_empty_classification() -> Dict[str, Any]:
        return {
            "business_purpose": "Unable to analyze",
            "business_rules": [],
            "business_triggers": [],
            "business_data": [],
            "integration_points": [],
            "business_workflow": "Unknown",
            "technical_pattern": "Unknown",
            "confidence": 0.0
        }

    @staticmethod
    def get_fallback_response(code_file) -> Dict[str, Any]:
        """Fallback response when API fails"""
        return {
            "business_purpose": f"CodeLlama analysis failed for {code_file.project_name}",
            "business_rules": ["API call failed - check Ollama service"],
            "business_triggers": ["Unknown due to API failure"],
            "business_data": ["Unable to determine"],
            "integration_points": ["Analysis incomplete"],
            "business_workflow": "Could not analyze due to API error",
            "technical_pattern": "Unknown",
            "confidence": 0.0
        }

    @staticmethod
    def normalize_classification_response(classification: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate classification response"""
        normalized = {
            "business_purpose": str(classification.get('business_purpose', 'Unknown')),
            "business_rules": _ensure_list(classification.get('business_rules', [])),
            "business_triggers": _ensure_list(classification.get('business_triggers', [])),
            "business_data": _ensure_list(classification.get('business_data', [])),
            "integration_points": _ensure_list(classification.get('integration_points', [])),
            "business_workflow": str(classification.get('business_workflow', 'Unknown')),
            "technical_pattern": str(classification.get('technical_pattern', 'Unknown')),
            "confidence": float(classification.get('confidence', 0.5))
        }

        # Ensure confidence is between 0 and 1
        normalized['confidence'] = max(0.0, min(1.0, normalized['confidence']))

        return normalized