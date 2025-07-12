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
            llm_provider=provider_name,
            classification_confidence=classification.get('confidence', 0.0)
        )