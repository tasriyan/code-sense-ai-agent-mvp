from abc import ABC, abstractmethod
from typing import Dict, Any, List

from classification.code_scanner import CodeFile

class LLMClassifier(ABC):
    """Abstract base class for LLM providers used in code classification."""

    @abstractmethod
    def classify_code(self, code_file: CodeFile) -> Dict[str, Any]:
        """Classify code file and return semantic information"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        pass

    @staticmethod
    def _get_empty_classification() -> Dict[str, Any]:
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
    def _get_fallback_response(code_file) -> Dict[str, Any]:
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
    def _normalize_classification_response(classification: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate classification response"""
        normalized = {
            "business_purpose": str(classification.get('business_purpose', 'Unknown')),
            "business_rules": LLMClassifier._ensure_list(classification.get('business_rules', [])),
            "business_triggers": LLMClassifier._ensure_list(classification.get('business_triggers', [])),
            "business_data": LLMClassifier._ensure_list(classification.get('business_data', [])),
            "integration_points": LLMClassifier._ensure_list(classification.get('integration_points', [])),
            "business_workflow": str(classification.get('business_workflow', 'Unknown')),
            "technical_pattern": str(classification.get('technical_pattern', 'Unknown')),
            "confidence": float(classification.get('confidence', 0.5))
        }

        # Ensure confidence is between 0 and 1
        normalized['confidence'] = max(0.0, min(1.0, normalized['confidence']))

        return normalized

    @staticmethod
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


class LLMExecutor(ABC):
    """Abstract base class for LLM providers used to provide code implementations"""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        pass

    @abstractmethod
    def suggest_coding_implementation(self, user_request: str,
                                      context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer from code file for provided query"""
        pass

    @staticmethod
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

class LLMOrchestrator(ABC):
    """Abstract base class for LLM providers used in multi-agent orchestration"""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        pass

