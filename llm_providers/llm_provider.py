from abc import ABC, abstractmethod
from typing import Dict, Any

from classification.code_scanner import CodeFile


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def classify_code(self, code_file: CodeFile) -> Dict[str, Any]:
        """Classify code file and return semantic information"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        pass
