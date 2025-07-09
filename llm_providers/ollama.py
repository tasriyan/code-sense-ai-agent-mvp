import json
import requests
from typing import Dict, Any

from classification.classification_result import ClassificationResult as cr
from llm_providers.llm_provider import LLMProvider

class OllamaProvider(LLMProvider):
    """Ollama local LLM implementation for CodeLlama"""

    def __init__(self, model: str = "codellama:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        print(f"Initialized Ollama provider with model: {model}")

    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        return f"Ollama-{self.model}"

    def classify_code(self, code_file) -> Dict[str, Any]:
        """Classify code using Ollama local LLM"""

        if code_file.file_type == 'appsettings':
            prompt = self._create_appsettings_prompt(code_file)
        else:
            prompt = self._create_code_prompt(code_file)

        try:
            # Make API call to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()

            # Parse response
            ollama_response = response.json()
            generated_text = ollama_response.get('response', '')

            # Parse JSON from the generated text
            try:
                classification = json.loads(generated_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from text
                classification = self._extract_json_from_text(generated_text)

            # Validate and normalize the response
            return cr.normalize_classification_response(classification)

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            # Fallback to basic response if API fails
            return cr.get_fallback_response(code_file)
        except Exception as e:
            print(f"Error processing Ollama response: {e}")
            return cr.get_fallback_response(code_file)

    @staticmethod
    def _create_code_prompt(code_file) -> str:
        """Create prompt for C# code analysis"""
        return f"""You are a senior software architect analyzing a C# code file from a loyalty points microservice.

File: {code_file.relative_path}
Project: {code_file.project_name}

Code:
```csharp
{code_file.content[:3000]}
```

Analyze this code and extract business semantic information. Return ONLY a valid JSON object with this exact structure:

{{
    "business_purpose": "What business problem does this code solve?",
    "business_rules": ["List of business rules this implements"],
    "business_triggers": ["List of business events that cause this code to execute"],
    "business_data": ["List of business data this code works with"],
    "integration_points": ["List of how this integrates with other services"],
    "business_workflow": "Describe the business workflow this participates in",
    "technical_pattern": "What architectural pattern does this implement?",
    "confidence": 0.85
}}

Focus on business semantics, not technical implementation details. Return only the JSON, no additional text."""

    @staticmethod
    def _create_appsettings_prompt(code_file) -> str:
        """Create prompt for appsettings.json analysis"""
        return f"""You are a senior software architect analyzing an appsettings.json file from a loyalty points microservice.

File: {code_file.relative_path}
Project: {code_file.project_name}

Configuration:
```json
{code_file.content}
```

Analyze this configuration and extract business integration information. Return ONLY a valid JSON object with this exact structure:

{{
    "business_purpose": "What business functionality is enabled by these configurations?",
    "business_rules": ["List of business rules configured here"],
    "business_triggers": ["List of business events configured"],
    "business_data": ["List of business data flows configured"],
    "integration_points": ["List of external services or systems configured"],
    "business_workflow": "Describe the business workflow enabled by this configuration",
    "technical_pattern": "What integration patterns are used?",
    "confidence": 0.75
}}

Focus on business impact of these configurations. Return only the JSON, no additional text."""

    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON from text that might contain additional content"""
        try:
            # Try to find JSON block in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, return empty structure
                return cr.get_empty_classification()

        except Exception:
            return cr.get_empty_classification()