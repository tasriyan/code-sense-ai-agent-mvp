import json
from typing import Dict, Any, List

import requests

from classification.classification_result import ClassificationResult as cr
from llm_providers.llm_provider import LLMProvider
from llm_providers.utils.prompt_builders import PromptBuilder
from llm_providers.utils.response_validation import validate_implementation_response, get_fallback_implementation


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4 implementation for code classification"""

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self._api_key = api_key
        self._model = model
        self._base_url = "https://api.openai.com/v1/chat/completions"
        print(f"Initialized OpenAI provider with model: {model}")

    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        return f"OpenAI-{self._model}"

    def classify_code(self, code_file) -> Dict[str, Any]:
        """Classify code using OpenAI GPT-4"""

        if code_file.file_type == 'appsettings':
            messages = self._create_appsettings_classification_prompt(code_file)
        else:
            messages = self._create_classification_prompt(code_file)

        try:
            # Make API call to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }

            payload = {
                "model": self._model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.1,  # Low temperature for consistent structured output
                "response_format": {"type": "json_object"}  # Force JSON response
            }

            response = requests.post(
                self._base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            # Parse response
            openai_response = response.json()
            generated_text = openai_response["choices"][0]["message"]["content"]

            # Parse JSON from the generated text
            try:
                classification = json.loads(generated_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from text
                classification = self._extract_json_from_text(generated_text)

            # Validate and normalize the response
            return cr.normalize_classification_response(classification)

        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenAI API: {e}")
            return cr.get_fallback_response(code_file)
        except Exception as e:
            print(f"Error processing OpenAI response: {e}")
            return cr.get_fallback_response(code_file)

    def suggest_coding_implementation(self, user_request: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate implementation using OpenAI GPT-4"""

        system_message = "You are a senior software architect specializing in microservices implementation. Always respond with valid JSON only."

        pb = PromptBuilder()
        user_message = pb.create_implementation_prompt(user_request, context_docs)

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }

            payload = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 2000,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            openai_response = response.json()
            generated_text = openai_response["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                implementation = json.loads(generated_text)
            except json.JSONDecodeError:
                implementation = self._extract_json_from_text(generated_text)

            return validate_implementation_response(implementation)

        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return get_fallback_implementation(user_request)

    @staticmethod
    def _create_classification_prompt(code_file) -> List[Dict[str, str]]:
        """Create OpenAI chat messages for C# code analysis"""

        system_message = """You are a senior software architect specializing in business domain analysis. Your task is to analyze C# code from microservices and extract semantic business information.

Focus on understanding:
- What business problems the code solves
- What business rules are implemented  
- How the code fits into business workflows
- What business events trigger the code
- How it integrates with other business services

Always respond with valid JSON only, no additional text or explanations."""

        user_message = f"""Analyze this C# code file from a loyalty points microservice:

**File:** {code_file.relative_path}
**Project:** {code_file.project_name}

**Code:**
```csharp
{code_file.content[:4000]}
```

Extract business semantic information and return a JSON object with this exact structure:

```json
{{
    "business_purpose": "What business problem does this code solve?",
    "business_rules": ["Array", "of", "business rules implemented"],
    "business_triggers": ["Array", "of", "business events that trigger this code"],
    "business_data": ["Array", "of", "business data this code works with"],
    "integration_points": ["Array", "of", "service integrations"],
    "business_workflow": "Description of the business workflow this participates in",
    "technical_pattern": "Architectural pattern implemented",
    "confidence": 0.85
}}
```

Focus on business semantics, not technical implementation details. Return only the JSON object."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    @staticmethod
    def _create_appsettings_classification_prompt(code_file) -> List[Dict[str, str]]:
        """Create OpenAI chat messages for appsettings.json analysis"""

        system_message = """You are a senior software architect specializing in microservices integration analysis. Your task is to analyze configuration files and extract business integration information.

Focus on understanding:
- What business functionality the configurations enable
- What business processes are configured
- How services integrate to support business workflows
- What business data flows are configured

Always respond with valid JSON only, no additional text or explanations."""

        user_message = f"""Analyze this appsettings.json file from a loyalty points microservice:

**File:** {code_file.relative_path}
**Project:** {code_file.project_name}

**Configuration:**
```json
{code_file.content}
```

Extract business integration information and return a JSON object with this exact structure:

```json
{{
    "business_purpose": "What business functionality is enabled by these configurations?",
    "business_rules": ["Array", "of", "business rules configured"],
    "business_triggers": ["Array", "of", "business events configured"],
    "business_data": ["Array", "of", "business data flows configured"],
    "integration_points": ["Array", "of", "external services/systems configured"],
    "business_workflow": "Business workflow enabled by this configuration",
    "technical_pattern": "Integration patterns used",
    "confidence": 0.75
}}
```

Focus on business impact of these configurations. Return only the JSON object."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON from OpenAI response that might contain additional content"""
        try:
            # OpenAI with response_format should return pure JSON, but handle edge cases
            if text.strip().startswith('{') and text.strip().endswith('}'):
                return json.loads(text.strip())

            # Try to find JSON object in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return cr.get_empty_classification()

        except Exception:
            return cr.get_empty_classification()

        return normalized
