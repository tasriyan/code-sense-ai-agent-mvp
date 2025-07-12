import json
from typing import Dict, Any

import requests

from llms.providers import LLMClassifier, LLMRecommender
from llms.utils.validation import get_validated_answer, get_fallback_answer
from shared.context_providers import BusinessContextProvider

class AnthropicClassifier(LLMClassifier):
    """Anthropic Claude implementation for code classification"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self._api_key = api_key
        self._model = model
        self._base_url = "https://api.anthropic.com/v1/messages"
        print(f"Initialized Anthropic provider with model: {model}")

    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        return f"Anthropic-{self._model}"

    def classify_code(self, code_file) -> Dict[str, Any]:
        """Classify code using Anthropic Claude"""

        if code_file.file_type == 'appsettings':
            prompt = self._create_appsettings_classification_prompt(code_file)
        else:
            prompt = self._create_classification_prompt(code_file)

        try:
            # Make API call to Anthropic
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01"
            }

            payload = {
                "model": self._model,
                "max_tokens": 1000,
                "temperature": 0.1,  # Low temperature for consistent structured output
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = requests.post(
                self._base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            # Parse response
            anthropic_response = response.json()
            generated_text = anthropic_response["content"][0]["text"]

            # Parse JSON from the generated text
            try:
                classification = json.loads(generated_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from text
                classification = self._extract_json_from_text(generated_text)

            # Validate and normalize the response
            return self._normalize_classification_response(classification)

        except requests.exceptions.RequestException as e:
            print(f"Error calling Anthropic API: {e}")
            return self._get_fallback_response(code_file)
        except Exception as e:
            print(f"Error processing Anthropic response: {e}")
            return self._get_fallback_response(code_file)

    @staticmethod
    def _create_classification_prompt(code_file) -> str:
        """Create Claude-optimized prompt for C# code analysis"""
        return f"""I need you to analyze a C# code file from a loyalty points microservice and extract business semantic information.

<file_context>
File: {code_file.relative_path}
Project: {code_file.project_name}
</file_context>

<code>
{code_file.content[:4000]}
</code>

Please analyze this code and extract business semantic information. I need you to think step-by-step about:

1. What business problem this code solves
2. What business rules are implemented
3. What business events trigger this code
4. What business data it works with
5. How it integrates with other services
6. The business workflow it participates in
7. The architectural pattern it implements

Return your analysis as a JSON object with this exact structure:

```json
{{
    "business_purpose": "Clear description of what business problem this solves",
    "business_rules": ["List", "of", "business rules implemented"],
    "business_triggers": ["List", "of", "business events that trigger this code"],
    "business_data": ["List", "of", "business data this works with"],
    "integration_points": ["List", "of", "service integrations"],
    "business_workflow": "Description of the business workflow this participates in",
    "technical_pattern": "Architectural pattern implemented",
    "confidence": 0.85
}}
```

Focus on the business semantics and domain logic, not just technical implementation details. Return only the JSON object, no additional explanation."""

    @staticmethod
    def _create_appsettings_classification_prompt(code_file) -> str:
        """Create Claude-optimized prompt for appsettings.json analysis"""
        return f"""I need you to analyze an appsettings.json configuration file from a loyalty points microservice.

<file_context>
File: {code_file.relative_path}
Project: {code_file.project_name}
</file_context>

<configuration>
{code_file.content}
</configuration>

Please analyze this configuration file and extract business integration information. Think about:

1. What business functionality these configurations enable
2. What business rules are configured here
3. What business events are configured
4. What business data flows are set up
5. What external services or systems are integrated
6. What business workflows are enabled
7. What integration patterns are used

Return your analysis as a JSON object with this exact structure:

```json
{{
    "business_purpose": "What business functionality is enabled by these configurations",
    "business_rules": ["List", "of", "business rules configured"],
    "business_triggers": ["List", "of", "business events configured"],
    "business_data": ["List", "of", "business data flows configured"],
    "integration_points": ["List", "of", "external services/systems configured"],
    "business_workflow": "Business workflow enabled by this configuration",
    "technical_pattern": "Integration patterns used",
    "confidence": 0.75
}}
```

Focus on the business impact and integration semantics of these configurations. Return only the JSON object, no additional explanation."""

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response that might contain additional content"""
        try:
            # Claude sometimes wraps JSON in code blocks
            if "```json" in text:
                start_marker = "```json"
                end_marker = "```"
                start_idx = text.find(start_marker) + len(start_marker)
                end_idx = text.find(end_marker, start_idx)
                if end_idx != -1:
                    json_str = text[start_idx:end_idx].strip()
                    return json.loads(json_str)

            # Try to find JSON object in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._get_empty_classification()

        except Exception:
            return self._get_empty_classification()


class AnthropicRecommender(LLMRecommender):
    """Anthropic Claude implementation for returning coding implementation suggestions"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self._api_key = api_key
        self._model = model
        self._base_url = "https://api.anthropic.com/v1/messages"
        print(f"Initialized Anthropic provider with model: {model}")

    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        return f"Anthropic-{self._model}"

    def fetch_answer(self, context_provider: BusinessContextProvider) -> Dict[str, Any]:
        """Generate implementation using Anthropic Claude"""

        prompt = context_provider.build_prompt()

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01"
            }

            payload = {
                "model": self._model,
                "max_tokens": 2000,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(
                self._base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            anthropic_response = response.json()
            generated_text = anthropic_response["content"][0]["text"]

            # Parse JSON response
            try:
                answer = json.loads(generated_text)
            except json.JSONDecodeError:
                answer = self._extract_json_from_text(generated_text)

            return get_validated_answer(answer)

        except Exception as e:
            print(f"Error calling Anthropic: {e}")
            return get_fallback_answer(context_provider.user_request)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response that might contain additional content"""
        try:
            # Claude sometimes wraps JSON in code blocks
            if "```json" in text:
                start_marker = "```json"
                end_marker = "```"
                start_idx = text.find(start_marker) + len(start_marker)
                end_idx = text.find(end_marker, start_idx)
                if end_idx != -1:
                    json_str = text[start_idx:end_idx].strip()
                    return json.loads(json_str)

            # Try to find JSON object in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._get_empty_implementation()

        except Exception:
            return self._get_empty_implementation()

