from typing import List, Any, Dict

MAX_CONTEXT_LENGTH = 8000  # Characters for LLM context

class PromptBuilder:

    def create_implementation_prompt(self, user_request: str, context_docs: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM implementation generation"""

        # Build context summary
        context_summary = self._build_context_summary(context_docs)

        prompt = f"""You are a senior software architect working on a loyalty points microservice system. A developer has requested a new feature implementation.

    DEVELOPER REQUEST:
    "{user_request}"

    RELEVANT EXISTING CODE CONTEXT:
    {context_summary}

    Based on the existing codebase patterns and architecture, provide specific implementation guidance. Return ONLY a valid JSON object with this structure:

    {{
        "suggested_service": "Which service/project should implement this (e.g., LoyaltyPoints, LoyaltyPoints.Internal)",
        "suggested_files": [
            {{
                "action": "create|modify|reference",
                "file_path": "/specific/path/to/file.cs",
                "purpose": "What this file does"
            }}
        ],
        "implementation_steps": [
            "Step 1: Specific action to take",
            "Step 2: Next specific action",
            "Step 3: etc."
        ],
        "business_rationale": "Why this approach makes business sense based on existing patterns",
        "integration_points": [
            "Service A: How it integrates",
            "Service B: How it integrates"
        ],
        "code_examples": [
            {{
                "file": "FileName.cs",
                "code": "public class Example {{ /* implementation */ }}"
            }}
        ],
        "confidence_score": 0.85
    }}

    Focus on:
    1. Following existing architectural patterns from the context
    2. Reusing existing business rule structures
    3. Maintaining consistency with current integration approaches
    4. Providing specific, actionable implementation steps

    Return only the JSON object, no additional text."""

        return prompt

    @staticmethod
    def _build_context_summary(context_docs: List[Dict[str, Any]]) -> str:
        """Build a concise summary of retrieved context"""

        if not context_docs:
            return "No relevant context found."

        summary_parts = []

        # Group by project
        projects = {}
        for doc in context_docs:
            project = doc['project_name']
            if project not in projects:
                projects[project] = []
            projects[project].append(doc)

        for project, docs in projects.items():
            summary_parts.append(f"\n--- {project} Project ---")

            for doc in docs:
                summary_parts.append(f"File: {doc['file_path']}")
                summary_parts.append(f"Purpose: {doc['business_purpose']}")
                summary_parts.append(f"Pattern: {doc['technical_pattern']}")

                if doc['business_rules']:
                    summary_parts.append(f"Rules: {', '.join(doc['business_rules'][:3])}")

                if doc['integration_points']:
                    summary_parts.append(f"Integrations: {', '.join(doc['integration_points'][:3])}")

                summary_parts.append("")  # Blank line between files

        # Truncate if too long
        full_summary = "\n".join(summary_parts)
        if len(full_summary) > MAX_CONTEXT_LENGTH:
            return full_summary[:MAX_CONTEXT_LENGTH] + "\n... [truncated]"

        return full_summary