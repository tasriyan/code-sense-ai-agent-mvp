from typing import List, Any, Dict

from rag.rag import RagQueryResult
from shared.tool_agent import ToolResult, ToolAgent

MAX_CONTEXT_LENGTH = 8000  # Characters for LLM context

class BusinessContextProvider:
    def __init__(self, user_request: str, rag_data: RagQueryResult):
        self.user_request = user_request
        self.rag_data = rag_data

    def build_prompt(self) -> str:
        """Create prompt for LLM implementation generation"""

        # Build context summary
        context_summary = self.rag_data.get_business_context()

        prompt = f"""You are a senior software architect working on a loyalty points microservice system. A developer has requested a new feature implementation.

    DEVELOPER REQUEST:
    "{self.user_request}"

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


class ConversationContextProvider(BusinessContextProvider):
    def __init__(self, user_request: str, rag_data: RagQueryResult, tool_agent: ToolAgent = None):
        super().__init__(user_request, rag_data)

        self._all_responses = []
        self._conversation_history = self._create_orchestrated_prompt()
        self._tool_responses = []
        self._tool_agent = tool_agent

    def build_prompt(self) -> str:
        """Build prompt from conversation history"""

        self._update_conversation_history()

        if len(self._conversation_history) == 1:
            return self._conversation_history[0]  # Initial prompt

        # Add tool results context
        if self._all_responses:
            latest_results = [r for r in self._all_responses if r.success][-3:]  # Last 3 successful results
            context_additions = []

            for result in latest_results:
                if result.result and 'content' in result.result:
                    context_additions.append(f"Code from {result.result['relative_path']}:")
                    context_additions.append(f"```csharp")
                    context_additions.append(result.result['content'][:1000])  # First 1000 chars
                    context_additions.append(f"```")

            if context_additions:
                return self._conversation_history[0] + "\n\nEXAMINED CODE:\n" + "\n".join(
                    context_additions) + "\n\nNow provide final JSON implementation guidance:"

        return self._conversation_history[0] + "\n\nProvide implementation guidance as JSON:"

    def add_llm_response(self, llm_response):
        self._conversation_history.append(f"LLM Response: {llm_response}")

    def add_tool_response(self, tool_response: ToolResult):
        self._all_responses.append(tool_response)
        self._tool_responses.append(tool_response)

    def _update_conversation_history(self):
        """Format tool results for conversation history"""
        if len(self._tool_responses) == 0:
            return

        formatted = []
        for response in self._tool_responses:
            if response.success and response.result:
                formatted.append(
                    f"Retrieved {response.result['relative_path']} ({len(response.result['content'])} chars)")
            else:
                formatted.append(f"Failed to retrieve: {response.error}")

        summary = "; ".join(formatted)
        self._conversation_history.append(f"Tool Results: {summary}")

    def _create_orchestrated_prompt(self) -> List[str]:
        """Create prompt that combines RAG context with tool availability"""

        # Get available tools
        available_tools = self._tool_agent.get_available_tools()
        tools_description = self._format_tools_for_prompt(available_tools)
        # Build business context
        business_context = self.rag_data.get_business_context()

        prompt = f"""You are a senior software architect implementing a new feature in a loyalty points microservice system.

DEVELOPER REQUEST:
"{self.user_request}"

BUSINESS CONTEXT FROM EXISTING CODEBASE:
{business_context}

AVAILABLE TOOLS:
{tools_description}

IMPLEMENTATION APPROACH:
1. First, analyze the business context above to understand existing patterns
2. If you need to see specific code implementations, use get_code_by_filepath(file_path) tool to examine relevant files
3. Generate specific implementation guidance following existing patterns

TOOL USAGE INSTRUCTIONS:
- Use get_code_by_filepath(file_path) when you need to:
  * See interface definitions (e.g., ILoyaltyRule)
  * Understand existing implementation patterns
  * Check dependency injection patterns
  * Examine configuration patterns
- Call tools using this format: get_code_by_filepath("/path/to/file.cs")

Based on the existing codebase patterns and architecture AND examining necessary code files, provide specific implementation guidance. Return ONLY a valid JSON object with this structure:

{{
    "analysis_performed": [
    "List of files examined and why",
    "Key patterns discovered"
    ],
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

Start by examining the most relevant files to understand implementation patterns."""

        return [prompt]

    @staticmethod
    def _format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
        """Format available tools for the prompt"""

        tool_descriptions = []
        for tool in tools:
            desc = f"- {tool['name']}: {tool['description']}"
            if 'parameters' in tool:
                params = tool['parameters']['properties']
                param_desc = ", ".join([f"{k} ({v['type']})" for k, v in params.items()])
                desc += f"\n  Parameters: {param_desc}"
            tool_descriptions.append(desc)

        return "\n".join(tool_descriptions)