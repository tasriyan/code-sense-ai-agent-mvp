from datetime import datetime
from typing import Dict, Any

from llms.providers import LLMRecommender
from shared.codesense_advice import CodeSenseAdvice
from shared.context_providers import ConversationContextProvider
from shared.tool_agent import ToolAgent
from rag.rag import RagSystem, RagQueryResult


class EnhancedCodeSenseGenerator:
    """Enhanced CodeSense generator with tool orchestration"""

    def __init__(self,
                 llm: LLMRecommender,
                 rag_system: RagSystem,
                 tool_agent: ToolAgent):
        self._llm = llm
        self._rag_system = rag_system
        self._tool_agent = tool_agent
        self._max_tool_iterations = 3

    def fetch_coding_advice(self, user_request: str) -> CodeSenseAdvice:
        """Generate implementation using the full orchestrated pipeline"""
        llm_provider_name = self._llm.get_provider_name()
        print(f"\n=== Generating Implementation Suggestion ===")
        print(f"Request: {user_request}")
        print(f"Provider: {llm_provider_name}")

        llm_answer = self._fetch_llm_answer(user_request=user_request)
        final_answer = CodeSenseAdvice(
            user_request=user_request,
            retrieved_context=[],
            suggested_service=llm_answer['suggested_service'],
            suggested_files=llm_answer['suggested_files'],
            implementation_steps=llm_answer['implementation_steps'],
            business_rationale=llm_answer['business_rationale'],
            integration_points=llm_answer['integration_points'],
            code_examples=llm_answer['code_examples'],
            confidence_score=llm_answer['confidence_score'],
            llm_provider=llm_provider_name,
            generated_at=datetime.now().isoformat()
        )

        return final_answer

    def _fetch_llm_answer(self, user_request: str) -> Dict[str, Any]:
        """Orchestrate the complete implementation generation process"""

        print(f"\n=== Orchestrating Implementation Generation ===")
        print(f"Request: {user_request}")
        print(f"Provider: {self._llm.get_provider_name()}")

        print("\nRetrieving business context via RAG...")
        rag_query_result = self._rag_system.retrieve_relevant_context(user_request)

        print("\nStarting LLM interaction with tool calling...")
        return self._execute_llm_with_tools(user_request, rag_query_result)

    def _execute_llm_with_tools(self, user_request: str, rag_query_result: RagQueryResult) -> Dict[str, Any]:
        print("Creating initial prompt with tool availability...")
        business_context = ConversationContextProvider(user_request=user_request, rag_data=rag_query_result, tool_agent=self._tool_agent)

        for iteration in range(self._max_tool_iterations):
            print(f"\n--- LLM Iteration {iteration + 1} ---")

            llm_response = self._llm.fetch_answer(business_context)
            if llm_response:
                business_context.add_llm_response(llm_response)
            else:
                break

            # Check if LLM wants to use tools
            tool_calls = self._tool_agent.parse_tool_calls_from_response(llm_response)

            if not tool_calls:
                print("No tool calls detected. Returning final response...")
                return llm_response

            print(f"Executing {len(tool_calls)} tool call(s)...")

            for tool_call in tool_calls:
                print(f"  Calling {tool_call['tool_name']} with {tool_call['parameters']}")
                tool_result = self._tool_agent.execute_tool(
                    tool_call['tool_name'],
                    **tool_call['parameters']
                )
                if tool_result.success:
                    print(f"  ✓ Success: Retrieved {len(tool_result.result['content'])} characters")
                else:
                    print(f"  ✗ Error: {tool_result.error}")
                business_context.add_tool_response(tool_result)

        # If we reach max iterations, try to get final response
        print("Reached max iterations. Requesting final response...")
        final_prompt = current_prompt + "\n\nProvide your final implementation guidance as JSON now."
        final_response = self._llm.fetch_answer(business_context)
        return final_response