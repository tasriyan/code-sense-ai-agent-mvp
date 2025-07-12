from datetime import datetime
from shared.codesense_advice import CodeSenseAdvice
from llms.providers import LLMRecommender
from rag.rag import RagSystem
from shared.context_providers import BusinessContextProvider


class CodeSenseGenerator:
    """Main class for generating implementation suggestions using RAG"""

    def __init__(self,
                 rag: RagSystem,
                 llm_provider: LLMRecommender):
        self._rag = rag
        self._llm = llm_provider

    def fetch_coding_advice(self, user_request: str) -> CodeSenseAdvice:
        """Generate complete implementation suggestion using RAG pipeline"""

        llm_provider_name = self._llm.get_provider_name()
        print(f"\n=== Generating Implementation Suggestion ===")
        print(f"Request: {user_request}")
        print(f"Provider: {llm_provider_name}")

        print("\n1. Retrieving business context via RAG...")
        rag_results =self._rag.retrieve_relevant_context(user_request=user_request)

        print(f"\n2. Generating implementation with {llm_provider_name}...")
        context_provider = BusinessContextProvider(user_request, rag_results)
        llm_answer = self._llm.fetch_answer(context_provider)

        code_sense_advice = CodeSenseAdvice(
            user_request=user_request,
            retrieved_context=rag_results.matches.get_context_docs(),
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

        return code_sense_advice