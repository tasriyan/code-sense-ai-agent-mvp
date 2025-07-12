from datetime import datetime

from generation.implementation_suggestion import ImplementationSuggestion
from generation.semantic_data_retriever import SemanticDataRetriever
from llm_providers.llm_provider import LLMExecutor


class CodeSenseImplementationGenerator:
    """Main class for generating implementation suggestions using RAG"""

    def __init__(self,
                 retriever: SemanticDataRetriever,
                 llm_provider: LLMExecutor):
        self.retriever = retriever
        self.llm_provider = llm_provider

    def generate_complete_suggestion(self, user_request: str) -> ImplementationSuggestion:
        """Generate complete implementation suggestion using RAG pipeline"""

        llm_provider_name = self.llm_provider.get_provider_name()
        print(f"\n=== Generating Implementation Suggestion ===")
        print(f"Request: {user_request}")
        print(f"Provider: {llm_provider_name}")

        # Step 1: Retrieve relevant context
        context_docs = self.retriever.get_relevant_context(user_request)

        if not context_docs:
            print("Warning: No relevant context found. Generating basic suggestion.")

        # Step 2: Generate implementation using specified LLM
        print(f"Generating implementation with {llm_provider_name}...")
        implementation = self.llm_provider.suggest_coding_implementation(user_request, context_docs)

        # Step 3: Create complete suggestion object
        suggestion = ImplementationSuggestion(
            user_request=user_request,
            retrieved_context=context_docs,
            suggested_service=implementation['suggested_service'],
            suggested_files=implementation['suggested_files'],
            implementation_steps=implementation['implementation_steps'],
            business_rationale=implementation['business_rationale'],
            integration_points=implementation['integration_points'],
            code_examples=implementation['code_examples'],
            confidence_score=implementation['confidence_score'],
            llm_provider=llm_provider_name,
            generated_at=datetime.now().isoformat()
        )

        return suggestion