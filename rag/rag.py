from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

from vectorization.vector_collection import VectorCollection, SemanticMatch


@dataclass
class RagQueryResult:
    timestamp: str
    matches: SemanticMatch
    user_query: str

    def get_business_context(self) -> str:
        return self.matches.build_business_context_summary()

class RagSystem(ABC):
    def __init__(self, vector_collection: VectorCollection):
        self._vector_collection = vector_collection

    @abstractmethod
    def retrieve_relevant_context(self, user_request: str) -> RagQueryResult:
        pass

class BasicContentRag(RagSystem):

    def __init__(self, vector_collection: VectorCollection):
        super().__init__(vector_collection)

    def retrieve_relevant_context(self, user_request: str) -> RagQueryResult:
        query_result = self._vector_collection.semantic_search(user_request, n_results=3)
        return RagQueryResult(timestamp=datetime.now().isoformat(), matches=query_result, user_query=user_request)

class FilteredContentRag(RagSystem):
    def __init__(self, vector_collection: VectorCollection, filters: Dict[str, Any]):
        super().__init__(vector_collection)

        self.filters = filters

    def retrieve_relevant_context(self, user_request: str) -> RagQueryResult:
        query_result = self._vector_collection.filtered_semantic_search(user_request, self.filters, n_results=3)
        return RagQueryResult(timestamp=datetime.now().isoformat(), matches=query_result, user_query=user_request)