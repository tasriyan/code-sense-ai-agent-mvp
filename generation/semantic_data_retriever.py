from typing import List, Dict, Any

from vectorization.vector_collection import VectorCollection

TOP_K_RETRIEVAL = 5        # Number of documents to retrieve

class SemanticDataRetriever:
    def __init__(self, collection: VectorCollection):
        self.collection = collection

    def get_relevant_context(self, user_request: str, n_results: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Retrieve relevant code context for user request"""

        print(f"Retrieving context for: '{user_request}'")

        try:
            # Query the vector database
            results_data = self.collection.semantic_search(user_request, n_results)
            results = results_data['results']
            if not results['ids'][0]:
                print("No relevant context found")
                return []
            print(f"\nFound {len(results['ids'][0])} results:")

            # Structure the retrieved context
            context_docs = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                context_doc = {
                    'rank': i + 1,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'project_name': metadata.get('project_name', 'Unknown'),
                    'file_type': metadata.get('file_type', 'Unknown'),
                    'business_purpose': metadata.get('business_purpose', 'Unknown'),
                    'business_rules': metadata.get('business_rules', '').split(' | ') if metadata.get(
                        'business_rules') else [],
                    'business_triggers': metadata.get('business_triggers', '').split(' | ') if metadata.get(
                        'business_triggers') else [],
                    'business_data': metadata.get('business_data', '').split(' | ') if metadata.get(
                        'business_data') else [],
                    'integration_points': metadata.get('integration_points', '').split(' | ') if metadata.get(
                        'integration_points') else [],
                    'business_workflow': metadata.get('business_workflow', 'Unknown'),
                    'technical_pattern': metadata.get('technical_pattern', 'Unknown'),
                    'distance': distance,
                    'full_document': document,
                    'confidence': metadata.get('classification_confidence', 0.0)
                }
                context_docs.append(context_doc)

            print(f"Retrieved {len(context_docs)} relevant documents")
            for doc in context_docs:
                print(f"  - {doc['file_path']} (distance: {doc['distance']:.4f})")

            return context_docs

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []