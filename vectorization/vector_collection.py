from typing import List, Dict, Any

import chromadb
import numpy as np

BATCH_SIZE = 100

class VectorCollection:
    def __init__(self, collection: chromadb.Collection):
        self._collection = collection


    def add_documents_to_collection(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to Chroma collection in batches"""

        total_docs = len(documents)
        processed = 0

        for i in range(0, total_docs, BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]

            # Prepare batch data
            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]

            try:
                self._collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )

                processed += len(batch)
                print(f"Added batch {i // BATCH_SIZE + 1}: {processed}/{total_docs} documents")

            except Exception as e:
                print(f"Error adding batch {i // BATCH_SIZE + 1}: {e}")
                continue

        print(f"Successfully added {self._collection.count()} documents to collection")

    def semantic_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """ Perform semantic search on collection """
        print(f"\n=== Performing Semantic Search ===")
        print(f"Query: '{query}'")
        print(f"Retrieving top {n_results} results...")

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            print(f"\nFound {len(results['ids'][0])} results:")

            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                print(f"\n--- Result {i + 1} (Distance: {distance:.4f}) ---")
                print(f"File: {metadata.get('file_path', 'Unknown')}")
                print(f"Business Purpose: {metadata.get('business_purpose', 'Unknown')}")
                print(f"Technical Pattern: {metadata.get('technical_pattern', 'Unknown')}")
                print(f"Business Workflow: {metadata.get('business_workflow', 'Unknown')[:100]}...")

            return {
                'query': query,
                'results': results,
                'summary': {
                    'total_results': len(results['ids'][0]),
                    'avg_distance': np.mean(results['distances'][0]) if results['distances'][0] else 0,
                    'files_found': [meta.get('file_path') for meta in results['metadatas'][0]]
                }
            }

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return {'error': str(e)}

    def filtered_retrieval(self, query: str, filters: Dict[str, Any], n_results: int = 5) -> Dict[str, Any]:
        """Test retrieval with metadata filters"""

        print(f"\n=== Filtered Retrieval Test ===")
        print(f"Query: '{query}'")
        print(f"Filters: {filters}")
        print(f"Retrieving top {n_results} results...")

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            if not results['ids'][0]:
                return {
                    'query': query,
                    'filters': filters,
                    'results': [],
                    'summary': {'total_results': 0, 'message': 'No results found with filters'}
                }

            print(f"\nFound {len(results['ids'][0])} results with filters:")

            retrieved_docs = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                print(f"\n--- Filtered Result {i + 1} (Distance: {distance:.4f}) ---")
                print(f"File: {metadata.get('file_path', 'Unknown')}")
                print(f"Project: {metadata.get('project_name', 'Unknown')}")
                print(f"File Type: {metadata.get('file_type', 'Unknown')}")
                print(f"Business Purpose: {metadata.get('business_purpose', 'Unknown')[:100]}...")

                retrieved_docs.append({
                    'id': doc_id,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'project_name': metadata.get('project_name', 'Unknown'),
                    'file_type': metadata.get('file_type', 'Unknown'),
                    'business_purpose': metadata.get('business_purpose', 'Unknown'),
                    'distance': distance,
                    'document': document
                })

            return {
                'query': query,
                'filters': filters,
                'results': retrieved_docs,
                'summary': {
                    'total_results': len(results['ids'][0]),
                    'avg_distance': np.mean(results['distances'][0]),
                    'files_found': [doc['file_path'] for doc in retrieved_docs]
                }
            }

        except Exception as e:
            print(f"Error during filtered retrieval: {e}")
            return {
                'query': query,
                'filters': filters,
                'error': str(e),
                'results': [],
                'summary': {'total_results': 0, 'error': str(e)}
            }

    def get_collection_stats_v1(self) -> Dict[str, Any]:
        """Get statistics about the collection"""

        try:
            # Get all documents (limit for stats)
            all_docs = self._collection.get(limit=1000, include=["metadatas"])

            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            metadatas = all_docs['metadatas']

            # Calculate statistics
            stats = {
                'total_documents': self._collection.count(),
                'projects': {},
                'file_types': {},
                'llm_providers': {},
                'technical_patterns': {},
                'avg_confidence': 0
            }

            confidences = []

            for metadata in metadatas:
                # Project distribution
                project = metadata.get('project_name', 'Unknown')
                stats['projects'][project] = stats['projects'].get(project, 0) + 1

                # File type distribution
                file_type = metadata.get('file_type', 'Unknown')
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1

                # LLM provider distribution
                provider = metadata.get('llm_provider', 'Unknown')
                stats['llm_providers'][provider] = stats['llm_providers'].get(provider, 0) + 1

                # Technical pattern distribution
                pattern = metadata.get('technical_pattern', 'Unknown')
                stats['technical_patterns'][pattern] = stats['technical_patterns'].get(pattern, 0) + 1

                # Confidence scores
                confidence = metadata.get('classification_confidence', 0)
                if isinstance(confidence, (int, float)):
                    confidences.append(confidence)

            # Calculate average confidence
            if confidences:
                stats['avg_confidence'] = np.mean(confidences)

            return stats

        except Exception as e:
            return {'error': str(e)}

    def get_collection_stats_v2(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            total_count = self._collection.count()
            sample_docs = self._collection.get(limit=100, include=["metadatas"])

            if not sample_docs['metadatas']:
                return {'total_documents': total_count, 'error': 'No metadata available'}

            # Analyze metadata distribution
            projects = {}
            file_types = {}
            patterns = {}

            for metadata in sample_docs['metadatas']:
                project = metadata.get('project_name', 'Unknown')
                projects[project] = projects.get(project, 0) + 1

                file_type = metadata.get('file_type', 'Unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1

                pattern = metadata.get('technical_pattern', 'Unknown')
                patterns[pattern] = patterns.get(pattern, 0) + 1

            return {
                'total_documents': total_count,
                'sample_size': len(sample_docs['metadatas']),
                'project_distribution': projects,
                'file_type_distribution': file_types,
                'pattern_distribution': patterns
            }

        except Exception as e:
            return {'error': str(e)}

class EmptyVectorCollection(VectorCollection):
    def __init__(self):
        super().__init__(collection=None)