from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions

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

    def get_collection_stats(self) -> Dict[str, Any]:
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

class EmptyVectorCollection(VectorCollection):
    def __init__(self):
        super().__init__(collection=None)

class SemanticVectorDatabase:
    """Manages the vector database for semantic code classification"""

    def __init__(self,
                 db_path: str = "./chroma_db",
                 embedding_model = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
        )

        print(f"Initialized Chroma database at: {self.db_path}")
        print(f"Using embedding model: {self.embedding_model}")

    def create_collection(self, collection_name: str, reset_if_exists: bool = False) -> VectorCollection:
        """Create or get the semantic code collection"""

        if reset_if_exists:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except NotFoundError:
                pass  # Collection doesn't exist

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Semantic classification of loyalty service code",
                "created_at": datetime.now().isoformat(),
                "embedding_model": self.embedding_model,
            }
        )

        print(f"Collection '{collection_name}' ready with {collection.count()} documents")
        return VectorCollection(collection)

    def get_collection(self, collection_name: str) -> VectorCollection:
        try:
            return VectorCollection(self.client.get_collection(name=collection_name))
        except NotFoundError:
            print(f"Collection '{collection_name}' not found.")
            return EmptyVectorCollection()