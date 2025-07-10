from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions

from vectorization.vector_collection import VectorCollection, EmptyVectorCollection


class SemanticVectorDatabase:
    """ Chroma Integration
        - Persistent storage for the vector database
        - Sentence-transformers for embeddings (all-MiniLM-L6-v2 default)
        - Collection management
    """

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