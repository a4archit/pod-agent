from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from typing import Optional, List
import uuid


class QdrantTempManager:
    """Manage temporary Qdrant vector databases with LangChain integration."""
    
    def __init__(
        self,
        location: str = ":memory:",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize Qdrant client for temporary storage.
        
        Args:
            location: ":memory:" for in-memory (default), or path for disk-based temp storage
            host: Qdrant server host (if using remote server)
            port: Qdrant server port (if using remote server)
        """
        if host and port:
            # Connect to remote Qdrant server
            self.client = QdrantClient(host=host, port=port)
            self.is_remote = True
        else:
            # Use local in-memory or disk-based storage
            self.client = QdrantClient(location=location)
            self.is_remote = False
        
        self.active_collections = set()
        self.vector_stores = {}
    



    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        recreate: bool = True
    ) -> str:
        """
        Create a temporary collection in Qdrant.
        
        Args:
            collection_name: Name for the collection (auto-generated if None)
            vector_size: Dimension of vectors (default: 1536 for OpenAI embeddings)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete and recreate if collection exists
            
        Returns:
            Collection name
        """
        # Generate unique collection name if not provided
        if collection_name is None:
            collection_name = f"temp_collection_{uuid.uuid4().hex[:8]}"
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(col.name == collection_name for col in collections)
        
        if exists:
            if recreate:
                print(f"Collection '{collection_name}' exists. Recreating...")
                self.client.delete_collection(collection_name)
                # Remove from vector_stores cache
                self.vector_stores.pop(collection_name, None)
            else:
                print(f"Collection '{collection_name}' already exists. Reusing...")
                self.active_collections.add(collection_name)
                return collection_name
        
        # Create new collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        
        self.active_collections.add(collection_name)
        print(f"Created collection: '{collection_name}'")
        
        return collection_name
    



    def get_vector_store(
        self,
        collection_name: str,
        embeddings: Embeddings
    ) -> QdrantVectorStore:
        """
        Get or create a QdrantVectorStore for the collection.
        
        Args:
            collection_name: Name of the collection
            embeddings: LangChain embeddings instance
            
        Returns:
            QdrantVectorStore instance
        """
        # Return cached vector store if exists
        if collection_name in self.vector_stores:
            return self.vector_stores[collection_name]
        
        # Create new vector store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        self.vector_stores[collection_name] = vector_store
        return vector_store
    




    def create_vector_store(
        self,
        embeddings: Embeddings,
        collection_name: Optional[str] = None,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        recreate: bool = True
    ) -> tuple[QdrantVectorStore, str]:
        """
        Create collection and return QdrantVectorStore in one step.
        
        Args:
            embeddings: LangChain embeddings instance
            collection_name: Name for collection (auto-generated if None)
            vector_size: Vector dimension
            distance: Distance metric
            recreate: Recreate if exists
            
        Returns:
            Tuple of (QdrantVectorStore, collection_name)
        """
        # Create collection
        collection = self.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            recreate=recreate
        )
        
        # Get vector store
        vector_store = self.get_vector_store(collection, embeddings)
        
        return vector_store, collection
    




    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a specific collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            self.active_collections.discard(collection_name)
            self.vector_stores.pop(collection_name, None)
            print(f"Deleted collection: '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def cleanup_all(self) -> None:
        """Delete all active collections created by this manager."""
        print(f"Cleaning up {len(self.active_collections)} collections...")
        for collection_name in list(self.active_collections):
            self.delete_collection(collection_name)
    
    def get_client(self) -> QdrantClient:
        """Get the underlying Qdrant client."""
        return self.client
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        collections = self.client.get_collections().collections
        return [col.name for col in collections]






# Standalone function for quick setup with VectorStore
def setup_temp_vector_store(
    embeddings: Embeddings,
    collection_name: Optional[str] = None,
    vector_size: int = 1536,
    in_memory: bool = True,
    host: Optional[str] = None,
    port: Optional[int] = None
) -> tuple[QdrantVectorStore, str, QdrantTempManager]:
    """
    Quick setup function for temporary Qdrant VectorStore.
    
    Args:
        embeddings: LangChain embeddings instance
        collection_name: Name for collection (auto-generated if None)
        vector_size: Vector dimension
        in_memory: Use in-memory storage (True) or disk (False)
        host: Remote Qdrant host (optional)
        port: Remote Qdrant port (optional)
        
    Returns:
        Tuple of (QdrantVectorStore, collection_name, manager)
    """
    location = ":memory:" if in_memory else "./temp_qdrant_data"
    
    manager = QdrantTempManager(location=location, host=host, port=port)
    vector_store, collection = manager.create_vector_store(
        embeddings=embeddings,
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=True
    )
    
    return vector_store, collection, manager





# Example usage
if __name__ == "__main__":
    # from langchain_openai import OpenAIEmbeddings
    # # or from langchain_huggingface import HuggingFaceEmbeddings
    
    # # Initialize embeddings
    # embeddings = OpenAIEmbeddings()
    
    # print("=== Method 1: Using Manager Class ===")
    # # Create manager
    # manager = QdrantTempManager(location=":memory:")
    
    # # Create vector store
    # vector_store, collection_name = manager.create_vector_store(
    #     embeddings=embeddings,
    #     collection_name="my_docs",
    #     vector_size=1536
    # )
    
    # # Now use the vector store
    # # vector_store.add_documents(documents)
    # # results = vector_store.similarity_search("query", k=5)
    
    # print(f"VectorStore ready for collection: {collection_name}")
    
    # # Cleanup when done
    # manager.cleanup_all()
    
    # print("\n=== Method 2: Quick Setup Function ===")
    # # Quick setup
    # vector_store, collection, manager = setup_temp_vector_store(
    #     embeddings=embeddings,
    #     collection_name="quick_setup",
    #     vector_size=1536,
    #     in_memory=True
    # )
    
    # print(f"Ready to use VectorStore: {collection}")
    
    # # Don't forget cleanup
    # manager.cleanup_all()

    print()