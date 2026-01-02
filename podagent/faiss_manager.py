from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import Optional, List, Dict
import uuid
import os
import shutil
from pathlib import Path


class FAISSTempManager:
    """Manage temporary FAISS vector stores with automatic cleanup."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize FAISS temporary manager.
        
        Args:
            base_path: Base directory for temporary stores (default: ./temp_faiss)
                      Set to None to use in-memory only (no persistence)
        """
        self.base_path = Path(base_path) if base_path else Path("./temp_faiss")
        self.vector_stores: Dict[str, FAISS] = {}
        self.store_paths: Dict[str, Path] = {}
        
        # Create base directory if using disk storage
        if base_path:
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(
        self,
        embeddings: Embeddings,
        store_name: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        save_to_disk: bool = False
    ) -> tuple[FAISS, str]:
        """
        Create a new FAISS vector store.
        
        Args:
            embeddings: LangChain embeddings instance
            store_name: Name for the store (auto-generated if None)
            documents: List of Document objects to add initially
            texts: List of text strings to add initially
            metadatas: List of metadata dicts (used with texts)
            save_to_disk: Whether to save the store to disk
            
        Returns:
            Tuple of (FAISS vector store, store_name)
        """
        # Generate unique store name if not provided
        if store_name is None:
            store_name = f"faiss_store_{uuid.uuid4().hex[:8]}"
        
        # Delete existing store if it exists
        if store_name in self.vector_stores:
            print(f"Store '{store_name}' exists. Recreating...")
            self.delete_store(store_name)
        
        # Create vector store
        if documents:
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            print(f"Created FAISS store '{store_name}' with {len(documents)} documents")
        elif texts:
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas
            )
            print(f"Created FAISS store '{store_name}' with {len(texts)} texts")
        else:
            # Create empty store - need at least one document to initialize
            dummy_doc = Document(page_content="initialization", metadata={"dummy": True})
            vector_store = FAISS.from_documents(
                documents=[dummy_doc],
                embedding=embeddings
            )
            print(f"Created empty FAISS store '{store_name}'")
        
        # Store reference
        self.vector_stores[store_name] = vector_store
        
        # Save to disk if requested
        if save_to_disk:
            store_path = self.base_path / store_name
            vector_store.save_local(str(store_path))
            self.store_paths[store_name] = store_path
            print(f"Saved to disk: {store_path}")
        
        return vector_store, store_name
    
    def load_vector_store(
        self,
        store_name: str,
        embeddings: Embeddings,
        allow_dangerous_deserialization: bool = True
    ) -> FAISS:
        """
        Load a FAISS vector store from disk.
        
        Args:
            store_name: Name of the store to load
            embeddings: LangChain embeddings instance
            allow_dangerous_deserialization: Required for FAISS loading
            
        Returns:
            FAISS vector store
        """
        store_path = self.base_path / store_name
        
        if not store_path.exists():
            raise FileNotFoundError(f"Store not found: {store_path}")
        
        vector_store = FAISS.load_local(
            str(store_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        self.vector_stores[store_name] = vector_store
        self.store_paths[store_name] = store_path
        print(f"Loaded FAISS store from: {store_path}")
        
        return vector_store
    
    def get_vector_store(self, store_name: str) -> Optional[FAISS]:
        """
        Get an existing vector store by name.
        
        Args:
            store_name: Name of the store
            
        Returns:
            FAISS vector store or None if not found
        """
        return self.vector_stores.get(store_name)
    
    def delete_store(self, store_name: str) -> bool:
        """
        Delete a vector store and its disk storage.
        
        Args:
            store_name: Name of the store to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Remove from memory
            if store_name in self.vector_stores:
                del self.vector_stores[store_name]
            
            # Remove from disk
            if store_name in self.store_paths:
                store_path = self.store_paths[store_name]
                if store_path.exists():
                    shutil.rmtree(store_path)
                del self.store_paths[store_name]
            
            print(f"Deleted store: '{store_name}'")
            return True
        except Exception as e:
            print(f"Error deleting store '{store_name}': {e}")
            return False
    
    def cleanup_all(self) -> None:
        """Delete all active stores created by this manager."""
        print(f"Cleaning up {len(self.vector_stores)} stores...")
        for store_name in list(self.vector_stores.keys()):
            self.delete_store(store_name)
        
        # Remove base directory if empty and exists
        if self.base_path.exists() and not any(self.base_path.iterdir()):
            self.base_path.rmdir()
            print(f"Removed empty directory: {self.base_path}")
    
    def list_stores(self) -> List[str]:
        """List all active stores in memory."""
        return list(self.vector_stores.keys())
    
    def list_disk_stores(self) -> List[str]:
        """List all stores saved to disk."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]


# Standalone function for quick setup
def setup_temp_faiss(
    embeddings: Embeddings,
    store_name: Optional[str] = None,
    documents: Optional[List[Document]] = None,
    texts: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None,
    save_to_disk: bool = False
) -> tuple[FAISS, str, FAISSTempManager]:
    """
    Quick setup function for temporary FAISS vector store.
    
    Args:
        embeddings: LangChain embeddings instance
        store_name: Name for the store (auto-generated if None)
        documents: Initial documents to add
        texts: Initial texts to add
        metadatas: Metadata for texts
        save_to_disk: Whether to persist to disk
        
    Returns:
        Tuple of (FAISS store, store_name, manager)
    """
    manager = FAISSTempManager()
    vector_store, name = manager.create_vector_store(
        embeddings=embeddings,
        store_name=store_name,
        documents=documents,
        texts=texts,
        metadatas=metadatas,
        save_to_disk=save_to_disk
    )
    
    return vector_store, name, manager






# Example usage
if __name__ == "__main__":

    print()
    
    '''
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    print("=== Method 1: Using Manager Class ===")
    # Create manager
    manager = FAISSTempManager(base_path="./temp_faiss")
    
    # Example documents
    documents = [
        Document(page_content="FAISS is a library for efficient similarity search", 
                metadata={"source": "doc1", "page": 1}),
        Document(page_content="Vector stores enable semantic search capabilities",
                metadata={"source": "doc2", "page": 1})
    ]
    
    # Create vector store with documents
    vector_store, store_name = manager.create_vector_store(
        embeddings=embeddings,
        store_name="my_docs",
        documents=documents,
        save_to_disk=True
    )
    
    # Use the vector store
    results = vector_store.similarity_search("What is FAISS?", k=2)
    print(f"\nSearch results: {len(results)} documents found")
    
    # Add more documents later
    more_docs = [
        Document(page_content="LangChain integrates with FAISS seamlessly",
                metadata={"source": "doc3", "page": 1})
    ]
    vector_store.add_documents(more_docs)
    
    # List stores
    print(f"\nActive stores: {manager.list_stores()}")
    print(f"Disk stores: {manager.list_disk_stores()}")
    
    # Cleanup
    manager.cleanup_all()
    
    print("\n=== Method 2: Quick Setup Function ===")
    # Quick setup
    vector_store, name, manager = setup_temp_faiss(
        embeddings=embeddings,
        store_name="quick_store",
        texts=["Hello world", "FAISS is fast"],
        metadatas=[{"id": 1}, {"id": 2}],
        save_to_disk=False
    )
    
    print(f"Ready to use store: {name}")
    
    # Cleanup
    manager.cleanup_all()
    '''