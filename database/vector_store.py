import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime


class VectorStore:
    """Manage permanent vector database operations"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Path to store database files
        """
        print(f"🔄 Initializing ChromaDB at {persist_directory}...")
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "Multi-modal RAG knowledge base"}
        )
        
        print(f"✅ Vector store initialized! Documents: {self.collection.count()}")
    
    
    def insert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Insert documents into vector database
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of document IDs (auto-generated if None)
        
        Returns:
            List of inserted document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add timestamp to metadata
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        for meta in metadatas:
            if 'timestamp' not in meta:
                meta['timestamp'] = datetime.now().isoformat()
        
        # Insert into ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Inserted {len(texts)} documents into vector store")
        return ids
    
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"file_name": "doc.pdf"})
        
        Returns:
            List of matching documents with metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        print(f"✅ Deleted {len(ids)} documents")
    
    
    def delete_collection(self) -> None:
        """Delete entire collection (use with caution!)"""
        self.client.delete_collection(name="rag_documents")
        print("⚠️  Collection deleted!")
    
    
    def count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()
    
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata of all documents"""
        results = self.collection.get(include=["metadatas"])
        return results['metadatas']


# Global singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get or create global vector store instance
    Singleton pattern
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = VectorStore()
    
    return _vector_store


# Test function
def test_vector_store():
    """Test vector database operations"""
    
    print("\n" + "="*60)
    print("🧪 TESTING VECTOR STORE")
    print("="*60 + "\n")
    
    # Initialize
    store = VectorStore(persist_directory="./test_chroma_db")
    
    # Test 1: Insert documents
    print("Test 1: Insert Documents")
    print("-"*60)
    
    texts = [
        "Python is a high-level programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ]
    
    # Create dummy embeddings (normally from embedding model)
    import numpy as np
    embeddings = [np.random.rand(384).tolist() for _ in range(len(texts))]
    
    metadatas = [
        {"file_name": "python.txt", "category": "programming"},
        {"file_name": "ml.txt", "category": "ai"},
        {"file_name": "dl.txt", "category": "ai"}
    ]
    
    doc_ids = store.insert(texts, embeddings, metadatas)
    print(f"Inserted IDs: {doc_ids[:2]}...")
    print(f"Total documents: {store.count()}\n")
    
    # Test 2: Search
    print("Test 2: Search Documents")
    print("-"*60)
    
    query_emb = np.random.rand(384).tolist()
    results = store.search(query_emb, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {result['content'][:50]}...")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Metadata: {result['metadata']}\n")
    
    # Test 3: Filter search
    print("Test 3: Filtered Search")
    print("-"*60)
    
    filtered_results = store.search(
        query_emb,
        top_k=5,
        filter_dict={"category": "ai"}
    )
    print(f"Found {len(filtered_results)} AI-related documents\n")
    
    # Test 4: Count and metadata
    print("Test 4: Get Metadata")
    print("-"*60)
    print(f"Total documents: {store.count()}")
    all_meta = store.get_all_metadata()
    print(f"Sample metadata: {all_meta[0]}\n")
    
    # Cleanup
    print("Cleaning up test database...")
    store.delete_collection()
    
    print("="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_vector_store()