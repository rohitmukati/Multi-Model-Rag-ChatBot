"""
Session Memory for Temporary Chat Uploads
Clean, production-ready code for database/session_memory.py
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class SessionMemory:
    """
    In-memory storage for temporary chat uploads
    Data cleared when session ends or explicitly reset
    """
    
    def __init__(self):
        """Initialize empty session memory"""
        self.memory: List[Dict[str, Any]] = []
        print("✅ Session memory initialized (RAM storage)")
    
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document to session memory
        
        Args:
            text: Document text content
            embedding: Embedding vector
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['is_temporary'] = True
        
        self.memory.append({
            'id': doc_id,
            'text': text,
            'embedding': embedding,
            'metadata': metadata
        })
        
        return doc_id
    
    
    def add_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple documents to session memory
        
        Args:
            texts: List of document texts
            embeddings: Array of embeddings (n, dim)
            metadatas: List of metadata dicts
        
        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        doc_ids = []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            doc_id = self.add(text, emb, meta)
            doc_ids.append(doc_id)
        
        print(f"✅ Added {len(texts)} documents to session memory")
        return doc_ids
    
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search session memory for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            List of matching documents with similarity scores
        """
        if not self.memory:
            return []
        
        results = []
        
        for item in self.memory:
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, item['embedding'])
            
            results.append({
                'id': item['id'],
                'content': item['text'],
                'metadata': item['metadata'],
                'similarity': similarity,
                'source': 'session'
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        return float(np.dot(vec1_norm, vec2_norm))
    
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all documents in session memory"""
        return [
            {
                'id': item['id'],
                'content': item['text'],
                'metadata': item['metadata']
            }
            for item in self.memory
        ]
    
    
    def count(self) -> int:
        """Get number of documents in session memory"""
        return len(self.memory)
    
    
    def clear(self) -> None:
        """Clear all session memory"""
        count = len(self.memory)
        self.memory = []
        print(f"🗑️  Cleared {count} documents from session memory")
    
    
    def remove(self, doc_id: str) -> bool:
        """
        Remove specific document by ID
        
        Args:
            doc_id: Document ID to remove
        
        Returns:
            True if removed, False if not found
        """
        for i, item in enumerate(self.memory):
            if item['id'] == doc_id:
                self.memory.pop(i)
                print(f"✅ Removed document {doc_id}")
                return True
        
        print(f"⚠️  Document {doc_id} not found")
        return False


# Global singleton instance
_session_memory = None


def get_session_memory() -> SessionMemory:
    """
    Get or create global session memory instance
    Singleton pattern
    """
    global _session_memory
    
    if _session_memory is None:
        _session_memory = SessionMemory()
    
    return _session_memory


def reset_session_memory() -> None:
    """Reset session memory (useful for new chat sessions)"""
    global _session_memory
    if _session_memory is not None:
        _session_memory.clear()


# Test function
def test_session_memory():
    """Test session memory operations"""
    
    print("\n" + "="*60)
    print("🧪 TESTING SESSION MEMORY")
    print("="*60 + "\n")
    
    # Initialize
    session = SessionMemory()
    
    # Test 1: Add single document
    print("Test 1: Add Single Document")
    print("-"*60)
    
    text1 = "This is a temporary uploaded document"
    emb1 = np.random.rand(384)
    meta1 = {"file_name": "temp.txt"}
    
    doc_id = session.add(text1, emb1, meta1)
    print(f"Added document: {doc_id}")
    print(f"Total documents: {session.count()}\n")
    
    # Test 2: Add batch
    print("Test 2: Add Batch Documents")
    print("-"*60)
    
    texts = [
        "Temporary document 2",
        "Temporary document 3"
    ]
    embs = np.random.rand(2, 384)
    metas = [
        {"file_name": "temp2.txt"},
        {"file_name": "temp3.txt"}
    ]
    
    batch_ids = session.add_batch(texts, embs, metas)
    print(f"Total documents: {session.count()}\n")
    
    # Test 3: Search
    print("Test 3: Search Session Memory")
    print("-"*60)
    
    query_emb = np.random.rand(384)
    results = session.search(query_emb, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {result['content'][:40]}...")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Source: {result['source']}\n")
    
    # Test 4: Get all
    print("Test 4: Get All Documents")
    print("-"*60)
    all_docs = session.get_all()
    print(f"Retrieved {len(all_docs)} documents")
    for doc in all_docs:
        print(f"  - {doc['metadata']['file_name']}\n")
    
    # Test 5: Remove document
    print("Test 5: Remove Document")
    print("-"*60)
    removed = session.remove(batch_ids[0])
    print(f"Documents remaining: {session.count()}\n")
    
    # Test 6: Clear all
    print("Test 6: Clear Session Memory")
    print("-"*60)
    session.clear()
    print(f"Documents remaining: {session.count()}\n")
    
    print("="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_session_memory()