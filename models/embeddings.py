"""
Embedding Generation Module
Clean, production-ready code for models/embeddings.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        print(f"🔄 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded successfully! Dimension: {self.dimension}")
    
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single string or list of strings
        
        Returns:
            numpy array of shape (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )
        
        return embeddings
    
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for search query
        
        Args:
            query: Search query string
        
        Returns:
            1D numpy array
        """
        return self.embed_text(query)[0]
    
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
        
        Returns:
            Similarity score between -1 and 1
        """
        return float(np.dot(vec1, vec2))
    
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


# Global singleton instance
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get or create global embedding generator
    Singleton pattern to avoid reloading model
    """
    global _embedding_generator
    
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    
    return _embedding_generator


# Test function
def test_embeddings():
    """Test embedding functionality"""
    
    print("\n" + "="*60)
    print("🧪 TESTING EMBEDDINGS")
    print("="*60 + "\n")
    
    embedder = get_embedding_generator()
    
    # Test 1: Single text
    print("Test 1: Single Text Embedding")
    print("-"*60)
    text = "Machine learning is transforming technology"
    emb = embedder.embed_text(text)
    print(f"Input: {text}")
    print(f"Shape: {emb.shape}")
    print(f"Sample values: {emb[0][:5]}")
    print()
    
    # Test 2: Batch embeddings
    print("Test 2: Batch Embeddings")
    print("-"*60)
    texts = [
        "Python is a programming language",
        "Java is used for enterprise applications",
        "Machine learning uses Python extensively"
    ]
    embs = embedder.embed_text(texts)
    print(f"Batch size: {len(texts)}")
    print(f"Output shape: {embs.shape}")
    print()
    
    # Test 3: Similarity
    print("Test 3: Cosine Similarity")
    print("-"*60)
    query = "Python programming"
    query_emb = embedder.embed_query(query)
    
    for i, txt in enumerate(texts):
        sim = embedder.cosine_similarity(query_emb, embs[i])
        print(f"'{query}' vs '{txt}'")
        print(f"Similarity: {sim:.4f}\n")
    
    print("="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_embeddings()