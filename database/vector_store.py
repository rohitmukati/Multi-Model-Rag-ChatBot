import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime


class VectorStore:
    """Manage permanent vector database operations"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        print(f"🔄 Initializing ChromaDB at {persist_directory}...")

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Always ensure collection exists
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "Multi-modal RAG knowledge base"}
        )

        print(f"✅ Vector store initialized! Documents: {self.collection.count()}")

    # ----------------------------------------------------------------------
    # INSERT
    # ----------------------------------------------------------------------
    def insert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:

        # Auto-recover broken/missing collection
        try:
            _ = self.collection.count()
        except Exception:
            print("⚠️ Collection missing. Recreating...")
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"description": "Multi-modal RAG knowledge base"}
            )

        # Generate IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Ensure metadata
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        for meta in metadatas:
            if 'timestamp' not in meta:
                meta['timestamp'] = datetime.now().isoformat()

        # Insert
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"✅ Inserted {len(texts)} documents into vector store")
        return ids

    # ----------------------------------------------------------------------
    # SEARCH
    # ----------------------------------------------------------------------
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        # Safe access
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            print("⚠️ Search failed — collection missing. Returning empty results.")
            return []

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i],
            })

        return formatted_results

    # ----------------------------------------------------------------------
    # DELETE DOCUMENTS
    # ----------------------------------------------------------------------
    def delete(self, ids: List[str]) -> None:
        try:
            self.collection.delete(ids=ids)
            print(f"✅ Deleted {len(ids)} documents")
        except Exception:
            print("⚠️ Could not delete — collection missing.")

    # ----------------------------------------------------------------------
    # DELETE COLLECTION (SAFE)
    # ----------------------------------------------------------------------
    def delete_collection(self) -> None:
        """Safely delete and recreate collection"""

        try:
            self.client.delete_collection(name="rag_documents")
            print("⚠️ Collection deleted!")
        except Exception:
            print("⚠️ Collection missing — skipping delete.")

        # Always recreate a fresh collection
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "Multi-modal RAG knowledge base"}
        )
        print("✅ Fresh empty collection created!")

    # ----------------------------------------------------------------------
    # COUNT
    # ----------------------------------------------------------------------
    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            print("⚠️ Count failed — collection missing.")
            return 0

    # ----------------------------------------------------------------------
    # METADATA
    # ----------------------------------------------------------------------
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        try:
            results = self.collection.get(include=["metadatas"])
            return results['metadatas']
        except Exception:
            print("⚠️ Metadata fetch failed — collection missing.")
            return []


# ----------------------------------------------------------------------
# SINGLETON
# ----------------------------------------------------------------------
_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store


# ----------------------------------------------------------------------
# TEST (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    store = get_vector_store()
    print(store.count())
