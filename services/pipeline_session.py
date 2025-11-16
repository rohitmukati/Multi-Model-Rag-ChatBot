"""
PIPELINE 2: Chat with RAG + Temporary Uploads
Handle chat queries with database search + session memory
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.embeddings import get_embedding_generator
from models.llm import generate_response
from database.vector_store import get_vector_store
from database.session_memory import get_session_memory
from services.extractors import FileExtractor
from services.chunking import chunk_document


class ChatPipeline:
    """Handle chat queries with RAG retrieval"""
    
    def __init__(self):
        self.embedder = get_embedding_generator()
        self.vector_store = get_vector_store()
        self.session_memory = get_session_memory()
        self.extractor = FileExtractor()
    
    
    def process_temp_files(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Process temporary uploaded files (not stored in DB)
        
        Args:
            file_paths: List of temporary file paths
            chunk_size: Chunk size
            overlap: Overlap size
        
        Returns:
            Processing results
        """
        print(f"\n📤 Processing {len(file_paths)} temporary files...")
        
        successful = 0
        failed = []
        
        for file_path in file_paths:
            try:
                # Extract content
                result = self.extractor.extract(file_path)
                
                if not result.get('success'):
                    failed.append({'file': file_path, 'error': result.get('error')})
                    continue
                
                # Get text
                if result.get('file_type') == 'video':
                    text = result.get('text', '')
                else:
                    text = result.get('text', '')
                
                if not text.strip():
                    continue
                
                # Chunk text
                chunks = chunk_document(
                    text=text,
                    file_name=result.get('file_name'),
                    file_type=result.get('file_type'),
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                # Generate embeddings
                chunk_texts = [c['text'] for c in chunks]
                embeddings = self.embedder.embed_text(chunk_texts)
                
                # Add to session memory
                for i, chunk_text in enumerate(chunk_texts):
                    metadata = chunks[i].get('metadata', {})
                    metadata['is_temporary'] = True
                    metadata['uploaded_at'] = datetime.now().isoformat()
                    
                    self.session_memory.add(
                        text=chunk_text,
                        embedding=embeddings[i],
                        metadata=metadata
                    )
                
                successful += 1
                print(f"  ✅ {result.get('file_name')} - {len(chunks)} chunks")
            
            except Exception as e:
                failed.append({'file': file_path, 'error': str(e)})
        
        return {
            'success': True,
            'processed': successful,
            'failed': failed,
            'session_size': self.session_memory.count()
        }
    
    
    def search_and_generate(
        self,
        query: str,
        top_k: int = 5,
        use_session: bool = True
    ) -> Dict[str, Any]:
        """
        Search database + session memory and generate response
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            use_session: Include session memory in search
        
        Returns:
            Generated response with sources
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search permanent database
        db_results = self.vector_store.search(
            query_embedding.tolist(),
            top_k=top_k
        )
        
        print(f"  📚 Found {len(db_results)} results from database")
        
        # Search session memory (if enabled)
        session_results = []
        if use_session and self.session_memory.count() > 0:
            session_results = self.session_memory.search(
                query_embedding,
                top_k=3
            )
            print(f"  💾 Found {len(session_results)} results from session")
        
        # Combine results
        all_results = db_results + session_results
        
        # Sort by similarity
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Take top results
        top_results = all_results[:top_k]
        
        if not top_results:
            return {
                'success': True,
                'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                'sources': [],
                'query': query
            }
        
        # Prepare context for LLM
        context_parts = []
        for i, result in enumerate(top_results, 1):
            source_info = f"[Source {i}: {result['metadata'].get('file_name', 'Unknown')}]"
            content = result['content']
            context_parts.append(f"{source_info}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        print("  🤖 Generating response...")
        answer = generate_response(query, context)
        
        # Prepare sources
        sources = []
        for result in top_results:
            sources.append({
                'file_name': result['metadata'].get('file_name', 'Unknown'),
                'file_type': result['metadata'].get('file_type', 'Unknown'),
                'similarity': result.get('similarity', 0),
                'is_temporary': result['metadata'].get('is_temporary', False),
                'content_preview': result['content'][:200] + "..."
            })
        
        return {
            'success': True,
            'answer': answer,
            'sources': sources,
            'query': query,
            'total_sources': len(all_results)
        }
    
    
    def chat(
        self,
        query: str,
        temp_files: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Complete chat workflow: process temp files + search + generate
        
        Args:
            query: User question
            temp_files: Optional temporary files to process
            top_k: Number of results to retrieve
        
        Returns:
            Chat response
        """
        # Process temporary files if provided
        if temp_files:
            upload_result = self.process_temp_files(temp_files)
            if upload_result['failed']:
                print(f"  ⚠️  {len(upload_result['failed'])} files failed to process")
        
        # Search and generate response
        response = self.search_and_generate(query, top_k=top_k)
        
        return response
    
    
    def clear_session(self):
        """Clear session memory"""
        self.session_memory.clear()
        print("🗑️  Session memory cleared")


# Global instance
_chat_pipeline = None


def get_chat_pipeline() -> ChatPipeline:
    """Get or create chat pipeline instance"""
    global _chat_pipeline
    if _chat_pipeline is None:
        _chat_pipeline = ChatPipeline()
    return _chat_pipeline


def chat_with_rag(
    query: str,
    temp_files: Optional[List[str]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Simple API for chatting with RAG
    
    Args:
        query: User question
        temp_files: Optional temporary files
        top_k: Number of sources to use
    
    Returns:
        Chat response with answer and sources
    """
    pipeline = get_chat_pipeline()
    return pipeline.chat(query, temp_files, top_k)

if __name__ == "__main__":
    # Test queries
    print("\n" + "="*70)
    print("🧪 TESTING CHAT PIPELINE")
    print("="*70)
    
    test_queries = [
        "Who is Rohit Mukati?",
        "What is the age of Rohit Mukati?",
        "Where is Rohit Mukati from?",
        "What is the profession of Rohit Mukati?"
    ]
    
    # ✅ FIX: Temporary file pass karo
    temp_file = ["Data/rohit.txt"]  # List me dalo
    
    for i, query in enumerate(test_queries):
        print("\n" + "-"*70)
        
        # ✅ FIX: temp_files parameter add karo
        # Pehli query me file process hogi, baaki queries me session memory se answer milega
        result = chat_with_rag(
            query=query, 
            temp_files=temp_file if i == 0 else None,  # Sirf pehli baar upload
            top_k=3
        )
        
        if result['success']:
            print(f"\n❓ Query: {query}")
            print(f"\n💬 Answer:\n{result['answer']}")
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for j, source in enumerate(result['sources'], 1):
                print(f"  {j}. {source['file_name']} ({source['file_type']}) - Similarity: {source['similarity']:.3f}")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    print("\n" + "="*70)
    print("✅ Chat pipeline test completed!")
    print("="*70)