"""
PIPELINE 2: Session Pipeline for Temporary Chat Uploads
Handle temporary file uploads during chat sessions
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.extractors import FileExtractor
from services.chunking import chunk_document
from models.embeddings import get_embedding_generator
from database.session_memory import get_session_memory, reset_session_memory


class SessionPipeline:
    """Handle temporary file uploads in chat sessions"""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session pipeline
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.extractor = FileExtractor()
        self.embedder = get_embedding_generator()
        self.session_memory = get_session_memory()
        
        print(f"🔵 Session Pipeline initialized: {self.session_id}")
    
    
    def add_files_to_session(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Add temporary files to current session
        
        Args:
            file_paths: List of file paths to add
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
        
        Returns:
            Processing results
        """
        print(f"\n{'='*70}")
        print(f"📤 ADDING FILES TO SESSION: {self.session_id}")
        print(f"{'='*70}\n")
        
        successful_files = 0
        failed_files = []
        total_chunks = 0
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
            print("-" * 70)
            
            try:
                # Extract content
                print("  📄 Extracting...")
                result = self.extractor.extract(file_path)
                
                if not result.get('success'):
                    print(f"  ❌ Failed: {result.get('error')}\n")
                    failed_files.append({'file': file_path, 'error': result.get('error')})
                    continue
                
                # Handle video files
                if result.get('file_type') == 'video':
                    captions_text = "\n".join([
                        f"[{c['time']}s] {c['caption']}" 
                        for c in result.get('captions', [])
                    ])
                    audio_text = result.get('audio_transcript', '')
                    extracted_text = f"{captions_text}\n\nAudio Transcript:\n{audio_text}"
                else:
                    extracted_text = result.get('text', '')
                
                if not extracted_text.strip():
                    print("  ⚠️  No content\n")
                    continue
                
                print(f"  ✅ Extracted {len(extracted_text)} chars")
                
                # Chunk
                print("  ✂️  Chunking...")
                chunks = chunk_document(
                    text=extracted_text,
                    file_name=os.path.basename(file_path),
                    file_type=result.get('file_type', 'unknown'),
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                if not chunks:
                    print("  ⚠️  No chunks created\n")
                    continue
                
                print(f"  ✅ Created {len(chunks)} chunks")
                
                # Generate embeddings
                print("  🧮 Embedding...")
                chunk_texts = [c['text'] for c in chunks]
                embeddings = self.embedder.embed_text(chunk_texts)
                
                # Add to session memory
                metadatas = []
                for j, chunk in enumerate(chunks):
                    meta = chunk.get('metadata', {})
                    meta.update({
                        'chunk_index': j,
                        'session_id': self.session_id,
                        'file_path': file_path
                    })
                    metadatas.append(meta)
                
                doc_ids = self.session_memory.add_batch(chunk_texts, embeddings, metadatas)
                
                successful_files += 1
                total_chunks += len(chunks)
                print(f"  ✅ Added to session\n")
            
            except Exception as e:
                print(f"  ❌ Error: {str(e)}\n")
                failed_files.append({'file': file_path, 'error': str(e)})
        
        # Summary
        print("="*70)
        print("📊 SESSION UPDATE SUMMARY")
        print("="*70)
        print(f"Session ID: {self.session_id}")
        print(f"Files processed: {len(file_paths)}")
        print(f"Successful: {successful_files}")
        print(f"Failed: {len(failed_files)}")
        print(f"Total chunks added: {total_chunks}")
        print(f"Session size: {self.session_memory.count()} chunks")
        print("="*70 + "\n")
        
        return {
            'success': True,
            'session_id': self.session_id,
            'total_files': len(file_paths),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_chunks': total_chunks,
            'session_size': self.session_memory.count()
        }
    
    
    def search_session(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search in current session memory
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of search results
        """
        print(f"\n🔍 Searching session: {self.session_id}")
        print(f"Query: '{query}'")
        print("-" * 70)
        
        # Generate query embedding
        query_emb = self.embedder.embed_query(query)
        
        # Search session memory
        results = self.session_memory.search(query_emb, top_k=top_k)
        
        print(f"✅ Found {len(results)} results\n")
        
        return results
    
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        all_docs = self.session_memory.get_all()
        
        # Group by file
        files = {}
        for doc in all_docs:
            file_name = doc['metadata'].get('file_name', 'unknown')
            if file_name not in files:
                files[file_name] = 0
            files[file_name] += 1
        
        return {
            'session_id': self.session_id,
            'total_chunks': self.session_memory.count(),
            'files': files,
            'file_count': len(files)
        }
    
    
    def clear_session(self) -> None:
        """Clear current session memory"""
        print(f"\n🗑️  Clearing session: {self.session_id}")
        self.session_memory.clear()
        print("✅ Session cleared\n")
    
    
    def remove_file_from_session(self, file_name: str) -> int:
        """
        Remove all chunks from a specific file
        
        Args:
            file_name: Name of file to remove
        
        Returns:
            Number of chunks removed
        """
        all_docs = self.session_memory.get_all()
        removed_count = 0
        
        for doc in all_docs:
            if doc['metadata'].get('file_name') == file_name:
                if self.session_memory.remove(doc['id']):
                    removed_count += 1
        
        print(f"✅ Removed {removed_count} chunks from '{file_name}'")
        return removed_count


# Global pipeline instance
_session_pipeline = None


def get_session_pipeline(session_id: Optional[str] = None) -> SessionPipeline:
    """Get or create session pipeline"""
    global _session_pipeline
    if _session_pipeline is None or (session_id and _session_pipeline.session_id != session_id):
        _session_pipeline = SessionPipeline(session_id)
    return _session_pipeline


def create_new_session(session_id: Optional[str] = None) -> SessionPipeline:
    """Create new session (clears old session memory)"""
    reset_session_memory()
    return SessionPipeline(session_id)


def add_temp_files(file_paths: List[str], session_id: Optional[str] = None) -> Dict[str, Any]:
    """Add temporary files to session"""
    pipeline = get_session_pipeline(session_id)
    return pipeline.add_files_to_session(file_paths)


def search_session(query: str, session_id: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search in session memory"""
    pipeline = get_session_pipeline(session_id)
    return pipeline.search_session(query, top_k)


def get_session_stats(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Get session statistics"""
    pipeline = get_session_pipeline(session_id)
    return pipeline.get_session_stats()


def clear_session(session_id: Optional[str] = None) -> None:
    """Clear session memory"""
    pipeline = get_session_pipeline(session_id)
    pipeline.clear_session()


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("🧪 TESTING SESSION PIPELINE")
    print("="*70 + "\n")
    
    # Create new session
    session = create_new_session("test_session_001")
    
    # Add files to session
    test_files = [
        "Data/hair.txt",
        "Data/Screenshot 2025-11-15 142915.png"
    ]
    
    result = session.add_files_to_session(test_files)
    
    # Search session
    search_results = session.search_session("hair care", top_k=2)
    
    print("\n🔍 Search Results:")
    for i, res in enumerate(search_results, 1):
        print(f"\nResult {i}:")
        print(f"  File: {res['metadata'].get('file_name')}")
        print(f"  Similarity: {res['similarity']:.4f}")
        print(f"  Content: {res['content'][:100]}...")
    
    # Get stats
    stats = session.get_session_stats()
    print(f"\n📊 Session Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Files: {stats['file_count']}")
    
    # Clear session
    session.clear_session()