"""
PIPELINE 1: Permanent RAG Database Builder
Build long-term knowledge base from uploaded files
"""

import os
import sys
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.extractors import FileExtractor
from services.chunking import chunk_document
from models.embeddings import get_embedding_generator
from database.vector_store import get_vector_store


class RAGDatabaseBuilder:
    """Build permanent RAG database from files"""
    
    def __init__(self):
        self.extractor = FileExtractor()
        self.embedder = get_embedding_generator()
        self.vector_store = get_vector_store()
    
    
    def build_from_files(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        overlap: int = 50,
        project_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Build RAG database from multiple files
        """
        print("\n" + "="*70)
        print("🚀 BUILDING RAG DATABASE")
        print("="*70 + "\n")
        
        total_files = len(file_paths)
        successful_files = 0
        total_chunks = 0
        failed_files = []
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        # Process each file
        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{total_files}] Processing: {os.path.basename(file_path)}")
            print("-" * 70)
            
            try:
                # Step 1: Extract content
                print("  📄 Extracting content...")
                extraction_result = self.extractor.extract(file_path)
                
                if not extraction_result.get('success'):
                    error_msg = extraction_result.get('error', 'Unknown error')
                    print(f"  ❌ Extraction failed: {error_msg}\n")
                    failed_files.append({
                        'file': file_path,
                        'error': error_msg
                    })
                    continue
                
                # Get extracted text
                if extraction_result.get('file_type') == 'video':
                    captions_text = "\n".join([
                        f"[{c['time']}s] {c['caption']}" 
                        for c in extraction_result.get('captions', [])
                    ])
                    audio_text = extraction_result.get('audio_transcript', '')
                    extracted_text = f"{captions_text}\n\nAudio Transcript:\n{audio_text}"
                else:
                    extracted_text = extraction_result.get('text', '')
                
                if not extracted_text or not extracted_text.strip():
                    print("  ⚠️  No text content extracted\n")
                    continue
                
                print(f"  ✅ Extracted {len(extracted_text)} characters")
                
                # Step 2: Chunk the text
                print("  ✂️  Chunking text...")
                chunks = chunk_document(
                    text=extracted_text,
                    file_name=os.path.basename(file_path),
                    file_type=extraction_result.get('file_type', 'unknown'),
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                if not chunks:
                    print("  ⚠️  No chunks created\n")
                    continue
                
                print(f"  ✅ Created {len(chunks)} chunks")
                
                # Step 3: Generate embeddings
                print("  🧮 Generating embeddings...")
                chunk_texts = [c['text'] for c in chunks]
                embeddings = self.embedder.embed_text(chunk_texts)
                
                print(f"  ✅ Generated {len(embeddings)} embeddings")
                
                # Step 4: Prepare metadata
                for j, chunk in enumerate(chunks):
                    # FIXED: create a fresh dict copy to avoid reusing same ref
                    metadata = dict(chunk.get('metadata', {}))

                    metadata.update({
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'project_id': project_id,
                        'timestamp': datetime.now().isoformat(),
                        'file_path': file_path
                    })
                    
                    all_chunks.append(chunk_texts[j])
                    all_embeddings.append(embeddings[j].tolist())
                    all_metadata.append(metadata)
                
                successful_files += 1
                total_chunks += len(chunks)
                print(f"  ✅ File processed successfully\n")
            
            except Exception as e:
                print(f"  ❌ Error: {str(e)}\n")
                failed_files.append({
                    'file': file_path,
                    'error': str(e)
                })
        
        # Step 5: Insert into vector database
        if all_chunks:
            print("="*70)
            print("💾 STORING IN VECTOR DATABASE")
            print("="*70 + "\n")
            
            try:
                doc_ids = self.vector_store.insert(
                    texts=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadata
                )
                
                print(f"✅ Stored {len(doc_ids)} chunks in database\n")
            
            except Exception as e:
                print(f"❌ Database storage failed: {str(e)}\n")
                return {
                    'success': False,
                    'error': f'Database storage failed: {str(e)}'
                }
        
        # Final summary
        print("="*70)
        print("📊 BUILD SUMMARY")
        print("="*70)
        print(f"Total files: {total_files}")
        print(f"Successful: {successful_files}")
        print(f"Failed: {len(failed_files)}")
        print(f"Total chunks: {total_chunks}")
        print(f"Database size: {self.vector_store.count()} documents")
        
        if failed_files:
            print("\n⚠️  Failed files:")
            for fail in failed_files:
                print(f"  - {fail['file']}: {fail['error']}")
        
        print("="*70 + "\n")
        
        return {
            'success': True,
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_chunks': total_chunks,
            'database_size': self.vector_store.count()
        }
    
    
    def get_database_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': self.vector_store.count(),
            'all_metadata': self.vector_store.get_all_metadata()
        }
    
    
    def clear_database(self) -> None:
        self.vector_store.delete_collection()
        print("⚠️  Database cleared!")


# Singleton
_builder = None


def get_rag_builder() -> RAGDatabaseBuilder:
    global _builder
    if _builder is None:
        _builder = RAGDatabaseBuilder()
    return _builder


def build_rag_database(
    file_paths: List[str],
    chunk_size: int = 512,
    overlap: int = 50,
    project_id: str = "default"
) -> Dict[str, Any]:
    builder = get_rag_builder()
    return builder.build_from_files(file_paths, chunk_size, overlap, project_id)


def get_stats() -> Dict[str, Any]:
    builder = get_rag_builder()
    return builder.get_database_stats()

