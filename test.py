"""
Database Inspector - View existing chunks in ChromaDB
Run: python test_db_inspector.py
"""

from database.vector_store import get_vector_store

def inspect_database():
    """Inspect all chunks in database"""
    
    print("\n" + "="*70)
    print("🔍 DATABASE INSPECTOR")
    print("="*70 + "\n")
    
    # Get vector store
    store = get_vector_store()
    
    # Total count
    total = store.count()
    print(f"📊 Total chunks in database: {total}\n")
    
    if total == 0:
        print("❌ Database is empty!")
        return
    
    # Get all documents with content
    all_results = store.collection.get(include=["documents", "metadatas"])
    
    # Group by file
    files_dict = {}
    for i, meta in enumerate(all_results['metadatas']):
        file_name = meta.get('file_name', 'unknown')
        if file_name not in files_dict:
            files_dict[file_name] = []
        files_dict[file_name].append({
            'metadata': meta,
            'content': all_results['documents'][i]
        })
    
    # Display chunks by file
    for file_name, chunks_data in files_dict.items():
        print("="*70)
        print(f"📄 File: {file_name}")
        print(f"📦 Type: {chunks_data[0]['metadata'].get('file_type', 'unknown')}")
        print(f"🔢 Total Chunks: {len(chunks_data)}")
        print(f"🆔 Project ID: {chunks_data[0]['metadata'].get('project_id', 'N/A')}")
        print("-"*70)
        
        # Show all chunks for this file
        for i, chunk_data in enumerate(chunks_data, 1):
            meta = chunk_data['metadata']
            content = chunk_data['content']
            
            # Get first 50 words
            words = content.split()[:50]
            preview = ' '.join(words)
            if len(content.split()) > 50:
                preview += "..."
            
            print(f"\n  Chunk {i}:")
            print(f"    - Chunk Index: {meta.get('chunk_index', 'N/A')}")
            print(f"    - Timestamp: {meta.get('timestamp', 'N/A')[:19]}")
            print(f"    - File Path: {meta.get('file_path', 'N/A')}")
            print(f"    - Preview (first 50 words):")
            print(f"      {preview}")
        
        print()
    
    print("="*70)
    print(f"✅ Inspection Complete! {len(files_dict)} files, {total} chunks")
    print("="*70 + "\n")


if __name__ == "__main__":
    inspect_database()