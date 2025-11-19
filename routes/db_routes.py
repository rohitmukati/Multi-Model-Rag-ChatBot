from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
import os
from services.pipeline_permanent import build_rag_database

router = APIRouter(prefix="/db", tags=["RAG Database"])


@router.post("/build")
async def build_database(
    files: List[UploadFile] = File(...)
):
    """
    Build Permanent RAG Database from uploaded files.
    Saves files temporarily, sends paths to RAG builder, deletes after processing.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    saved_paths = []

    try:
        # Save uploaded files locally
        for f in files:
            save_path = os.path.join(temp_dir, f.filename)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(save_path)

        # Build database
        result = build_rag_database(
            file_paths=saved_paths,
            chunk_size=512,
            overlap=50,
            project_id="api_project"
        )

        return result

    finally:
        # Cleanup temp files
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)


@router.post("/search")
async def search_documents(
    query: str,
    top_k: int = 5
):
    """
    Search documents from permanent vector database.
    Safe even if collection or DB is empty / missing.
    """
    from models.embeddings import get_embedding_generator
    from database.vector_store import get_vector_store
    import chromadb

    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")

    # Generate embedding
    embedder = get_embedding_generator()
    query_embedding = embedder.embed_query(query)

    # Vector store
    store = get_vector_store()

    # Fail-safe: no documents
    try:
        total_docs = store.count()
    except Exception:
        # Chroma folder or collection missing
        return {"query": query, "results": []}

    if total_docs == 0:
        return {"query": query, "results": []}

    # Try search safely
    try:
        results = store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
    except Exception:
        # Query failure (collection corrupted / missing)
        return {"query": query, "results": []}

    return {
        "query": query,
        "results": results
    }


@router.delete("/delete")
async def delete_database():
    """
    Delete the entire permanent vector database collection.
    """
    from database.vector_store import get_vector_store

    store = get_vector_store()
    store.delete_collection()

    return {"status": "success", "message": "Vector database collection deleted"}

@router.get("/stats")
async def database_stats():
    """
    Get detailed statistics of the vector database.
    Groups chunks by file and shows proper file-level stats.
    Safe even if DB is empty.
    """
    from database.vector_store import get_vector_store

    store = get_vector_store()

    try:
        total_docs = store.count()
    except Exception:
        return {
            "total_documents": 0,
            "total_files": 0,
            "files": [],
            "message": "Database or collection missing"
        }

    # Fetch all metadata (one metadata per chunk)
    try:
        all_meta = store.get_all_metadata()
    except Exception:
        all_meta = []

    if not all_meta:
        return {
            "total_documents": 0,
            "total_files": 0,
            "files": []
        }

    # Group by file_path
    files = {}
    for meta in all_meta:
        file_path = meta.get("file_path", "unknown")
        if file_path not in files:
            files[file_path] = {
                "file_path": file_path,
                "project_id": meta.get("project_id"),
                "total_chunks": 0,
                "chunk_indices": [],
                "first_timestamp": meta.get("timestamp"),
                "last_timestamp": meta.get("timestamp")
            }

        files[file_path]["total_chunks"] += 1
        files[file_path]["chunk_indices"].append(meta.get("chunk_index"))

        # Update timestamps
        if meta.get("timestamp") < files[file_path]["first_timestamp"]:
            files[file_path]["first_timestamp"] = meta.get("timestamp")

        if meta.get("timestamp") > files[file_path]["last_timestamp"]:
            files[file_path]["last_timestamp"] = meta.get("timestamp")

    return {
        "total_documents": total_docs,
        "total_files": len(files),
        "files": list(files.values())
    }

@router.post("/query")
async def rag_query(
    query: str,
    top_k: int = 5   # optional + default
):
    """
    Full RAG flow:
    1. Internally call /db/search
    2. Build context from retrieved chunks
    3. Call LLM with context
    4. Return final answer
    """
    from models.llm import generate_response

    # Step 1: internally call the existing search function
    search_result = await search_documents(query=query, top_k=top_k)

    results = search_result.get("results", [])

    # Step 2: build context string
    if results:
        context = "\n\n".join([
            f"[Source: {r['metadata'].get('file_path','unknown')}] {r['content']}"
            for r in results
        ])
    else:
        context = ""

    # Step 3: call LLM with context
    answer = generate_response(query=query, context=context)

    # Step 4: return full RAG pipeline result
    return {
        "query": query,
        "answer": answer,
        "context_used": context,
        "retrieved_chunks": results
    }
