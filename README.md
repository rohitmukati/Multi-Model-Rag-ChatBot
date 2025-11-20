# Multi-Model RAG Chatbot

A modular, production-ready Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **ChromaDB**, and **LLM integration** (Gemini/OpenAI). Supports permanent vector database creation, semantic search, full RAG querying, statistics, and cleanup operations.

---

## 🚀 Features

- **Permanent RAG Vector Database** - Upload PDF, DOCX, TXT, images, audio, and video files
- **Semantic Search** - Embed queries and search the vector store with configurable top-k results
- **Full RAG Querying** - Retrieve context, generate LLM responses with cited sources
- **Database Statistics** - Detailed file-wise grouping, chunk counts, and timestamps
- **Database Management** - Full cleanup and rebuild capabilities
- **Minimal Web UI** - Built with Jinja2 templates for easy interaction
- **Multi-Format Support** - Extract text from PDFs, DOCX, images (OCR), audio (Whisper), and video

---

## 📁 Project Structure

```
MULTI-MODEL-RAG-CHATBOT/
│
├── app.py                         # FastAPI application entry point
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup configuration
├── README.md                      # Project documentation
├── .env                          # Environment variables (API keys)
│
├── config/                        # Configuration files
│
├── database/
│   ├── vector_store.py           # ChromaDB wrapper and operations
│   └── session_memory.py         # Temporary session memory management
│
├── models/
│   ├── embeddings.py             # Embedding generation (Gemini/OpenAI)
│   └── llm.py                    # LLM response generator
│
├── routes/
│   └── db_routes.py              # RAG API endpoints
│
├── services/
│   ├── extractors.py             # Multi-format text extraction
│   ├── chunking.py               # Text chunking with overlap
│   ├── pipeline_permanent.py    # Permanent database pipeline
│   ├── pipeline_session.py      # Session-based RAG (optional)
│   └── query_processor.py       # Query processing logic
│
├── temp_uploads/                 # Temporary file storage
│
└── templates/
    └── index.html                # Frontend UI
```

---

## Demo video 
<h1> <a href="https://www.loom.com/share/5211bffd2fd04960b0149bd3436102e6">Demo Video</a></h1>

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohitmukati/Multi-Model-Rag-ChatBot.git
   cd MULTI-MODEL-RAG-CHATBOT
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional
   ```

6. **Run the FastAPI server**
   ```bash
   uvicorn app:app --reload
   ```

7. **Access the application**
   - **Web UI:** http://127.0.0.1:8000/
   - **API Documentation (Swagger):** http://127.0.0.1:8000/docs
   - **Alternative API Docs (ReDoc):** http://127.0.0.1:8000/redoc

---

## 📡 API Endpoints

### 1. Build Database
**`POST /db/build`**

Upload files to create/update the vector database.

**Request:**
- `files`: List of files (PDF, DOCX, TXT, images, audio, video)

**Response:**
```json
{
  "status": "success",
  "files_processed": 5,
  "chunks_created": 142
}
```

---

### 2. Semantic Search
**`POST /db/search`**

Search the vector database for relevant chunks.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Machine learning is...",
      "metadata": {
        "file_path": "ml_basics.pdf",
        "chunk_index": 3
      },
      "similarity_score": 0.89
    }
  ]
}
```

---

### 3. Full RAG Query
**`POST /db/query`**

Perform a complete RAG operation with LLM-generated response.

**Request Body:**
```json
{
  "query": "Explain neural networks",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "Neural networks are computational models...",
  "sources": [
    {
      "file": "deep_learning.pdf",
      "chunk": "Neural networks consist of..."
    }
  ]
}
```

---

### 4. Database Statistics
**`GET /db/stats`**

Retrieve detailed statistics about the vector database.

**Response:**
```json
{
  "total_chunks": 487,
  "total_files": 12,
  "files": [
    {
      "file_path": "document1.pdf",
      "chunk_count": 45,
      "created_at": "2025-01-15T10:30:00"
    }
  ]
}
```

---

### 5. Delete Database
**`DELETE /db/delete`**

Remove all data from the vector database.

**Response:**
```json
{
  "status": "success",
  "message": "Database deleted successfully"
}
```

---

## 🏗️ Technical Architecture

### Embedding Layer
- **Gemini Embeddings** (primary)
- **OpenAI Embeddings** (alternative)
- Extensible to custom embedding models

### Vector Store
- **ChromaDB** with persistent storage
- Metadata-rich chunks including:
  - `file_path` - Source document
  - `project_id` - Organizational grouping
  - `timestamp` - Creation time
  - `chunk_index` - Position in document

### Text Extraction Pipeline
- **PDFs** - PyPDF2
- **DOCX** - python-docx
- **Text Files** - Direct reading
- **Images** - OCR (Tesseract/EasyOCR)
- **Audio** - Whisper STT (Speech-to-Text)
- **Video** - Audio extraction + Whisper

### Chunking Strategy
- **Chunk Size:** 512 tokens
- **Overlap:** 50 tokens
- Preserves context across chunk boundaries

### RAG Pipeline Flow
```
Query → Embedding → Vector Search → Context Retrieval → LLM Prompt → Response
```

---

## 🎯 Design Principles

- **Zero Permanent File Storage** - Files processed and discarded
- **Fault Tolerant** - Graceful handling of empty databases
- **Modular Architecture** - Clean separation of concerns
- **Production Ready** - Structured for containerization and scaling
- **Safe Operations** - No data loss from failed operations

---

## 🔮 Future Enhancements

- [ ] Session-based RAG for temporary contexts
- [ ] Reranking layer for improved relevance
- [ ] Hybrid search (vector + keyword)
- [ ] Background task workers
- [ ] Multi-LLM routing and fallbacks
- [ ] Conversation history management
- [ ] Advanced metadata filtering
- [ ] Batch processing endpoints
- [ ] User authentication and multi-tenancy

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Rohit Mukati**

- GitHub: [@rohitmukati](https://github.com/rohitmukati)

---

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- ChromaDB for vector storage
- Google Gemini and OpenAI for LLM capabilities
- The open-source community

---

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

**Happy Building! 🚀**