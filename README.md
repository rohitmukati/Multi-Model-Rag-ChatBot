# Multi-Model RAG Chatbot – Backend (FastAPI + ChromaDB + Gemini/OpenAI)

A modular, production-ready Retrieval-Augmented Generation (RAG) backend built with FastAPI, ChromaDB, and LLM integration.  
Supports permanent vector DB creation, semantic search, full RAG querying, stats, and cleanup.  
Includes a minimal frontend using Jinja2 templates.

============================================================
PROJECT STRUCTURE
============================================================

MULTI-MODEL-RAG-CHATBOT/
│
├── app.py                         # FastAPI application entry
├── requirements.txt
├── setup.py
├── README.md
│
├── config/
│
├── database/
│   ├── vector_store.py            # ChromaDB wrapper
│   └── session_memory.py          # Temporary session memory
│
├── models/
│   ├── embeddings.py              # Embedding generator
│   └── llm.py                     # LLM response generator
│
├── routes/
│   └── db_routes.py               # RAG API endpoints
│
├── services/
│   ├── extractors.py              # PDF/DOCX/Text/Image/Whisper extractors
│   ├── chunking.py                # Chunking logic
│   ├── pipeline_permanent.py      # Permanent RAG DB builder
│   ├── pipeline_session.py        # Optional session RAG
│   └── query_processor.py         # Query logic
│
├── temp_uploads/                  # Temporary uploads
│
├── templates/
│   └── index.html                 # Frontend UI
│
└── .env                           # API keys


============================================================
FEATURES
============================================================

1. Permanent RAG Vector Database
   - Upload PDF, DOCX, TXT, images, audio, video.
   - Extraction → Chunking → Embeddings → ChromaDB store.

2. Semantic Search
   - Embeds query and searches vector store.
   - Fully safe if DB missing or empty.

3. Full RAG Querying
   - Search → Build context → LLM → Final answer.

4. Detailed Database Stats
   - File-wise grouping, chunk-count, timestamps.

5. Full Database Cleanup

6. Minimal UI using Jinja2 templates.


============================================================
API ENDPOINTS
============================================================

1. Build Database  
POST /db/build  
files: List[UploadFile]

2. Search  
POST /db/search  
query: str  
top_k: int

3. Full RAG Query  
POST /db/query  
query: str  
top_k: int

4. Delete Database  
DELETE /db/delete

5. DB Stats  
GET /db/stats


============================================================
INSTALLATION
============================================================

1. Clone the project:
   git clone https://github.com/rohitmukati/Multi-Model-Rag-ChatBot.git
   cd MULTI-MODEL-RAG-CHATBOT

2. Create Virtual Environment:
   python3 -m venv venv

3. Activate Virtual Environment:
   - Windows:
       venv\Scripts\activate
   - Linux / macOS:
       source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

5. Create a .env file:
   GEMINI_API_KEY=your_key

6. Run the FastAPI server:
   uvicorn app:app --reload

7. Open in browser:
   http://127.0.0.1:8000/

8. API Docs (Swagger):
   http://127.0.0.1:8000/docs


============================================================
TECHNICAL OVERVIEW
============================================================

Embedding Layer:
- Gemini embeddings / OpenAI embeddings / custom models.

Vector Store:
- ChromaDB persistent DB.
- Metadata-rich chunks (file_path, project_id, timestamp, indices).

Extractor Pipeline:
- PDFs (PyPDF2)
- DOCX (python-docx)
- Text files
- Images (OCR)
- Audio (Whisper STT)

Chunking Engine:
- chunk_size=512  
- overlap=50  

RAG Pipeline:
- Retrieve → Build Context → LLM → Final Answer.


============================================================
DESIGN PRINCIPLES
============================================================

- No file stored permanently.
- Fault-tolerant: empty DB → empty response.
- Modular, clean, production-friendly structure.
- Ready for containerization and scaling.


============================================================
FUTURE EXTENSIONS
============================================================

- Session RAG
- Reranking layer
- Hybrid search
- Background workers
- Multi-LLM routing
