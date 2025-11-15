import os
from pathlib import Path

# ========================
# Folder Structure Layout
# ========================

STRUCTURE = {
    "config": ["__init__.py", "settings.py", "database.py"],
    "models": ["__init__.py", "embeddings.py", "llm.py"],
    "services": [
        "__init__.py",
        "extractors.py",
        "chunking.py",
        "pipeline_permanent.py",
        "pipeline_session.py",
        "query_processor.py",
    ],
    "database": ["__init__.py", "vector_store.py", "session_memory.py"],
    "utils": ["__init__.py", "file_handler.py", "helpers.py"],
    "static": {
        "css": ["style.css"],
        "js": ["main.js"],
        "uploads": [],
    },
    "templates": ["index.html", "chat.html", "upload.html"],
    "data": {
        "temp": []
    },
    "logs": ["app.log"],
    "tests": ["__init__.py", "test_extractors.py", "test_embeddings.py", "test_pipeline.py"],
}

TOP_FILES = {
    "app.py": "# Main application entry (FastAPI or Flask)\n",
    "requirements.txt": "# Add dependencies here\nfastapi\nuvicorn\npymilvus\ntransformers\nsentence-transformers\npaddleocr\nwhisper\nopencv-python\npython-dotenv\n",
    ".env": "# Add environment variables here\n",
}

# ========================
# Helper Functions
# ========================

def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def make_file(path: Path, content=""):
    if not path.exists():
        path.write_text(content, encoding="utf-8")

# ========================
# Build Project Structure
# ========================

def build_structure():
    root = Path.cwd()  # current folder (multimodal-rag)

    for folder, items in STRUCTURE.items():
        folder_path = root / folder

        # nested dict (static, data)
        if isinstance(items, dict):
            make_dir(folder_path)
            for subfolder, files in items.items():
                sub_path = folder_path / subfolder
                make_dir(sub_path)
                for f in files:
                    make_file(sub_path / f, "# Auto-generated file\n")

        # list of files
        elif isinstance(items, list):
            make_dir(folder_path)
            for f in items:
                if "." in f:
                    make_file(folder_path / f, "# Auto-generated file\n")
                else:
                    make_dir(folder_path / f)

    # Top level files
    for fname, content in TOP_FILES.items():
        make_file(root / fname, content)

    print("\n[✔] Project structure created successfully!")
    print("[✔] All folders and placeholder files are ready.\n")

# ========================
# Run Script
# ========================

if __name__ == "__main__":
    build_structure()
