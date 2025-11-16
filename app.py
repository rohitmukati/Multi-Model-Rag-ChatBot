"""
Flask API for Multi-Modal RAG System
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime

from services.pipeline_permanent import build_rag_database, get_rag_builder
from services.pipeline_session import chat_with_rag, get_chat_pipeline
from database.vector_store import get_vector_store

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a', 'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        'message': 'Multi-Modal RAG API',
        'version': '1.0',
        'endpoints': {
            'POST /build-rag': 'Build permanent RAG database from uploaded files',
            'POST /chat': 'Chat with RAG (with optional temp files)',
            'POST /upload-temp': 'Upload temporary files to session',
            'GET /stats': 'Get database statistics',
            'DELETE /delete-db': 'Clear entire database',
            'DELETE /delete-session': 'Clear session memory',
            'GET /health': 'Health check'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        store = get_vector_store()
        db_count = store.count()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database_documents': db_count
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/build-rag', methods=['POST'])
def build_rag():
    """
    Build permanent RAG database from uploaded files
    
    Request: multipart/form-data with 'files' field
    Optional: project_id, chunk_size, overlap
    """
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Get optional parameters
        project_id = request.form.get('project_id', 'default')
        chunk_size = int(request.form.get('chunk_size', 512))
        overlap = int(request.form.get('overlap', 50))
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                saved_files.append(filepath)
            else:
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
        
        # Build RAG database
        result = build_rag_database(
            file_paths=saved_files,
            chunk_size=chunk_size,
            overlap=overlap,
            project_id=project_id
        )
        
        # Cleanup uploaded files (optional - comment out if you want to keep them)
        # for filepath in saved_files:
        #     os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': 'RAG database built successfully',
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat with RAG system
    
    Request JSON:
    {
        "query": "Your question here",
        "top_k": 5  (optional)
    }
    
    OR multipart/form-data with:
    - query: text
    - files: optional temporary files
    - top_k: optional
    """
    try:
        # Handle JSON request
        if request.is_json:
            data = request.get_json()
            query = data.get('query')
            top_k = data.get('top_k', 5)
            temp_files = None
        
        # Handle multipart/form-data (with temp files)
        else:
            query = request.form.get('query')
            top_k = int(request.form.get('top_k', 5))
            
            # Handle temporary files
            temp_files = None
            if 'files' in request.files:
                files = request.files.getlist('files')
                temp_files = []
                
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        unique_filename = f"temp_{timestamp}_{filename}"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                        file.save(filepath)
                        temp_files.append(filepath)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get chat response
        response = chat_with_rag(
            query=query,
            temp_files=temp_files,
            top_k=top_k
        )
        
        # Cleanup temp files
        if temp_files:
            for filepath in temp_files:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/upload-temp', methods=['POST'])
def upload_temp():
    """
    Upload temporary files to session memory (without querying)
    
    Request: multipart/form-data with 'files' field
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Save files
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"temp_{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                saved_files.append(filepath)
        
        # Process files into session memory
        pipeline = get_chat_pipeline()
        result = pipeline.process_temp_files(saved_files)
        
        # Cleanup
        for filepath in saved_files:
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Files uploaded to session memory',
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get database and session statistics"""
    try:
        store = get_vector_store()
        pipeline = get_chat_pipeline()
        
        # Get database stats
        db_count = store.count()
        all_metadata = store.get_all_metadata()
        
        # Group by file
        files_dict = {}
        for meta in all_metadata:
            file_name = meta.get('file_name', 'unknown')
            if file_name not in files_dict:
                files_dict[file_name] = 0
            files_dict[file_name] += 1
        
        return jsonify({
            'success': True,
            'database': {
                'total_chunks': db_count,
                'total_files': len(files_dict),
                'files': files_dict
            },
            'session': {
                'total_chunks': pipeline.session_memory.count()
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/delete-db', methods=['DELETE'])
def delete_db():
    """Clear entire permanent database"""
    try:
        builder = get_rag_builder()
        builder.clear_database()
        
        return jsonify({
            'success': True,
            'message': 'Database cleared successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/delete-session', methods=['DELETE'])
def delete_session():
    """Clear session memory"""
    try:
        pipeline = get_chat_pipeline()
        pipeline.clear_session()
        
        return jsonify({
            'success': True,
            'message': 'Session memory cleared successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 100MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error'
    }), 500


# ==========================================
# RUN APP
# ==========================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Multi-Modal RAG API Server")
    print("="*70)
    print(f"📍 Server: http://localhost:5000")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📊 Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)