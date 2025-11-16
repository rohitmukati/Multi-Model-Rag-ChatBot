"""
Text Chunking for RAG
Clean production code for services/chunking.py
"""

from typing import List, Dict, Any
import re


class TextChunker:
    """Smart text chunking with overlap for better context"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = text.strip()
        
        # Split into sentences for better chunking
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If single sentence exceeds chunk_size, split it
            if sentence_size > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence
                sub_chunks = self._split_long_text(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding sentence exceeds chunk_size
            if current_size + sentence_size > self.chunk_size:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk)[-self.overlap:]
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_size = len(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Format with metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text.strip(),
                'chunk_index': i,
                'chunk_size': len(chunk_text)
            }
            
            # Add metadata if provided
            if metadata:
                chunk_dict['metadata'] = metadata
            
            result.append(chunk_dict)
        
        return result
    
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    
    def _split_long_text(self, text: str) -> List[str]:
        """Split text that exceeds chunk_size"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks
    
    
    def chunk_by_paragraphs(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by paragraphs (preserves structure better)
        
        Args:
            text: Input text
            metadata: Optional metadata
        
        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Format results
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text.strip(),
                'chunk_index': i,
                'chunk_size': len(chunk_text)
            }
            
            if metadata:
                chunk_dict['metadata'] = metadata
            
            result.append(chunk_dict)
        
        return result
    
    
    def chunk_document(
        self,
        text: str,
        file_name: str,
        file_type: str,
        extra_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document with automatic metadata
        
        Args:
            text: Document text
            file_name: Name of source file
            file_type: Type of file (pdf, docx, etc.)
            extra_metadata: Additional metadata
        
        Returns:
            List of chunks with metadata
        """
        base_metadata = {
            'file_name': file_name,
            'file_type': file_type
        }
        
        if extra_metadata:
            base_metadata.update(extra_metadata)
        
        # Use paragraph chunking for better structure
        chunks = self.chunk_by_paragraphs(text, base_metadata)
        
        return chunks


# Global chunker instance
_chunker = None


def get_chunker(chunk_size: int = 512, overlap: int = 50) -> TextChunker:
    """Get or create chunker instance"""
    global _chunker
    if _chunker is None:
        _chunker = TextChunker(chunk_size, overlap)
    return _chunker


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Quick function to chunk text (returns only text)
    
    Args:
        text: Input text
        chunk_size: Max characters per chunk
        overlap: Overlap characters
    
    Returns:
        List of text chunks
    """
    chunker = get_chunker(chunk_size, overlap)
    chunks = chunker.chunk_text(text)
    return [c['text'] for c in chunks]


def chunk_document(
    text: str,
    file_name: str,
    file_type: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk document with metadata
    
    Args:
        text: Document text
        file_name: Source file name
        file_type: File type
        chunk_size: Max characters per chunk
        overlap: Overlap characters
    
    Returns:
        List of chunks with metadata
    """
    chunker = get_chunker(chunk_size, overlap)
    return chunker.chunk_document(text, file_name, file_type)

if __name__ == "__main__":
    sample_text = (
        "This is a sample text. It is meant to demonstrate the chunking functionality. "
        "The text will be split into smaller chunks based on the specified chunk size and overlap. "
        "Each chunk will preserve context by overlapping with the previous chunk. "
        "This helps in maintaining coherence when processing text for RAG applications. "
        "The chunker can handle long sentences by splitting them appropriately. "
        "It also allows for metadata to be attached to each chunk for better traceability."
    )
    
    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text(sample_text)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} (size {chunk['chunk_size']}): {chunk['text']}\n")