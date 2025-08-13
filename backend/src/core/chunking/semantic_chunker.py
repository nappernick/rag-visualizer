"""
Semantic chunking implementation
"""
from typing import List, Optional
import re


class SemanticChunker:
    """Semantic text chunker that attempts to preserve semantic boundaries."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into semantically meaningful pieces.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Simple implementation - split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Add overlap from the end of the previous chunk
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Paragraph is too long, need to split it
                    sentences = self._split_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk)
                                if self.chunk_overlap > 0:
                                    overlap_text = current_chunk[-self.chunk_overlap:]
                                    current_chunk = overlap_text + " " + sentence
                                else:
                                    current_chunk = sentence
                            else:
                                # Single sentence is too long, split by chunk size
                                for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                                    chunks.append(sentence[i:i + self.chunk_size])
                                current_chunk = ""
                        else:
                            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            else:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        # Add any remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]