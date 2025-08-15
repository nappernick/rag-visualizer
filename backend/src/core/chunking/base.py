"""
Base chunker interface and standard implementation
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
# Try to import tiktoken for better token counting, fallback to simple word-based estimation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
import re
from ...models.schemas import Document, Chunk, ChunkType


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies"""
    
    def __init__(self, max_chunk_size: int = 400, chunk_overlap: int = 80):
        """Initialize chunker with industry-standard defaults.
        
        Args:
            max_chunk_size: Maximum tokens per chunk (default 400, industry standard)
            chunk_overlap: Token overlap between chunks (default 80, which is 20%)
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                print(f"✅ Tiktoken initialized successfully")
            except Exception as e:
                print(f"⚠️ Tiktoken initialization failed: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Process document into chunks"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to simple word-based estimation (roughly 4 characters per token)
            return len(text) // 4


class StandardChunker(BaseChunker):
    """Standard fixed-size chunking with overlap"""
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Create fixed-size chunks with overlap"""
        chunks = []
        text = document.content
        
        # Split into sentences for better boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding sentence exceeds limit, create chunk
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append(self._create_chunk(
                    content=chunk_content,
                    document=document,
                    index=chunk_index,
                    chunk_type=ChunkType.STANDARD
                ))
                chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    # Keep last sentences for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        sent_tokens = self.count_tokens(sent)
                        if overlap_tokens + sent_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(self._create_chunk(
                content=chunk_content,
                document=document,
                index=chunk_index,
                chunk_type=ChunkType.STANDARD
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, content: str, document: Document, 
                     index: int, chunk_type: ChunkType) -> Chunk:
        """Create a chunk object"""
        return Chunk(
            content=content,
            document_id=document.id,
            chunk_index=index,
            chunk_type=chunk_type,
            tokens=self.count_tokens(content),
            metadata={
                'document_title': document.title,
                'chunking_strategy': 'standard'
            }
        )