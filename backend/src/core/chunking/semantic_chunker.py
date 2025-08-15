"""
Semantic chunker implementation with proper overlap
"""
import re
from typing import List, Optional
from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that preserves paragraph and sentence boundaries
    with proper sliding window overlap implementation.
    """
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        """Initialize semantic chunker with overlap"""
        super().__init__(max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[str]:
        """
        Chunk text with proper sliding window overlap.
        Uses a more efficient approach that maintains context.
        """
        if not text:
            return []
        
        # Split into sentences first for better boundary preservation
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence is too large, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split the large sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + " ")
                    if temp_tokens + word_tokens > self.chunk_size and temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                        # Create overlap from the end of this chunk
                        if self.chunk_overlap > 0:
                            overlap_words = []
                            overlap_tokens = 0
                            for w in reversed(temp_chunk):
                                w_tokens = self.count_tokens(w + " ")
                                if overlap_tokens + w_tokens <= self.chunk_overlap:
                                    overlap_words.insert(0, w)
                                    overlap_tokens += w_tokens
                                else:
                                    break
                            temp_chunk = overlap_words
                            temp_tokens = overlap_tokens
                        else:
                            temp_chunk = []
                            temp_tokens = 0
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                    current_chunk = []
                    current_tokens = 0
                i += 1
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Create overlap for next chunk
                if self.chunk_overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_chunk = []
                    overlap_tokens = 0
                    
                    # Work backwards through current chunk to build overlap
                    for j in range(len(current_chunk) - 1, -1, -1):
                        sent_tokens = self.count_tokens(current_chunk[j])
                        if overlap_tokens + sent_tokens <= self.chunk_overlap:
                            overlap_chunk.insert(0, current_chunk[j])
                            overlap_tokens += sent_tokens
                        else:
                            # Try to include partial sentence if there's room
                            remaining_tokens = self.chunk_overlap - overlap_tokens
                            if remaining_tokens > 20:  # Only include if meaningful
                                words = current_chunk[j].split()
                                partial = []
                                partial_tokens = 0
                                for word in reversed(words):
                                    w_tokens = self.count_tokens(word + " ")
                                    if partial_tokens + w_tokens <= remaining_tokens:
                                        partial.insert(0, word)
                                        partial_tokens += w_tokens
                                    else:
                                        break
                                if partial:
                                    overlap_chunk.insert(0, " ".join(partial))
                                    overlap_tokens += partial_tokens
                            break
                    
                    current_chunk = overlap_chunk
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Post-process chunks to ensure quality
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and len(chunk) > 50:  # Minimum chunk size
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving important boundaries.
        Handles edge cases like URLs, abbreviations, decimals, etc.
        """
        # Protect special patterns from splitting
        # Protect URLs
        text = re.sub(r'(https?://[^\s]+)', r'<<<URL:\1>>>', text)
        # Protect email addresses
        text = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'<<<EMAIL:\1>>>', text)
        # Protect decimals and version numbers
        text = re.sub(r'(\d+)\.(\d+)', r'\1<<<DECIMAL>>>\2', text)
        # Protect common abbreviations
        abbrevs = ['Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr', 'Ph.D', 'M.D', 'B.A', 'M.A', 'B.S', 'M.S',
                   'i.e', 'e.g', 'etc', 'vs', 'Inc', 'Ltd', 'Co', 'Corp']
        for abbrev in abbrevs:
            text = re.sub(rf'\b{re.escape(abbrev)}\.\s*', f'{abbrev}<<<DOT>>> ', text)
        
        # Split on sentence boundaries
        # Look for . ! ? followed by space and capital letter, or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$', text)
        
        # Also split on newlines that indicate new paragraphs
        expanded_sentences = []
        for sent in sentences:
            if '\n\n' in sent:
                parts = sent.split('\n\n')
                for i, part in enumerate(parts):
                    if part.strip():
                        if i > 0:
                            # Add paragraph break as a boundary marker
                            expanded_sentences.append('<<<PARAGRAPH>>>')
                        expanded_sentences.append(part.strip())
            elif sent.strip():
                expanded_sentences.append(sent.strip())
        
        # Restore protected patterns and clean up
        restored = []
        for sent in expanded_sentences:
            if sent == '<<<PARAGRAPH>>>':
                # Handle paragraph boundaries
                if restored and not restored[-1].endswith(('.', '!', '?')):
                    restored[-1] += '.'
                continue
            
            # Restore protected patterns
            sent = sent.replace('<<<DOT>>>', '.')
            sent = sent.replace('<<<DECIMAL>>>', '.')
            sent = re.sub(r'<<<URL:(.*?)>>>', r'\1', sent)
            sent = re.sub(r'<<<EMAIL:(.*?)>>>', r'\1', sent)
            
            # Clean up whitespace
            sent = ' '.join(sent.split())
            
            if sent and len(sent) > 1:
                # Ensure sentence ends with punctuation
                if not sent[-1] in '.!?:;)"\']}':
                    # Check if it's a heading or title (short and no ending punctuation)
                    if len(sent) < 100 and not any(p in sent for p in '.!?'):
                        sent = sent  # Keep as is (likely a heading)
                    else:
                        sent += '.'  # Add period to incomplete sentence
                restored.append(sent)
        
        return restored
    
    def chunk_document(self, document: dict) -> List[dict]:
        """
        Chunk a document - implements abstract method from BaseChunker
        """
        text = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Get text chunks
        text_chunks = self.chunk(text, metadata)
        
        # Convert to document chunks format
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'content': chunk_text,
                'chunk_index': i,
                'chunk_type': 'semantic',
                'tokens': self.count_tokens(chunk_text),
                'metadata': metadata.copy()
            }
            chunks.append(chunk)
        
        return chunks