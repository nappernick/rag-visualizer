"""
Hierarchical chunker that preserves document structure
"""
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .base import BaseChunker


@dataclass
class HierarchicalChunk:
    """Represents a chunk in a hierarchical structure"""
    id: str
    content: str
    level: int  # 0 = root, 1 = section, 2 = subsection, etc.
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    chunk_index: int = 0
    tokens: int = 0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}


class HierarchicalChunker(BaseChunker):
    """
    Chunker that creates hierarchical document structure based on:
    - Markdown headers
    - Document sections
    - Semantic boundaries
    """
    
    def __init__(self, 
                 chunk_size: int = 400, 
                 chunk_overlap: int = 80,
                 min_chunk_size: int = 100,
                 max_header_level: int = 4):
        """Initialize hierarchical chunker"""
        super().__init__(max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_header_level = max_header_level
    
    def chunk_document(self, text: str, **kwargs) -> List[str]:
        """Implementation of abstract method from BaseChunker"""
        # For compatibility, convert hierarchical chunks to simple text chunks
        document_id = kwargs.get('document_id', 'doc')
        hierarchical_chunks = self.chunk_hierarchical(text, document_id)
        return [chunk.content for chunk in hierarchical_chunks]
    
    def chunk_hierarchical(self, text: str, document_id: str) -> List[HierarchicalChunk]:
        """
        Create hierarchical chunks from text
        Returns list of HierarchicalChunk objects with parent-child relationships
        """
        if not text:
            return []
        
        # Detect document structure
        structure = self._detect_structure(text)
        
        if structure:
            # Create hierarchical chunks based on structure
            chunks = self._create_structured_chunks(structure, document_id)
        else:
            # Fallback to order-based hierarchy
            chunks = self._create_order_based_hierarchy(text, document_id)
        
        return chunks
    
    def _detect_structure(self, text: str) -> Optional[List[Dict]]:
        """
        Detect document structure from markdown headers, numbered sections, etc.
        Returns list of sections with their content and hierarchy level
        """
        sections = []
        
        # Try markdown headers first
        markdown_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            match = re.match(markdown_pattern, line, re.MULTILINE)
            if match:
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    if current_section['content']:  # Only add non-empty sections
                        sections.append(current_section)
                
                # Start new section
                level = len(match.group(1))  # Number of # symbols
                title = match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'line_start': len(sections)
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            if current_section['content']:
                sections.append(current_section)
        
        # If no markdown headers, try numbered sections (1., 1.1, 1.1.1, etc.)
        if not sections:
            sections = self._detect_numbered_sections(text)
        
        # If still no structure, try detecting paragraphs as sections
        if not sections:
            sections = self._detect_paragraph_structure(text)
        
        return sections if sections else None
    
    def _detect_numbered_sections(self, text: str) -> List[Dict]:
        """Detect numbered sections like 1., 1.1, 2.3.1, etc."""
        sections = []
        
        # Pattern for numbered sections
        pattern = r'^(\d+(?:\.\d+)*)\.\s+(.+)$'
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    if current_section['content']:
                        sections.append(current_section)
                
                # Calculate level based on number of dots
                number = match.group(1)
                level = len(number.split('.'))
                title = match.group(2).strip()
                
                current_section = {
                    'level': level,
                    'title': f"{number}. {title}",
                    'content': '',
                    'number': number
                }
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add last section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            if current_section['content']:
                sections.append(current_section)
        
        return sections
    
    def _detect_paragraph_structure(self, text: str) -> List[Dict]:
        """Detect paragraph-based structure for unstructured text"""
        sections = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para) > self.min_chunk_size:
                # Try to extract a title from first sentence
                first_sentence = para.split('.')[0] if '.' in para else para[:50]
                sections.append({
                    'level': 2,  # All paragraphs at same level
                    'title': first_sentence[:50] + ('...' if len(first_sentence) > 50 else ''),
                    'content': para,
                    'index': i
                })
        
        return sections
    
    def _create_structured_chunks(self, structure: List[Dict], document_id: str) -> List[HierarchicalChunk]:
        """Create hierarchical chunks from detected structure"""
        chunks = []
        chunk_index = 0
        
        # Create parent-child relationships
        parent_stack = []  # Stack of (level, chunk_id) tuples
        
        for section in structure:
            level = section['level']
            
            # Pop parents that are at same or lower level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()
            
            # Determine parent
            parent_id = parent_stack[-1][1] if parent_stack else None
            
            # Create chunk ID
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            
            # Split section content if too large
            section_chunks = self._split_section_content(
                section['content'], 
                section['title'],
                chunk_id,
                level
            )
            
            # Create main section chunk (header/summary)
            main_chunk = HierarchicalChunk(
                id=chunk_id,
                content=f"# {section['title']}\n\n{section_chunks[0] if section_chunks else section['content'][:200]}",
                level=level,
                parent_id=parent_id,
                children_ids=[],
                chunk_index=chunk_index,
                tokens=self.count_tokens(section['title'] + (section_chunks[0] if section_chunks else section['content'][:200])),
                metadata={
                    'title': section['title'],
                    'type': 'section_header',
                    'section_number': section.get('number', ''),
                    'document_id': document_id
                }
            )
            chunks.append(main_chunk)
            chunk_index += 1
            
            # Update parent's children
            if parent_id:
                for chunk in chunks:
                    if chunk.id == parent_id:
                        chunk.children_ids.append(chunk_id)
                        break
            
            # Add to parent stack
            parent_stack.append((level, chunk_id))
            
            # Create child chunks for remaining content
            for i, content_piece in enumerate(section_chunks[1:], 1):
                child_id = f"{chunk_id}_part_{i}"
                child_chunk = HierarchicalChunk(
                    id=child_id,
                    content=content_piece,
                    level=level + 1,  # Sub-chunks are one level deeper
                    parent_id=chunk_id,
                    chunk_index=chunk_index,
                    tokens=self.count_tokens(content_piece),
                    metadata={
                        'title': f"{section['title']} - Part {i}",
                        'type': 'section_content',
                        'part_number': i,
                        'document_id': document_id
                    }
                )
                chunks.append(child_chunk)
                main_chunk.children_ids.append(child_id)
                chunk_index += 1
        
        return chunks
    
    def _split_section_content(self, content: str, title: str, chunk_id: str, level: int) -> List[str]:
        """Split section content into smaller chunks if needed"""
        if not content:
            return []
        
        tokens = self.count_tokens(content)
        if tokens <= self.chunk_size:
            return [content]
        
        # Use semantic chunking for content splitting
        sentences = self._split_into_sentences_simple(content)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Add overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences
                current_tokens = sum(self.count_tokens(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences_simple(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_order_based_hierarchy(self, text: str, document_id: str) -> List[HierarchicalChunk]:
        """Create hierarchy based on chunk order when no structure is detected"""
        chunks = []
        chunk_index = 0
        
        # First, create regular semantic chunks
        text_chunks = self.chunk(text)
        
        # Group every 3-5 chunks under a parent
        group_size = 4
        
        for group_idx in range(0, len(text_chunks), group_size):
            group_chunks = text_chunks[group_idx:group_idx + group_size]
            
            # Create parent chunk (summary of group)
            parent_id = f"{document_id}_group_{group_idx // group_size}"
            parent_content = " ".join([c[:100] for c in group_chunks])[:300]  # Brief summary
            
            parent_chunk = HierarchicalChunk(
                id=parent_id,
                content=f"Section {group_idx // group_size + 1}: {parent_content}...",
                level=1,
                parent_id=None,
                children_ids=[],
                chunk_index=chunk_index,
                tokens=self.count_tokens(parent_content),
                metadata={
                    'type': 'group_parent',
                    'group_index': group_idx // group_size,
                    'document_id': document_id
                }
            )
            chunks.append(parent_chunk)
            chunk_index += 1
            
            # Create child chunks
            for i, chunk_content in enumerate(group_chunks):
                child_id = f"{document_id}_chunk_{chunk_index}"
                child_chunk = HierarchicalChunk(
                    id=child_id,
                    content=chunk_content,
                    level=2,
                    parent_id=parent_id,
                    chunk_index=chunk_index,
                    tokens=self.count_tokens(chunk_content),
                    metadata={
                        'type': 'content_chunk',
                        'position': group_idx + i,
                        'document_id': document_id
                    }
                )
                chunks.append(child_chunk)
                parent_chunk.children_ids.append(child_id)
                chunk_index += 1
        
        return chunks
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[str]:
        """
        Standard chunk method for compatibility
        Returns flat list of chunk contents
        """
        # For backward compatibility, return flat chunks
        if not text:
            return []
        
        # Use parent class semantic chunking
        return super().chunk(text, metadata)