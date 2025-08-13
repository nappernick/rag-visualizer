"""
Hierarchical chunker that preserves document structure
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re
import uuid
from collections import defaultdict

from .base import BaseChunker
from ...models.schemas import Document, Chunk, ChunkType


@dataclass
class HierarchicalNode:
    """Represents a node in document hierarchy"""
    id: str
    level: int  # 0=document, 1=section, 2=subsection, etc.
    title: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalChunker(BaseChunker):
    """
    Creates multi-level chunks preserving document structure.
    Features:
    - Document, section, and subsection chunks
    - Parent-child relationships
    - Summary chunks for navigation
    - Preserves tables and code blocks
    """
    
    def __init__(self, max_chunk_size: int = 800, chunk_overlap: int = 100,
                 max_hierarchy_depth: int = 4, create_summaries: bool = True):
        super().__init__(max_chunk_size, chunk_overlap)
        self.max_hierarchy_depth = max_hierarchy_depth
        self.create_summaries = create_summaries
        self.min_section_size = 100  # Min tokens for section summary
        
        # Track hierarchy
        self.hierarchy_tree = {}
        self.chunk_hierarchy = defaultdict(list)
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Create hierarchical chunks from document"""
        chunks = []
        
        # Reset state
        self.hierarchy_tree = {}
        self.chunk_hierarchy = defaultdict(list)
        
        # Parse document structure
        sections = self._parse_document_structure(document.content)
        
        # Build hierarchy tree
        root_node = self._build_hierarchy(document, sections)
        
        # Create document summary chunk
        if self.create_summaries:
            doc_summary = self._create_summary_chunk(
                node=root_node,
                document=document,
                index=0,
                chunk_type=ChunkType.SUMMARY
            )
            chunks.append(doc_summary)
        
        # Process sections recursively
        chunk_index = len(chunks)
        for section in sections:
            section_chunks = self._process_section(
                section=section,
                document=document,
                parent_node=root_node,
                chunk_index=chunk_index,
                depth=1
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # Add hierarchical relationships
        self._add_relationships(chunks)
        
        return chunks
    
    def _parse_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse document into sections based on markdown headers"""
        sections = []
        
        # Split by headers (# ## ### etc)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'subsections': []
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        # If no sections found, treat entire content as one section
        if not sections:
            sections = [{
                'level': 1,
                'title': 'Main Content',
                'content': content,
                'subsections': []
            }]
        
        # Organize into hierarchy
        return self._organize_sections(sections)
    
    def _organize_sections(self, sections: List[Dict]) -> List[Dict]:
        """Organize flat sections into hierarchical structure"""
        if not sections:
            return []
        
        organized = []
        stack = []
        
        for section in sections:
            level = section['level']
            
            # Find parent
            while stack and stack[-1]['level'] >= level:
                stack.pop()
            
            if stack:
                # Add as subsection to parent
                stack[-1]['subsections'].append(section)
            else:
                # Top-level section
                organized.append(section)
            
            stack.append(section)
        
        return organized
    
    def _build_hierarchy(self, document: Document, sections: List[Dict]) -> HierarchicalNode:
        """Build hierarchy tree from sections"""
        root_id = str(uuid.uuid4())
        root_node = HierarchicalNode(
            id=root_id,
            level=0,
            title=document.title,
            content=self._extract_summary(document.content, 200),
            metadata={'doc_type': document.doc_type}
        )
        
        self.hierarchy_tree[root_id] = root_node
        
        # Process sections
        for section in sections:
            self._build_section_hierarchy(section, root_node, 1)
        
        return root_node
    
    def _build_section_hierarchy(self, section: Dict, parent: HierarchicalNode, level: int):
        """Recursively build hierarchy for sections"""
        if level > self.max_hierarchy_depth:
            return
        
        section_id = str(uuid.uuid4())
        section_node = HierarchicalNode(
            id=section_id,
            level=level,
            title=section['title'],
            content=section['content'],
            parent_id=parent.id,
            metadata={'has_subsections': len(section.get('subsections', [])) > 0}
        )
        
        parent.children_ids.append(section_id)
        self.hierarchy_tree[section_id] = section_node
        
        # Process subsections
        for subsection in section.get('subsections', []):
            self._build_section_hierarchy(subsection, section_node, level + 1)
    
    def _process_section(self, section: Dict, document: Document, 
                        parent_node: HierarchicalNode, chunk_index: int, 
                        depth: int) -> List[Chunk]:
        """Process a section into chunks"""
        chunks = []
        
        # Find section node
        section_node = None
        for node in self.hierarchy_tree.values():
            if node.title == section['title'] and node.parent_id == parent_node.id:
                section_node = node
                break
        
        if not section_node:
            return chunks
        
        # Create section summary if large enough
        section_tokens = self.count_tokens(section['content'])
        if self.create_summaries and section_tokens > self.min_section_size:
            summary_chunk = self._create_summary_chunk(
                node=section_node,
                document=document,
                index=chunk_index + len(chunks),
                chunk_type=ChunkType.SECTION
            )
            chunks.append(summary_chunk)
        
        # Extract special content (tables, code blocks)
        special_chunks = self._extract_special_content(
            content=section['content'],
            document=document,
            parent_node=section_node,
            start_index=chunk_index + len(chunks)
        )
        chunks.extend(special_chunks)
        
        # Create content chunks
        if section['content']:
            content_chunks = self._create_content_chunks(
                content=section['content'],
                document=document,
                parent_node=section_node,
                start_index=chunk_index + len(chunks)
            )
            chunks.extend(content_chunks)
        
        # Process subsections
        if depth < self.max_hierarchy_depth:
            for subsection in section.get('subsections', []):
                subsection_chunks = self._process_section(
                    section=subsection,
                    document=document,
                    parent_node=section_node,
                    chunk_index=chunk_index + len(chunks),
                    depth=depth + 1
                )
                chunks.extend(subsection_chunks)
        
        return chunks
    
    def _create_summary_chunk(self, node: HierarchicalNode, document: Document,
                             index: int, chunk_type: ChunkType) -> Chunk:
        """Create a summary chunk for a node"""
        if chunk_type == ChunkType.SUMMARY:
            summary = f"# {node.title}\n\n{node.content}"
        else:
            summary = f"## {node.title}\n\n{self._extract_summary(node.content, 300)}"
        
        chunk = Chunk(
            content=summary,
            document_id=document.id,
            chunk_index=index,
            chunk_type=chunk_type,
            tokens=self.count_tokens(summary),
            parent_id=node.parent_id,
            metadata={
                'document_title': document.title,
                'hierarchy_level': node.level,
                'node_id': node.id,
                'is_summary': True,
                'section_title': node.title
            }
        )
        
        self.chunk_hierarchy[node.id].append(chunk.id)
        return chunk
    
    def _extract_special_content(self, content: str, document: Document,
                                parent_node: HierarchicalNode, start_index: int) -> List[Chunk]:
        """Extract tables and code blocks as separate chunks"""
        chunks = []
        
        # Extract code blocks
        code_pattern = r'```(?:(\w+)\n)?(.*?)```'
        for i, match in enumerate(re.finditer(code_pattern, content, re.DOTALL)):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            
            chunk = Chunk(
                content=f"```{language}\n{code}\n```",
                document_id=document.id,
                chunk_index=start_index + len(chunks),
                chunk_type=ChunkType.CODE,
                tokens=self.count_tokens(code),
                parent_id=parent_node.id,
                metadata={
                    'document_title': document.title,
                    'language': language,
                    'is_code': True,
                    'node_id': parent_node.id
                }
            )
            chunks.append(chunk)
        
        # Extract tables (simplified - looks for pipe-delimited tables)
        table_pattern = r'(\|.*\|[\s\S]*?\n)(?:\|.*\|.*\n)+'
        for i, match in enumerate(re.finditer(table_pattern, content)):
            table = match.group(0).strip()
            
            chunk = Chunk(
                content=table,
                document_id=document.id,
                chunk_index=start_index + len(chunks),
                chunk_type=ChunkType.TABLE,
                tokens=self.count_tokens(table),
                parent_id=parent_node.id,
                metadata={
                    'document_title': document.title,
                    'is_table': True,
                    'node_id': parent_node.id
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_content_chunks(self, content: str, document: Document,
                              parent_node: HierarchicalNode, start_index: int) -> List[Chunk]:
        """Create content chunks from text"""
        # Remove special content
        cleaned = self._remove_special_content(content)
        
        if not cleaned.strip():
            return []
        
        # Use standard chunking for content
        chunks = []
        paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunk = Chunk(
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=start_index + len(chunks),
                    chunk_type=ChunkType.HIERARCHICAL,
                    tokens=self.count_tokens(chunk_content),
                    parent_id=parent_node.id,
                    metadata={
                        'document_title': document.title,
                        'hierarchy_level': parent_node.level,
                        'node_id': parent_node.id
                    }
                )
                chunks.append(chunk)
                self.chunk_hierarchy[parent_node.id].append(chunk.id)
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = ' '.join(chunk_content.split()[-self.chunk_overlap:])
                    current_chunk = [overlap_text, para]
                    current_tokens = self.count_tokens(' '.join(current_chunk))
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                document_id=document.id,
                chunk_index=start_index + len(chunks),
                chunk_type=ChunkType.HIERARCHICAL,
                tokens=self.count_tokens(chunk_content),
                parent_id=parent_node.id,
                metadata={
                    'document_title': document.title,
                    'hierarchy_level': parent_node.level,
                    'node_id': parent_node.id
                }
            )
            chunks.append(chunk)
            self.chunk_hierarchy[parent_node.id].append(chunk.id)
        
        return chunks
    
    def _remove_special_content(self, content: str) -> str:
        """Remove code blocks and tables from content"""
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        # Remove tables
        content = re.sub(r'(\|.*\|[\s\S]*?\n)(?:\|.*\|.*\n)+', '', content)
        return content.strip()
    
    def _extract_summary(self, content: str, max_length: int) -> str:
        """Extract first part of content as summary"""
        # Take first paragraph or max_length characters
        paragraphs = content.split('\n\n')
        if paragraphs:
            summary = paragraphs[0]
            if len(summary) > max_length:
                summary = summary[:max_length] + '...'
            return summary
        return content[:max_length] + '...' if len(content) > max_length else content
    
    def _add_relationships(self, chunks: List[Chunk]):
        """Add parent-child relationships to chunks"""
        # Create chunk ID mapping
        chunk_map = {chunk.id: chunk for chunk in chunks}
        
        # Add relationships from hierarchy
        for node_id, chunk_ids in self.chunk_hierarchy.items():
            node = self.hierarchy_tree.get(node_id)
            if not node:
                continue
            
            # Add children relationships
            for chunk_id in chunk_ids:
                if chunk_id in chunk_map:
                    chunk = chunk_map[chunk_id]
                    
                    # Add siblings
                    sibling_ids = [cid for cid in chunk_ids if cid != chunk_id]
                    if sibling_ids:
                        chunk.metadata['sibling_chunks'] = sibling_ids
                    
                    # Add children from child nodes
                    child_chunk_ids = []
                    for child_node_id in node.children_ids:
                        if child_node_id in self.chunk_hierarchy:
                            child_chunk_ids.extend(self.chunk_hierarchy[child_node_id])
                    
                    if child_chunk_ids:
                        chunk.children_ids = child_chunk_ids