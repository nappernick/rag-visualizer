"""
Entity extraction service using Claude AI
"""
import os
from typing import List, Dict, Optional, Tuple
import logging
import hashlib

# Import Claude extractor
try:
    from ..core.graph.claude_extractor import ClaudeGraphExtractor
    CLAUDE_AVAILABLE = True
except ImportError:
    ClaudeGraphExtractor = None
    CLAUDE_AVAILABLE = False

logger = logging.getLogger(__name__)


class EntityService:
    """Service for extracting entities from text using Claude AI"""
    
    def __init__(self):
        # Initialize Claude extractor
        if CLAUDE_AVAILABLE:
            try:
                self.claude_extractor = ClaudeGraphExtractor()
                self.claude_initialized = True
                logger.info("Claude graph extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude extractor: {e}")
                self.claude_extractor = None
                self.claude_initialized = False
        else:
            logger.error("Claude extractor not available - entity extraction will not work")
            self.claude_extractor = None
            self.claude_initialized = False
        
        self.initialized = self.claude_initialized
    
    async def extract_entities(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None,
        use_claude: Optional[bool] = None  # Keep for compatibility
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text using Claude AI
        Returns: (entities, relationships)"""
        
        if not self.claude_initialized or not self.claude_extractor:
            logger.error("Claude extractor not initialized")
            return [], []
        
        return await self._extract_with_claude(text, document_id, chunk_ids)
    
    async def _extract_with_claude(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships using Claude AI
        Returns: (entities, relationships)"""
        
        try:
            from ..models import Chunk
            
            # Create mock chunks for the Claude extractor
            mock_chunks = [Chunk(
                content=text,
                document_id=document_id,
                chunk_index=0,
                tokens=len(text) // 4  # Simple estimation
            )]
            
            # Extract entities and relationships using Claude
            entities, relationships = self.claude_extractor.extract_from_chunks(
                chunks=mock_chunks, 
                document_id=document_id
            )
            
            # Convert Entity objects to dictionaries
            entity_dicts = []
            for entity in entities:
                entity_dict = {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "document_ids": entity.document_ids,
                    "chunk_ids": chunk_ids or [],
                    "frequency": entity.frequency,
                    "metadata": entity.metadata
                }
                entity_dicts.append(entity_dict)
            
            # Convert Relationship objects to dictionaries
            relationship_dicts = []
            for rel in relationships:
                rel_dict = {
                    "id": rel.id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "relationship_type": rel.relationship_type,
                    "weight": rel.weight,
                    "document_ids": rel.document_ids,
                    "metadata": rel.metadata
                }
                relationship_dicts.append(rel_dict)
            
            logger.info(f"Extracted {len(entity_dicts)} entities and {len(relationship_dicts)} relationships using Claude")
            return entity_dicts, relationship_dicts
            
        except Exception as e:
            logger.error(f"Error extracting entities with Claude: {e}")
            # Return empty lists on error
            return [], []
    
    async def extract_relationships(
        self, 
        entities: List[Dict],
        text: str,
        document_id: str
    ) -> List[Dict]:
        """Extract relationships between entities - deprecated, Claude does this automatically"""
        # This method is kept for backward compatibility but Claude handles relationships internally
        logger.warning("extract_relationships called but Claude handles this automatically")
        return []


# Dependency injection for lazy initialization
def get_entity_service():
    """Get or create entity service instance"""
    if not hasattr(get_entity_service, "_instance"):
        get_entity_service._instance = EntityService()
    return get_entity_service._instance