"""
Entity extraction service using SpaCy and Claude
"""
import os
from typing import List, Dict, Optional, Tuple
import logging
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
from collections import Counter
import hashlib

# Try to import Claude extractor
try:
    from ..core.graph.claude_extractor import ClaudeGraphExtractor
    CLAUDE_AVAILABLE = True
except ImportError:
    ClaudeGraphExtractor = None
    CLAUDE_AVAILABLE = False

logger = logging.getLogger(__name__)


class EntityService:
    """Service for extracting entities from text using SpaCy and Claude"""
    
    def __init__(self):
        model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
        
        # Initialize SpaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                self.spacy_initialized = True
                logger.info(f"SpaCy model loaded: {model_name}")
            except OSError:
                logger.warning(f"SpaCy model '{model_name}' not found")
                self.nlp = None
                self.spacy_initialized = False
        else:
            logger.info("SpaCy not available, will use Claude for entity extraction")
            self.nlp = None
            self.spacy_initialized = False
        
        # Initialize Claude extractor if available (default to True)
        self.use_claude = os.getenv("USE_CLAUDE_EXTRACTION", "true").lower() == "true"
        if self.use_claude and CLAUDE_AVAILABLE:
            try:
                self.claude_extractor = ClaudeGraphExtractor()
                self.claude_initialized = True
                logger.info("Claude graph extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude extractor: {e}")
                self.claude_extractor = None
                self.claude_initialized = False
        else:
            self.claude_extractor = None
            self.claude_initialized = False
        
        self.initialized = self.spacy_initialized or self.claude_initialized
    
    async def extract_entities(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None,
        use_claude: Optional[bool] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text
        Returns: (entities, relationships)"""
        
        # Allow override of Claude usage
        should_use_claude = use_claude if use_claude is not None else (self.use_claude and self.claude_initialized)
        
        if should_use_claude and self.claude_extractor:
            return await self._extract_with_claude(text, document_id, chunk_ids)
        elif self.spacy_initialized and self.nlp:
            entities = await self._extract_with_spacy(text, document_id, chunk_ids)
            return entities, []  # SpaCy doesn't extract relationships
        else:
            entities = await self._extract_fallback(text, document_id, chunk_ids)
            return entities, []  # Fallback doesn't extract relationships
    
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
            # Fallback to SpaCy or simple extraction
            if self.spacy_initialized:
                entities = await self._extract_with_spacy(text, document_id, chunk_ids)
                return entities, []  # SpaCy doesn't extract relationships
            else:
                entities = await self._extract_fallback(text, document_id, chunk_ids)
                return entities, []  # Fallback doesn't extract relationships
    
    async def _extract_with_spacy(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Extract entities using SpaCy NLP"""
        
        try:
            doc = self.nlp(text)
            entity_counts = Counter()
            entities = []
            
            # Extract named entities
            for ent in doc.ents:
                if len(ent.text.strip()) < 2:  # Skip very short entities
                    continue
                    
                entity_text = ent.text.strip()
                entity_type = ent.label_
                
                # Skip common but not useful entities
                if entity_type in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    continue
                
                entity_counts[entity_text] += 1
                
                # Create entity ID based on text
                entity_id = hashlib.md5(f"{entity_text}_{entity_type}".encode()).hexdigest()[:12]
                
                entity = {
                    "id": entity_id,
                    "name": entity_text,
                    "entity_type": entity_type,
                    "document_ids": [document_id],
                    "chunk_ids": chunk_ids or [],
                    "frequency": 1,  # Will be updated later
                    "metadata": {
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": 1.0
                    }
                }
                
                entities.append(entity)
            
            # Update frequencies
            entity_freq_map = {}
            for entity in entities:
                key = entity["name"]
                if key in entity_freq_map:
                    entity_freq_map[key]["frequency"] += 1
                else:
                    entity_freq_map[key] = entity
            
            # Return unique entities with correct frequencies
            unique_entities = list(entity_freq_map.values())
            
            logger.info(f"Extracted {len(unique_entities)} unique entities using SpaCy")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities with SpaCy: {e}")
            return await self._extract_fallback(text, document_id, chunk_ids)
    
    async def _extract_fallback(
        self, 
        text: str, 
        document_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fallback entity extraction using simple patterns"""
        
        entities = []
        text_lower = text.lower()
        
        # Simple keyword-based extraction
        tech_keywords = {
            "redis": "Technology",
            "cache": "Concept",
            "memory": "Concept", 
            "database": "Technology",
            "graph": "Concept",
            "vector": "Concept",
            "embedding": "Concept",
            "search": "Concept",
            "query": "Concept",
            "document": "Concept",
            "chunk": "Concept",
            "entity": "Concept",
            "relationship": "Concept",
            "python": "Technology",
            "fastapi": "Technology",
            "supabase": "Technology",
            "qdrant": "Technology",
            "neo4j": "Technology"
        }
        
        for keyword, entity_type in tech_keywords.items():
            if keyword in text_lower:
                entity_id = hashlib.md5(f"{keyword}_{entity_type}".encode()).hexdigest()[:12]
                
                entity = {
                    "id": entity_id,
                    "name": keyword.title(),
                    "entity_type": entity_type,
                    "document_ids": [document_id],
                    "chunk_ids": chunk_ids or [],
                    "frequency": text_lower.count(keyword),
                    "metadata": {
                        "extraction_method": "fallback",
                        "confidence": 0.7
                    }
                }
                
                entities.append(entity)
        
        logger.info(f"Extracted {len(entities)} entities using fallback method")
        return entities
    
    async def extract_relationships(
        self, 
        entities: List[Dict],
        text: str,
        document_id: str
    ) -> List[Dict]:
        """Extract relationships between entities"""
        
        # When using Claude, relationships are extracted by Claude itself
        # This method is only called as a fallback when Claude doesn't return relationships
        # or when explicitly using SpaCy/fallback methods
        
        # Return empty list - relationships should come from Claude extraction
        # We don't want automatic relationship creation between all entity pairs
        logger.info("Relationship extraction delegated to Claude AI")
        return []
    
    def _determine_relationship_type(self, type1: str, type2: str) -> str:
        """Determine relationship type based on entity types"""
        
        if type1 == type2:
            return "similar_to"
        
        if (type1 == "Technology" and type2 == "Concept") or (type1 == "Concept" and type2 == "Technology"):
            return "implements"
        
        if type1 == "PERSON" and type2 == "ORG":
            return "works_for"
        
        if type1 == "ORG" and type2 == "GPE":
            return "located_in"
        
        return "related_to"


# Dependency injection for lazy initialization
def get_entity_service():
    """Get or create entity service instance"""
    if not hasattr(get_entity_service, "_instance"):
        get_entity_service._instance = EntityService()
    return get_entity_service._instance