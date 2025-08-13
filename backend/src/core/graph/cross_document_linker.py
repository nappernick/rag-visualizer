"""
Cross-document entity linking for knowledge graph integration
"""
import logging
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from ...models.schemas import Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    entity1: Entity
    entity2: Entity
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'semantic'


class CrossDocumentLinker:
    """Links entities across different documents to create a unified knowledge graph"""
    
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize the cross-document linker
        
        Args:
            similarity_threshold: Minimum similarity score for fuzzy matching (0-1)
        """
        self.similarity_threshold = similarity_threshold
        
    def link_document_graphs(self, 
                           all_entities: Dict[str, List[Entity]], 
                           all_relationships: Dict[str, List[Relationship]]) -> Tuple[List[Relationship], List[EntityMatch]]:
        """
        Create cross-document relationships between entities
        
        Args:
            all_entities: Dictionary of document_id -> list of entities
            all_relationships: Dictionary of document_id -> list of relationships
            
        Returns:
            Tuple of (new cross-document relationships, entity matches found)
        """
        # Find matching entities across documents
        entity_matches = self._find_entity_matches(all_entities)
        
        # Create cross-document relationships
        cross_doc_relationships = self._create_cross_document_relationships(entity_matches)
        
        # Connect isolated document graphs through common concepts
        bridge_relationships = self._create_bridge_relationships(all_entities, entity_matches)
        
        all_new_relationships = cross_doc_relationships + bridge_relationships
        
        logger.info(f"Created {len(all_new_relationships)} cross-document relationships from {len(entity_matches)} entity matches")
        
        return all_new_relationships, entity_matches
    
    def _find_entity_matches(self, all_entities: Dict[str, List[Entity]]) -> List[EntityMatch]:
        """Find matching entities across different documents"""
        matches = []
        processed_pairs = set()
        
        doc_ids = list(all_entities.keys())
        
        for i, doc1_id in enumerate(doc_ids):
            for j, doc2_id in enumerate(doc_ids[i+1:], i+1):
                for entity1 in all_entities[doc1_id]:
                    for entity2 in all_entities[doc2_id]:
                        # Skip if already processed
                        pair_key = tuple(sorted([entity1.id, entity2.id]))
                        if pair_key in processed_pairs:
                            continue
                        processed_pairs.add(pair_key)
                        
                        # Check for matches
                        match = self._match_entities(entity1, entity2)
                        if match:
                            matches.append(match)
        
        return matches
    
    def _match_entities(self, entity1: Entity, entity2: Entity) -> EntityMatch:
        """Check if two entities match based on various criteria"""
        
        # Normalize names for comparison
        name1 = self._normalize_name(entity1.name)
        name2 = self._normalize_name(entity2.name)
        
        # 1. Exact match
        if name1 == name2:
            return EntityMatch(entity1, entity2, 1.0, 'exact')
        
        # 2. Fuzzy match for similar names
        similarity = SequenceMatcher(None, name1, name2).ratio()
        if similarity >= self.similarity_threshold:
            return EntityMatch(entity1, entity2, similarity, 'fuzzy')
        
        # 3. Check for common acronyms/abbreviations
        if self._are_acronyms_related(entity1.name, entity2.name):
            return EntityMatch(entity1, entity2, 0.9, 'acronym')
        
        # 4. Check for technology versions (e.g., "GPT-3" and "GPT-4")
        if self._are_versions_related(entity1.name, entity2.name):
            return EntityMatch(entity1, entity2, 0.8, 'version')
        
        # 5. Check for same entity type with high name similarity
        if entity1.entity_type == entity2.entity_type and similarity >= 0.7:
            return EntityMatch(entity1, entity2, similarity, 'type_match')
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison"""
        # Convert to lowercase
        normalized = name.lower()
        # Remove special characters except spaces and hyphens
        normalized = re.sub(r'[^a-z0-9\s\-]', '', normalized)
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _are_acronyms_related(self, name1: str, name2: str) -> bool:
        """Check if one name is an acronym of the other"""
        # Common acronym patterns in tech
        acronym_map = {
            'rag': 'retrieval augmented generation',
            'llm': 'large language model',
            'nlp': 'natural language processing',
            'api': 'application programming interface',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'bert': 'bidirectional encoder representations from transformers',
            'gpt': 'generative pre-trained transformer',
        }
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        for acronym, full in acronym_map.items():
            if (acronym in name1_lower and full in name2_lower) or \
               (acronym in name2_lower and full in name1_lower):
                return True
        
        return False
    
    def _are_versions_related(self, name1: str, name2: str) -> bool:
        """Check if names represent different versions of the same technology"""
        # Extract base name and version patterns
        version_pattern = r'(.+?)[\s\-]*(v?\d+(?:\.\d+)*|\d+)$'
        
        match1 = re.match(version_pattern, name1, re.IGNORECASE)
        match2 = re.match(version_pattern, name2, re.IGNORECASE)
        
        if match1 and match2:
            base1 = match1.group(1).lower().strip()
            base2 = match2.group(1).lower().strip()
            return base1 == base2
        
        return False
    
    def _create_cross_document_relationships(self, entity_matches: List[EntityMatch]) -> List[Relationship]:
        """Create relationships between matched entities across documents"""
        relationships = []
        
        for match in entity_matches:
            # Create bidirectional relationships based on match type
            if match.match_type == 'exact':
                rel_type = "SAME_AS"
            elif match.match_type == 'fuzzy':
                rel_type = "SIMILAR_TO"
            elif match.match_type == 'acronym':
                rel_type = "ABBREVIATION_OF"
            elif match.match_type == 'version':
                rel_type = "VERSION_OF"
            else:
                rel_type = "RELATED_TO"
            
            # Create cross-document relationship
            relationship = Relationship(
                source_entity_id=match.entity1.id,
                target_entity_id=match.entity2.id,
                relationship_type=rel_type,
                weight=match.similarity_score,
                document_ids=list(set(match.entity1.document_ids + match.entity2.document_ids)),
                metadata={
                    "cross_document": True,
                    "match_type": match.match_type,
                    "similarity_score": match.similarity_score
                }
            )
            relationships.append(relationship)
        
        return relationships
    
    def _create_bridge_relationships(self, 
                                   all_entities: Dict[str, List[Entity]], 
                                   entity_matches: List[EntityMatch]) -> List[Relationship]:
        """Create bridge relationships to connect isolated document graphs"""
        relationships = []
        
        # Find common high-level concepts across documents
        concept_entities = self._find_concept_entities(all_entities)
        
        if len(concept_entities) >= 2:
            # Connect high-level concepts from different documents
            for i, (doc1_id, entity1) in enumerate(concept_entities):
                for doc2_id, entity2 in concept_entities[i+1:]:
                    if doc1_id != doc2_id:
                        # Create a bridge relationship between concepts
                        relationship = Relationship(
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            relationship_type="DOMAIN_RELATED",
                            weight=0.7,
                            document_ids=[doc1_id, doc2_id],
                            metadata={
                                "cross_document": True,
                                "bridge_relationship": True,
                                "description": "High-level domain connection"
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _find_concept_entities(self, all_entities: Dict[str, List[Entity]]) -> List[Tuple[str, Entity]]:
        """Find high-level concept entities that can serve as bridges"""
        concept_keywords = [
            'rag', 'retrieval', 'generation', 'ai', 'ml', 'llm',
            'model', 'system', 'pipeline', 'framework', 'database',
            'vector', 'embedding', 'knowledge', 'graph', 'search'
        ]
        
        concept_entities = []
        
        for doc_id, entities in all_entities.items():
            for entity in entities:
                # Check if entity is a high-level concept
                name_lower = entity.name.lower()
                if any(keyword in name_lower for keyword in concept_keywords):
                    if entity.entity_type in ['concept', 'technology', 'framework', 'system']:
                        concept_entities.append((doc_id, entity))
                        break  # One concept per document
        
        return concept_entities