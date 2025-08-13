"""
Entity and relationship extraction for knowledge graph construction
"""
from typing import List, Dict, Tuple, Optional, Set
import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from models.schemas import Chunk, Entity, Relationship

logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    logger.warning("spaCy not installed. Using pattern-based extraction only.")
    SPACY_AVAILABLE = False
    nlp = None


@dataclass
class EntityRecord:
    name: str
    entity_type: str
    chunk_ids: List[str] = field(default_factory=list)
    frequency: int = 1
    metadata: Dict = field(default_factory=dict)


@dataclass  
class RelationRecord:
    source: str
    target: str
    relation_type: str = "RELATES_TO"
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


class GraphExtractor:
    """Extracts entities and relationships from chunks for knowledge graph"""
    
    # NER label mapping
    NER_TYPE_MAPPING = {
        "PERSON": "person",
        "ORG": "organization", 
        "GPE": "location",
        "PRODUCT": "product",
        "DATE": "date",
        "MONEY": "metric",
        "PERCENT": "metric",
        "FAC": "facility",
        "EVENT": "event"
    }
    
    # Technical entity patterns
    TECHNICAL_PATTERNS = [
        # AI/ML Models
        (r'\b(GPT-[234]|BERT|RoBERTa|T5|CLIP|DALL-E|Claude|LLaMA|Mistral)\b', 'model'),
        (r'\b(transformer|attention|encoder|decoder|embedding)\s+model\b', 'model'),
        
        # Frameworks & Libraries
        (r'\b(PyTorch|TensorFlow|Keras|scikit-learn|pandas|numpy|FastAPI|Django)\b', 'library'),
        (r'\b(LangChain|LlamaIndex|Haystack|Semantic Kernel)\b', 'framework'),
        
        # Databases
        (r'\b(PostgreSQL|MySQL|MongoDB|Redis|Neo4j|Qdrant|Pinecone|Weaviate)\b', 'database'),
        (r'\b(vector\s+(database|db|store))\b', 'database'),
        
        # Cloud Services
        (r'\b(AWS|Azure|GCP|Google Cloud)\b', 'cloud_service'),
        (r'\b(S3|EC2|Lambda|SageMaker|Bedrock)\b', 'cloud_service'),
        
        # Algorithms
        (r'\b(HNSW|IVF|LSH|k-means|PageRank|BFS|DFS)\b', 'algorithm'),
        
        # RAG Concepts
        (r'\b(RAG|retrieval[- ]augmented|GraphRAG|HybridRAG)\b', 'rag_pattern'),
        (r'\b(chunking|embedding|reranking|retrieval)\b', 'rag_component'),
        
        # Prompting
        (r'\b(few-shot|zero-shot|chain-of-thought|CoT|ReAct)\b', 'prompting'),
    ]
    
    def __init__(self, use_spacy: bool = True, extract_technical: bool = True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.extract_technical = extract_technical
        
        if self.use_spacy and not nlp:
            logger.warning("spaCy requested but not available")
            self.use_spacy = False
    
    def extract_from_chunks(self, chunks: List[Chunk], 
                           document_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from chunks"""
        entity_map: Dict[str, EntityRecord] = {}
        relation_counts: Dict[Tuple[str, str, str], float] = defaultdict(float)
        
        for chunk in chunks:
            # Extract entities from chunk
            entities_in_chunk = self._extract_entities(chunk)
            
            # Update entity records
            for entity_name, entity_type in entities_in_chunk:
                key = entity_name.lower()
                if key not in entity_map:
                    entity_map[key] = EntityRecord(
                        name=entity_name,
                        entity_type=entity_type,
                        chunk_ids=[chunk.id]
                    )
                else:
                    if chunk.id not in entity_map[key].chunk_ids:
                        entity_map[key].chunk_ids.append(chunk.id)
                        entity_map[key].frequency += 1
            
            # Extract relationships
            if len(entities_in_chunk) > 1:
                relationships = self._extract_relationships(chunk, entities_in_chunk)
                for source, target, rel_type in relationships:
                    key = (source.lower(), target.lower(), rel_type)
                    relation_counts[key] += 1.0
        
        # Convert to output format
        entities = []
        for record in entity_map.values():
            entity = Entity(
                name=record.name,
                entity_type=record.entity_type,
                document_ids=[document_id],
                chunk_ids=record.chunk_ids,
                frequency=record.frequency,
                metadata=record.metadata
            )
            entities.append(entity)
        
        relationships = []
        for (source, target, rel_type), weight in relation_counts.items():
            # Get proper names
            source_name = entity_map.get(source, EntityRecord(name=source, entity_type="unknown")).name
            target_name = entity_map.get(target, EntityRecord(name=target, entity_type="unknown")).name
            
            # Find entity IDs
            source_entity = next((e for e in entities if e.name.lower() == source), None)
            target_entity = next((e for e in entities if e.name.lower() == target), None)
            
            if source_entity and target_entity:
                relationship = Relationship(
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relationship_type=rel_type,
                    weight=weight,
                    document_ids=[document_id]
                )
                relationships.append(relationship)
        
        # Sort by relevance
        entities.sort(key=lambda e: e.frequency, reverse=True)
        relationships.sort(key=lambda r: r.weight, reverse=True)
        
        return entities, relationships
    
    def _extract_entities(self, chunk: Chunk) -> List[Tuple[str, str]]:
        """Extract entities from a chunk"""
        entities = []
        text = chunk.content
        
        # Use spaCy NER if available
        if self.use_spacy and nlp:
            doc = nlp(text[:10000])  # Limit text length
            for ent in doc.ents:
                entity_type = self.NER_TYPE_MAPPING.get(ent.label_, ent.label_.lower())
                entities.append((ent.text, entity_type))
            
            # Extract noun phrases as potential entities
            for chunk in doc.noun_chunks:
                if any(token.pos_ == "PROPN" for token in chunk):
                    # Check if not already captured
                    chunk_text = chunk.text
                    if not any(chunk_text in e[0] or e[0] in chunk_text for e in entities):
                        entities.append((chunk_text, "concept"))
        
        # Extract technical entities using patterns
        if self.extract_technical:
            for pattern, entity_type in self.TECHNICAL_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group(0)
                    # Avoid duplicates
                    if not any(entity_text.lower() == e[0].lower() for e in entities):
                        entities.append((entity_text, entity_type))
        
        # Filter and clean entities
        cleaned_entities = []
        for entity_text, entity_type in entities:
            entity_text = entity_text.strip()
            if len(entity_text) > 2 and len(entity_text) < 100:  # Reasonable length
                cleaned_entities.append((entity_text, entity_type))
        
        return cleaned_entities
    
    def _extract_relationships(self, chunk: Chunk, 
                              entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities in a chunk"""
        relationships = []
        text = chunk.content.lower()
        
        # Extract relationships using spaCy dependency parsing
        if self.use_spacy and nlp and len(entities) > 1:
            doc = nlp(chunk.content[:10000])
            entity_names = [e[0] for e in entities]
            
            # Find relationships based on dependency patterns
            for ent1_name, _ in entities:
                for ent2_name, _ in entities:
                    if ent1_name != ent2_name:
                        rel_type = self._find_relationship_type(
                            doc, ent1_name, ent2_name, text
                        )
                        if rel_type:
                            relationships.append((ent1_name, ent2_name, rel_type))
        
        # Fallback to co-occurrence relationships
        if not relationships:
            entity_names = [e[0] for e in entities]
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    # Check proximity
                    name1, name2 = entity_names[i], entity_names[j]
                    if self._are_related(name1, name2, text):
                        relationships.append((name1, name2, "CO_OCCURS"))
        
        return relationships
    
    def _find_relationship_type(self, doc, ent1: str, ent2: str, text: str) -> Optional[str]:
        """Determine relationship type between two entities"""
        text_between = self._get_text_between(ent1.lower(), ent2.lower(), text)
        
        if not text_between:
            return None
        
        # Check for specific relationship patterns
        patterns = {
            "USES": ["uses", "utilizes", "employs", "leverages"],
            "IMPLEMENTS": ["implements", "realizes", "executes"],
            "CONTAINS": ["contains", "includes", "comprises", "has"],
            "DEPENDS_ON": ["depends on", "requires", "needs"],
            "PRODUCES": ["produces", "generates", "creates", "outputs"],
            "IMPROVES": ["improves", "enhances", "optimizes"],
            "COMPARES_TO": ["versus", "vs", "compared to", "better than"],
            "PART_OF": ["part of", "component of", "belongs to"],
            "INTEGRATES": ["integrates", "connects", "interfaces"]
        }
        
        for rel_type, keywords in patterns.items():
            if any(kw in text_between for kw in keywords):
                return rel_type
        
        # Default relationship if entities are close
        if len(text_between.split()) < 10:
            return "RELATES_TO"
        
        return None
    
    def _get_text_between(self, ent1: str, ent2: str, text: str) -> str:
        """Get text between two entities"""
        try:
            # Find positions
            pos1 = text.find(ent1)
            pos2 = text.find(ent2)
            
            if pos1 == -1 or pos2 == -1:
                return ""
            
            # Get text between them
            if pos1 < pos2:
                return text[pos1 + len(ent1):pos2]
            else:
                return text[pos2 + len(ent2):pos1]
        except:
            return ""
    
    def _are_related(self, ent1: str, ent2: str, text: str) -> bool:
        """Check if two entities are related based on proximity"""
        try:
            pos1 = text.lower().find(ent1.lower())
            pos2 = text.lower().find(ent2.lower())
            
            if pos1 == -1 or pos2 == -1:
                return False
            
            # Check if within reasonable distance (e.g., same sentence/paragraph)
            distance = abs(pos2 - pos1)
            return distance < 200  # Within ~200 characters
        except:
            return False