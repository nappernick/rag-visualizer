from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import IDLink


class IDMapper:
    """Service for comprehensive bidirectional ID mapping and traversal."""

    def __init__(self, session: Session):
        self.session = session

    def add_link(
        self,
        a_type: str,
        a_id: str,
        b_type: str,
        b_id: str,
        relation: str | None = None,
        bidirectional: bool = True,
    ) -> List[IDLink]:
        created: List[IDLink] = []
        pairs: List[Tuple[str, str, str, str]] = [(a_type, a_id, b_type, b_id)]
        if bidirectional:
            pairs.append((b_type, b_id, a_type, a_id))

        for at, aid, bt, bid in pairs:
            link = IDLink(a_type=at, a_id=aid, b_type=bt, b_id=bid, relation=relation)
            try:
                self.session.add(link)
                self.session.flush()
                created.append(link)
            except IntegrityError:
                self.session.rollback()
                # Retrieve existing
                existing = self.session.execute(
                    select(IDLink).where(
                        and_(
                            IDLink.a_type == at,
                            IDLink.a_id == aid,
                            IDLink.b_type == bt,
                            IDLink.b_id == bid,
                            IDLink.relation.is_(relation) if relation is None else IDLink.relation == relation,
                        )
                    )
                ).scalar_one_or_none()
                if existing:
                    created.append(existing)

        return created

    def get_links_for(self, id_type: str, id_value: str) -> List[IDLink]:
        stmt = select(IDLink).where(
            or_(
                and_(IDLink.a_type == id_type, IDLink.a_id == id_value),
                and_(IDLink.b_type == id_type, IDLink.b_id == id_value),
            )
        )
        return list(self.session.execute(stmt).scalars().all())

    def traverse(self, id_type: str, id_value: str) -> Dict[str, List[str]]:
        related: Dict[str, set] = defaultdict(set)
        for link in self.get_links_for(id_type, id_value):
            if link.a_type == id_type and link.a_id == id_value:
                related[link.b_type].add(link.b_id)
            if link.b_type == id_type and link.b_id == id_value:
                related[link.a_type].add(link.a_id)

        return {k: sorted(list(v)) for k, v in related.items()}

    # Specialized traversal methods for common use cases
    def get_chunks_for_document(self, doc_id: str) -> List[str]:
        """Get all chunk IDs for a document."""
        related = self.traverse("document", doc_id)
        return related.get("chunk", [])

    def get_entities_for_chunk(self, chunk_id: str) -> List[str]:
        """Get all entity IDs for a chunk."""
        related = self.traverse("chunk", chunk_id)
        return related.get("entity", [])

    def get_documents_for_entity(self, entity_id: str) -> List[str]:
        """Get all document IDs for an entity."""
        related = self.traverse("entity", entity_id)
        return related.get("document", [])

    def get_entities_for_document(self, doc_id: str) -> List[str]:
        """Get all entity IDs for a document."""
        related = self.traverse("document", doc_id)
        return related.get("entity", [])

    def get_chunks_for_entity(self, entity_id: str) -> List[str]:
        """Get all chunk IDs for an entity."""
        related = self.traverse("entity", entity_id)
        return related.get("chunk", [])

    def get_vector_for_chunk(self, chunk_id: str) -> Optional[str]:
        """Get the vector ID for a chunk."""
        related = self.traverse("chunk", chunk_id)
        vectors = related.get("vector", [])
        return vectors[0] if vectors else None

    def get_graph_nodes_for_chunk(self, chunk_id: str) -> List[str]:
        """Get all graph node IDs for a chunk."""
        related = self.traverse("chunk", chunk_id)
        return related.get("graph_node", [])

    def get_chunks_with_vectors(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Get mapping of chunk IDs to their vector IDs."""
        result = {}
        for chunk_id in chunk_ids:
            vector_id = self.get_vector_for_chunk(chunk_id)
            if vector_id:
                result[chunk_id] = vector_id
        return result

    def get_related_entities(self, entity_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """Get related entities through shared documents/chunks up to max_depth."""
        visited = {entity_id}
        result = defaultdict(list)
        current_level = [entity_id]
        
        for depth in range(max_depth):
            next_level = []
            for eid in current_level:
                # Get chunks for this entity
                chunks = self.get_chunks_for_entity(eid)
                
                # Get other entities from those chunks
                for chunk_id in chunks:
                    related_entities = self.get_entities_for_chunk(chunk_id)
                    for related_id in related_entities:
                        if related_id not in visited:
                            visited.add(related_id)
                            result[f"depth_{depth + 1}"].append(related_id)
                            next_level.append(related_id)
            
            current_level = next_level
            if not current_level:
                break
        
        return dict(result)

    def bulk_add_links(self, links: List[Dict[str, any]]) -> int:
        """Bulk add multiple ID links efficiently."""
        count = 0
        for link_data in links:
            created = self.add_link(
                a_type=link_data['a_type'],
                a_id=link_data['a_id'],
                b_type=link_data['b_type'],
                b_id=link_data['b_id'],
                relation=link_data.get('relation'),
                bidirectional=link_data.get('bidirectional', True)
            )
            count += len(created)
        return count