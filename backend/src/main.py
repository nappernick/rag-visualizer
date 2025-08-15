from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Configure logging to show all logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables from parent directory if .env exists
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logging.info(f"Loaded .env from {env_path}")
else:
    logging.info("No .env file found - using environment variables")

from .db import get_session, init_db
from .models import IDLinkIn, IDLinkOut, TraverseResponse, IngestDocument
from .services.id_mapper import IDMapper
from .api import fusion, documents, chunking, graph, visualization, query, concurrent_processing, demo


def get_db():
    with get_session() as s:
        yield s


app = FastAPI(title="RAG Visualizer Backend", version="0.1.0")

# Wide open CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()

# Include API routes
app.include_router(fusion.router)
app.include_router(documents.router)
app.include_router(chunking.router)
app.include_router(graph.router)
app.include_router(visualization.router)
app.include_router(query.router)
app.include_router(concurrent_processing.router)
app.include_router(demo.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# ID Mapping endpoints
@app.post("/api/idmap/link", response_model=List[IDLinkOut])
def add_link(payload: IDLinkIn, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    links = mapper.add_link(
        a_type=payload.a_type,
        a_id=payload.a_id,
        b_type=payload.b_type,
        b_id=payload.b_id,
        relation=payload.relation,
        bidirectional=payload.bidirectional,
    )
    return [
        IDLinkOut(
            id=l.id,
            a_type=l.a_type,
            a_id=l.a_id,
            b_type=l.b_type,
            b_id=l.b_id,
            relation=l.relation,
            created_at=l.created_at,
        )
        for l in links
    ]


@app.get("/api/idmap/links", response_model=List[IDLinkOut])
def get_links(id_type: str, id_value: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    links = mapper.get_links_for(id_type, id_value)
    return [
        IDLinkOut(
            id=l.id,
            a_type=l.a_type,
            a_id=l.a_id,
            b_type=l.b_type,
            b_id=l.b_id,
            relation=l.relation,
            created_at=l.created_at,
        )
        for l in links
    ]


@app.get("/api/idmap/traverse", response_model=TraverseResponse)
def traverse(id_type: str, id_value: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    related = mapper.traverse(id_type, id_value)
    return TraverseResponse(origin_type=id_type, origin_id=id_value, related=related)


# Specialized traversal endpoints
@app.get("/api/idmap/document/{doc_id}/chunks")
def get_document_chunks(doc_id: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    chunk_ids = mapper.get_chunks_for_document(doc_id)
    return {"document_id": doc_id, "chunk_ids": chunk_ids, "count": len(chunk_ids)}


@app.get("/api/idmap/document/{doc_id}/entities")
def get_document_entities(doc_id: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    entity_ids = mapper.get_entities_for_document(doc_id)
    return {"document_id": doc_id, "entity_ids": entity_ids, "count": len(entity_ids)}


@app.get("/api/idmap/chunk/{chunk_id}/entities")
def get_chunk_entities(chunk_id: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    entity_ids = mapper.get_entities_for_chunk(chunk_id)
    return {"chunk_id": chunk_id, "entity_ids": entity_ids, "count": len(entity_ids)}


@app.get("/api/idmap/chunk/{chunk_id}/vector")
def get_chunk_vector(chunk_id: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    vector_id = mapper.get_vector_for_chunk(chunk_id)
    return {"chunk_id": chunk_id, "vector_id": vector_id}


@app.get("/api/idmap/entity/{entity_id}/documents")
def get_entity_documents(entity_id: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    doc_ids = mapper.get_documents_for_entity(entity_id)
    return {"entity_id": entity_id, "document_ids": doc_ids, "count": len(doc_ids)}


@app.get("/api/idmap/entity/{entity_id}/chunks")
def get_entity_chunks(entity_id: str, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    chunk_ids = mapper.get_chunks_for_entity(entity_id)
    return {"entity_id": entity_id, "chunk_ids": chunk_ids, "count": len(chunk_ids)}


@app.get("/api/idmap/entity/{entity_id}/related")
def get_related_entities(entity_id: str, max_depth: int = 2, db: Session = Depends(get_db)):
    mapper = IDMapper(db)
    related = mapper.get_related_entities(entity_id, max_depth)
    return {"entity_id": entity_id, "related": related, "max_depth": max_depth}


@app.post("/api/idmap/ingest")
def ingest_document(payload: IngestDocument, db: Session = Depends(get_db)):
    """Populate ID mappings for a single document.

    Links created (if data present):
    - document -> chunk
    - document -> entity
    - chunk -> entity
    - chunk -> vector
    - chunk -> graph_node
    - chunk parent relations (chunk -> chunk with relation="parent")
    """
    mapper = IDMapper(db)
    links_created = 0

    # Document to chunks
    if payload.chunk_ids:
        for cid in payload.chunk_ids:
            mapper.add_link("document", payload.id, "chunk", cid, relation="contains")
            links_created += 1

    # Document to entities
    if payload.entity_ids:
        for eid in payload.entity_ids:
            mapper.add_link("document", payload.id, "entity", eid, relation="mentions")
            links_created += 1

    # Detailed chunk data
    if payload.chunks:
        for ch in payload.chunks:
            # Ensure doc->chunk
            mapper.add_link("document", payload.id, "chunk", ch.id, relation="contains")
            links_created += 1

            # Chunk -> vector
            if ch.vector_id:
                mapper.add_link("chunk", ch.id, "vector", ch.vector_id, relation="embeds")
                links_created += 1

            # Chunk -> entities
            if ch.entity_ids:
                for eid in ch.entity_ids:
                    mapper.add_link("chunk", ch.id, "entity", eid, relation="mentions")
                    links_created += 1

            # Chunk -> graph nodes
            if ch.graph_node_ids:
                for gid in ch.graph_node_ids:
                    mapper.add_link("chunk", ch.id, "graph_node", gid, relation="represents")
                    links_created += 1

            # Parent relation (child -> parent)
            if ch.parent_id:
                mapper.add_link("chunk", ch.id, "chunk", ch.parent_id, relation="parent")
                links_created += 1

    return {"status": "ok", "links_created": links_created}


@app.post("/api/idmap/bulk")
def bulk_add_links(links: List[Dict], db: Session = Depends(get_db)):
    """Bulk add multiple ID links."""
    mapper = IDMapper(db)
    count = mapper.bulk_add_links(links)
    return {"status": "ok", "links_created": count}