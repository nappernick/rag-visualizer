import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = f"sqlite:///{Path(__file__).parent / 'test_idmap.db'}"

from backend.src.main import app  # noqa: E402
from backend.src.db import init_db  # noqa: E402


def setup_module(module):
    # Ensure tables exist for sqlite db
    init_db()


client = TestClient(app)


def test_add_and_traverse_links():
    # Add bidirectional link between doc and chunk
    payload = {
        "a_type": "document",
        "a_id": "doc1",
        "b_type": "chunk",
        "b_id": "chunk1",
        "relation": "contains",
        "bidirectional": True,
    }
    r = client.post("/api/idmap/link", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2

    # Traverse from document
    tr = client.get("/api/idmap/traverse", params={"id_type": "document", "id_value": "doc1"})
    assert tr.status_code == 200
    rel = tr.json()["related"]
    assert "chunk" in rel and "chunk1" in rel["chunk"]

    # Traverse from chunk
    tr2 = client.get("/api/idmap/traverse", params={"id_type": "chunk", "id_value": "chunk1"})
    rel2 = tr2.json()["related"]
    assert "document" in rel2 and "doc1" in rel2["document"]


def test_specialized_endpoints():
    # Test document -> chunks endpoint
    r = client.get("/api/idmap/document/doc1/chunks")
    assert r.status_code == 200
    assert "chunk1" in r.json()["chunk_ids"]

    # Test chunk -> vector endpoint
    payload = {
        "a_type": "chunk",
        "a_id": "chunk1",
        "b_type": "vector",
        "b_id": "vec1",
        "relation": "embeds",
    }
    client.post("/api/idmap/link", json=payload)
    
    r = client.get("/api/idmap/chunk/chunk1/vector")
    assert r.status_code == 200
    assert r.json()["vector_id"] == "vec1"


def test_ingest_document_links():
    ingest = {
        "id": "doc2",
        "chunk_ids": ["c1", "c2"],
        "entity_ids": ["e1"],
        "chunks": [
            {"id": "c1", "vector_id": "v1", "entity_ids": ["e1", "e2"], "graph_node_ids": ["g1"]},
            {"id": "c2", "parent_id": "c1"},
        ],
    }
    r = client.post("/api/idmap/ingest", json=ingest)
    assert r.status_code == 200
    assert r.json()["links_created"] > 0

    # Traverse from doc -> chunks, entities
    tr = client.get("/api/idmap/traverse", params={"id_type": "document", "id_value": "doc2"})
    rel = tr.json()["related"]
    assert set(rel.get("chunk", [])) == {"c1", "c2"}
    assert "e1" in rel.get("entity", [])

    # Traverse from chunk -> vector, entity, parent, graph_node
    trc1 = client.get("/api/idmap/traverse", params={"id_type": "chunk", "id_value": "c1"})
    relc1 = trc1.json()["related"]
    assert "v1" in relc1.get("vector", [])
    assert "e1" in relc1.get("entity", []) and "e2" in relc1.get("entity", [])
    assert "g1" in relc1.get("graph_node", [])

    trc2 = client.get("/api/idmap/traverse", params={"id_type": "chunk", "id_value": "c2"})
    relc2 = trc2.json()["related"]
    assert "c1" in relc2.get("chunk", [])


def test_related_entities():
    # Add more relationships for testing
    links = [
        {"a_type": "entity", "a_id": "e1", "b_type": "chunk", "b_id": "c3"},
        {"a_type": "entity", "a_id": "e3", "b_type": "chunk", "b_id": "c3"},
        {"a_type": "entity", "a_id": "e3", "b_type": "chunk", "b_id": "c4"},
        {"a_type": "entity", "a_id": "e4", "b_type": "chunk", "b_id": "c4"},
    ]
    r = client.post("/api/idmap/bulk", json=links)
    assert r.status_code == 200

    # Get related entities
    r = client.get("/api/idmap/entity/e1/related", params={"max_depth": 2})
    assert r.status_code == 200
    related = r.json()["related"]
    assert "depth_1" in related  # Should find e3 through c3