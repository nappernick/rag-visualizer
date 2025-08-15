// Neo4j Schema Setup for RAG Visualizer
// Run this in Neo4j Browser or via cypher-shell

// Create constraints for unique entity IDs
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Create constraints for unique relationship IDs  
CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR ()-[r:RELATIONSHIP]->() REQUIRE r.id IS UNIQUE;

// Create indexes for better query performance
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_document_index IF NOT EXISTS FOR (e:Entity) ON (e.document_id);

// Create index for relationship types
CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]->() ON (r.type);
CREATE INDEX relationship_weight_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]->() ON (r.weight);

// Create full-text search indexes for entity names and content
CALL db.index.fulltext.createNodeIndex("entityNameFulltext", ["Entity"], ["name", "description"]) IF NOT EXISTS;

// Show created constraints and indexes
SHOW CONSTRAINTS;
SHOW INDEXES;