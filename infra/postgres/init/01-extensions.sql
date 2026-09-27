-- Runs once, on first start with an empty data volume.
-- pgvector powers hybrid retrieval (Phase 4); the checkpointer and store create their own tables.
CREATE EXTENSION IF NOT EXISTS vector;
