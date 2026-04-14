-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id            SERIAL PRIMARY KEY,
    name          TEXT        NOT NULL,
    image_data    BYTEA,                  -- raw bytes of the original uploaded image
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-product feature store
CREATE TABLE IF NOT EXISTS features (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER     NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    sift_desc       BYTEA,                  -- pickle-serialized numpy (N, 128) float32
    hist_hsv        BYTEA,                  -- pickle-serialized numpy (50, 60) float32
    hu_moments      DOUBLE PRECISION[],     -- 7-element log-Hu vector
    corner_density  DOUBLE PRECISION,       -- normalized Harris corner scalar
    embedding       VECTOR(3000)            -- flattened, L2-normed HSV histogram for ANN
);

CREATE INDEX IF NOT EXISTS features_product_id_idx ON features(product_id);

-- pgvector ANN index (IVFFlat — created after data load, so deferred to app startup)
-- CREATE INDEX ON features USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
