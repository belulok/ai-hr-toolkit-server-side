-- db_setup.sql

-- 1) Enable the pgvector extension (for 'vector' type) if not already
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Create resumes table
CREATE TABLE IF NOT EXISTS resumes (
  id               SERIAL PRIMARY KEY,
  name             VARCHAR(255),
  email            VARCHAR(255),
  raw_text         TEXT,
  embedding        VECTOR(768),
  extracted_skills JSONB,
  parsed_data      JSONB
);

-- 3) Create job_descriptions table
CREATE TABLE IF NOT EXISTS job_descriptions (
  id              SERIAL PRIMARY KEY,
  title           VARCHAR(255),
  description     TEXT,
  embedding       VECTOR(768),
  required_skills JSONB,
  structured_data JSONB
);
