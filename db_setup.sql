-- If you haven't created the database and user yet, do that inside psql:
-- CREATE DATABASE ai_hr_toolkit;
-- CREATE USER ai_hr_user WITH ENCRYPTED PASSWORD 'mypassword';
-- GRANT ALL PRIVILEGES ON DATABASE ai_hr_toolkit TO ai_hr_user;

-- Now, connect to the DB (psql -U ai_hr_user -d ai_hr_toolkit), then run:
-- \i db_setup.sql

-- OPTIONAL: If you want pgvector:
-- CREATE EXTENSION IF NOT EXISTS vector;

-- For a 768-dimensional model (e.g., all-mpnet-base-v2):
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    raw_text TEXT,
    embedding VECTOR(768)
);

CREATE TABLE IF NOT EXISTS job_descriptions (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    description TEXT,
    embedding VECTOR(768)
);
