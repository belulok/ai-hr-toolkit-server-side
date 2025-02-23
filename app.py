import json
from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from flask_cors import CORS
import re
import requests
from bs4 import BeautifulSoup

from parse_utils import parse_resume_spacy
from parse_utils import extract_skills_from_text  # if needed

app = Flask(__name__)
CORS(app)

analysis_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"  # or "google/flan-t5-base" for smaller
)

# Load the tokenizer & model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

DB_CONFIG = {
    "dbname": "ai_hr_toolkit",
    "user": "ai_hr_user",
    "password": "123456",
    "host": "127.0.0.1",
    "port": 5432
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def generate_embedding(text):
    # Tokenize the input text
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: [batch_size, sequence_length, hidden_size]

    # 1) Simple Mean Pooling
    # We take the average of all token embeddings for each sequence in the batch
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # embeddings shape: [batch_size, hidden_size]

    # Since we only passed a single string, get the first row:
    return embeddings[0].numpy()

@app.route("/")
def home():
    return jsonify({"message": "Backend is up and running!"})

# -------------------------
# 1) UPLOAD RESUME
# -------------------------
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    file = request.files.get("resume")
    if not file:
        return jsonify({"error": "No resume file"}), 400

    # Extract PDF text
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    # Embedding
    emb = generate_embedding(text)
    emb_list = emb.tolist()

    # Parse with spaCy
    parsed_data = parse_resume_spacy(text)

    parsed_data["skills"] = extract_skills_from_text(text)

    parsed_data["experiences"] = [
        {
            "company": "Acme Corp",
            "roles": [
                {
                    "title": "Software Engineer",
                    "period": "Jan 2020 - Present"
                }
            ]
        },
        {
            "company": "Example Inc",
            "roles": [
                {
                    "title": "Intern",
                    "period": "Jun 2019 - Dec 2019"
                }
            ]
        }
    ]

    # Insert into DB
    conn = get_connection()
    cur = conn.cursor()
    insert_query = """
        INSERT INTO resumes (raw_text, embedding, parsed_data)
        VALUES (%s, %s, %s)
        RETURNING id
    """
    cur.execute(insert_query, (text, emb_list, Json(parsed_data)))
    resume_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "message": "Resume uploaded successfully (AI parsed)!",
        "resume_id": resume_id,
        "parsed_data": parsed_data,
        "raw_text": text
    }), 200

# --------------------------------
# 2) EXTRACT SKILLS (UTILITY)
# --------------------------------
def extract_skills_from_text(text):
    SKILL_LIST = [
        "python", "java", "machine learning", "nlp",
        "docker", "aws", "react", "c++", "sql",
        "deep learning", "pytorch", "tensorflow"
    ]
    found = []
    lower = text.lower()
    for skill in SKILL_LIST:
        if skill in lower:
            found.append(skill)
    return list(set(found))

# --------------------------------
# 3) UPLOAD JOB (manual text)
# --------------------------------
@app.route("/upload_job", methods=["POST"])
def upload_job():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    title = data.get("title", "")
    description = data.get("description", "")

    emb = generate_embedding(description)
    emb_list = emb.tolist()

    job_required_skills = extract_skills_from_text(description)

    conn = get_connection()
    cur = conn.cursor()
    insert_query = """
        INSERT INTO job_descriptions (title, description, embedding, required_skills)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """
    cur.execute(insert_query, (title, description, emb_list, Json(job_required_skills)))
    job_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "message": "Job uploaded successfully.",
        "job_id": job_id,
        "required_skills": job_required_skills
    }), 200

# ---------------------------------------
# 4) UPLOAD JOB BY URL
# ---------------------------------------
@app.route("/upload_job_by_url", methods=["POST"])
def upload_job_by_url():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    job_url = data.get("job_url")
    title = data.get("title", "")

    if not job_url:
        return jsonify({"error": "No 'job_url' provided"}), 400

    try:
        resp = requests.get(job_url)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 400

    soup = BeautifulSoup(resp.text, "html.parser")
    job_text = soup.get_text(separator="\n")

    emb = generate_embedding(job_text)
    emb_list = emb.tolist()

    job_required_skills = extract_skills_from_text(job_text)

    conn = get_connection()
    cur = conn.cursor()
    insert_query = """
        INSERT INTO job_descriptions (title, description, embedding, required_skills)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """
    cur.execute(insert_query, (title, job_text, emb_list, Json(job_required_skills)))
    job_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "message": "Job from URL uploaded successfully.",
        "job_id": job_id,
        "required_skills": job_required_skills
    }), 200

# -------------------------------------------------------
# 5) CALL CHATGPT FOR ANALYSIS (NEW HELPER FUNCTION)
# -------------------------------------------------------
def call_local_llm_for_analysis(job_req_skills, resume_skills, similarity_score):
    missing = list(set(job_req_skills) - set(resume_skills))

    prompt = f"""
    You are a recruitment assistant. 
    Return valid JSON with these exact fields, and no extra text:
    {{
    "skillAnalysis": {{
        "technical": 85,
        "experience": 90,
        "education": 75,
        "tooling": 80
    }},
    "recommendations": [
        {{
        "title": "Some Title",
        "description": "Some Description",
        "resources": [{{ "name": "...", "url": "..."}}]
        }}
    ]
    }}

    Important rules:
    - Output exactly one JSON object like this, with curly braces {{}} at the root.
    - Use double quotes for field names and string values.
    - No list at the top level. Must start with {{ and end with }}.
    - No extra text or disclaimers. 
    - Candidate Skills: {resume_skills}
    - Job Requirements: {job_req_skills}
    - Similarity Score: {similarity_score:.2f}

    Now produce ONLY JSON, in the format above (no code blocks).

    again...
"""

    try:
        # text-generation returns a list of dicts, each with "generated_text"
        response = analysis_pipeline(
        prompt,
        max_length=512,
        temperature=0.0,       # no randomness
        do_sample=False        # pick top tokens only
    )[0]["generated_text"]
        print("LLM RAW OUTPUT:", repr(response))
        parsed = json.loads(response)
        return parsed
    except Exception as e:
        print("Local LLM error:", e)
        # fallback
        return {
            "skillAnalysis": {
                "technical": 0,
                "experience": 0,
                "education": 0,
                "tooling": 0
            },
            "recommendations": []
        }

# ---------------------------------------
# 6) MATCH with ChatGPT-based Analysis
# ---------------------------------------
@app.route("/match/<int:job_id>", methods=["GET"])
def match_resumes(job_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT embedding::float4[] AS job_emb, required_skills
        FROM job_descriptions
        WHERE id = %s
    """, (job_id,))
    job_row = cur.fetchone()
    if not job_row:
        cur.close()
        conn.close()
        return jsonify({"error": "Job not found"}), 404

    job_embedding_list = job_row["job_emb"]
    job_embedding = np.array(job_embedding_list, dtype=np.float32) if job_embedding_list else None
    job_req_skills = job_row["required_skills"] or []

    # Retrieve resumes
    cur.execute("""
        SELECT id, name, email, embedding::float4[] AS embedding, extracted_skills
        FROM resumes
    """)
    resumes = cur.fetchall()
    cur.close()
    conn.close()

    if not resumes:
        return jsonify({"error": "No resumes found"}), 200

    results = []
    for r in resumes:
        r_id = r["id"]
        r_name = r["name"]
        r_email = r["email"]
        resume_emb_list = r["embedding"]
        if not resume_emb_list:
            continue

        resume_emb = np.array(resume_emb_list, dtype=np.float32)
        norm_job = np.linalg.norm(job_embedding) if job_embedding is not None else 0
        norm_resume = np.linalg.norm(resume_emb)
        sim_score = 0.0
        if norm_job != 0 and norm_resume != 0:
            sim_score = float(np.dot(job_embedding, resume_emb) / (norm_job * norm_resume))

        resume_skills = r.get("extracted_skills") or []
        missing_skills = list(set(job_req_skills) - set(resume_skills))

        # >>> ChatGPT-based analysis <<<
        chatgpt_json = call_local_llm_for_analysis(job_req_skills, resume_skills, sim_score)
        skillAnalysis = chatgpt_json.get("skillAnalysis", {})
        recommendations = chatgpt_json.get("recommendations", [])

        results.append({
            "resume_id": r_id,
            "name": r_name,
            "email": r_email,
            "similarity_score": sim_score,
            "resume_skills": resume_skills,
            "job_required_skills": job_req_skills,
            "missing_skills": missing_skills,
            "skillAnalysis": skillAnalysis,
            "recommendations": recommendations
        })

    # Sort by highest similarity
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    return jsonify({"matches": results[:5]})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
