import json
import os
from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModel
from flask_cors import CORS
import re
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from urllib.parse import urlparse


from parse_utils import parse_resume_spacy, extract_skills_from_text

app = Flask(__name__)
CORS(app)

load_dotenv() 
# 1) Set your OpenAI API key from environment variable or direct
openai.api_key = os.getenv("OPENAI_API_KEY")


# 2) Hugging Face model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

DATABASE_URL = os.getenv("DATABASE_URL")  # Provided by Render

result = urlparse(DATABASE_URL)

DB_CONFIG = {
    "dbname": result.path[1:],      # path typically starts with '/'
    "user": result.username,
    "password": result.password,
    "host": result.hostname,
    "port": result.port
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def generate_embedding(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
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

    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    emb = generate_embedding(text)
    emb_list = emb.tolist()

    parsed_data = parse_resume_spacy(text)

    parsed_data["skills"] = extract_skills_from_text(text)
    # Placeholder experiences
    parsed_data["experiences"] = [
        {
            "company": "Acme Corp",
            "roles": [
                {"title": "Software Engineer", "period": "Jan 2020 - Present"}
            ]
        },
        {
            "company": "Example Inc",
            "roles": [
                {"title": "Intern", "period": "Jun 2019 - Dec 2019"}
            ]
        }
    ]

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

# 2) Extract Skills (Utility)
# (unchanged) ... you already have `extract_skills_from_text`.

# 3) Upload Job by URL
@app.route("/upload_job_by_url", methods=["POST"])
def upload_job_by_url():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    job_url = data.get("job_url")
    if not job_url:
        return jsonify({"error": "No 'job_url' provided"}), 400

    from urllib.parse import urlparse
    try:
        resp = requests.get(job_url)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 400

    parsed_url = urlparse(job_url)
    website_name = parsed_url.netloc

    # 1) Scrape job text
    soup = BeautifulSoup(resp.text, "html.parser")
    job_text = soup.get_text(separator="\n")

    # 2) GPT extraction
    extracted_data = call_gpt_for_job_extraction(website_name, job_text)

    # 3) Generate an embedding (like before)
    emb = generate_embedding(job_text)
    emb_list = emb.tolist()

    # 4) job_title from GPT or fallback
    title = extracted_data.get("job_title", "Unknown Title")

    # 5) job_required_skills from your standard function or from GPT
    job_required_skills = extract_skills_from_text(job_text)

    # 6) Insert into DB
    conn = get_connection()
    cur = conn.cursor()
    insert_query = """
        INSERT INTO job_descriptions (
            title,
            description,
            embedding,
            required_skills,
            structured_data
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """
    cur.execute(
        insert_query,
        (
            title,
            job_text,
            emb_list,
            Json(job_required_skills),
            Json(extracted_data),
        ),
    )
    job_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "message": "Job from URL uploaded successfully with structured info.",
        "job_id": job_id,
        "website_name": extracted_data.get("website_name", website_name),
        "structured_data": extracted_data,
        "required_skills": job_required_skills
    }), 200

def normalize_skills(skills):
    cleaned = set()
    for s in skills:
        # Trim spaces and convert to lowercase
        s = s.strip().lower()
        # Remove trailing punctuation or commas. Adjust to your needs:
        s = re.sub(r'[^\w\+\-\.#]', '', s)  
        cleaned.add(s)
    return cleaned

def call_gpt_for_job_extraction(website, job_text):
    """
    Calls GPT to parse 'job_text' from 'website' into a strict JSON format with:
    {
      "website_name": "string",
      "job_title": "string",
      "company": "string",
      "experience_required": "string",
      "education": "string",
      "skills_required": ["skill1","skill2","skill3"],
      "role_description": "string"
    }
    Returns a dict with those keys or a fallback if GPT fails to produce valid JSON.
    """
    system_instructions = """
    You are an AI that extracts structured data from a job posting.
    Return ONLY valid JSON (no extra text). 
    Must include exactly these fields:
    {
      "website_name": "string",
      "job_title": "string",
      "company": "string",
      "experience_required": "string",
      "education": "string",
      "skills_required": ["skill1","skill2","skill3"],
      "role_description": "string"
    }
    No disclaimers or extra formatting. 
    """

    user_prompt = f"""
    Website: {website}
    Job text:
    ---
    {job_text[:4000]}
    ---
    Parse the text to fill these fields. 
    If a field is unknown, you can put "Unknown" or an empty list.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_instructions.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            max_tokens=600
        )
        gpt_output = response.choices[0].message["content"].strip()

        # Attempt to extract JSON via a regex or parse the entire string
        match = re.search(r"\{.*\}", gpt_output, flags=re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data
        else:
            print("No JSON braces found in GPT output, returning fallback.")
            return {
                "website_name": website,
                "job_title": "Unknown",
                "company": "Unknown",
                "experience_required": "Unknown",
                "education": "Unknown",
                "skills_required": [],
                "role_description": job_text[:200]
            }
    except Exception as e:
        print("Error calling GPT for job extraction:", e)
        return {
            "website_name": website,
            "job_title": "Unknown",
            "company": "Unknown",
            "experience_required": "Unknown",
            "education": "Unknown",
            "skills_required": [],
            "role_description": job_text[:200]
        }

# 5) GPT-3.5 Call for Analysis (Enhanced)
def call_gpt_for_analysis(job_req_skills, resume_skills, similarity_score):
    # Convert both sets to normalized forms
    job_req_lower = normalize_skills(job_req_skills)
    resume_skills_lower = normalize_skills(resume_skills)

    # Now compute missing
    missing = list(job_req_lower - resume_skills_lower)

    # Additional context for GPT about each dimension:
    system_instructions = """
    You are a specialized recruitment assistant that compares candidate resumes to job requirements.
    We have four numeric categories (0-100):
      1) technical: how well the candidate's technical skills match the job requirements
      2) experience: how well the candidate's work experience, years, domains, etc. fit the job
      3) education: how well the candidate's education matches job requirements
      4) tooling: how well the candidate knows required tools (like Docker, AWS, Git)

    Output must be a SINGLE JSON object with these fields exactly:
    {
      "skillAnalysis": {
        "technical": number,
        "experience": number,
        "education": number,
        "tooling": number
      },
      "recommendations": [
        {
          "title": "...",
          "description": "...",
          "resources": [{ "name": "...", "url": "..." }]
        }
      ]
    }

    No disclaimers or text outside the JSON. No code blocks. 
    """

    user_prompt = f"""
    Job requires these skills (lowercased): {sorted(job_req_lower)}
    Candidate has these skills (lowercased): {sorted(resume_skills_lower)}
    Missing: {sorted(missing)}
    Similarity score: {similarity_score:.2f}

    1) Provide skillAnalysis (technical, experience, education, tooling). 
       Each is from 0 to 100, with 100 being a perfect match.
    2) Provide up to 2 recommendations for the candidate. 
       Use the "recommendations" array with "title", "description", and "resources".

    Return valid JSON ONLY. 
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,  # minimal creativity
            messages=[
                {"role": "system", "content": system_instructions.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
        )
        gpt_output = response.choices[0].message["content"].strip()

        # Attempt to find JSON
        match = re.search(r"\{.*\}", gpt_output, flags=re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            return parsed
        else:
            print("No curly braces found in GPT output.")
            return {
                "skillAnalysis": {
                    "technical": 0,
                    "experience": 0,
                    "education": 0,
                    "tooling": 0
                },
                "recommendations": []
            }
    except Exception as e:
        print("Error calling GPT:", e)
        return {
            "skillAnalysis": {
                "technical": 0,
                "experience": 0,
                "education": 0,
                "tooling": 0
            },
            "recommendations": []
        }

# 6) MATCH with GPT-based Analysis
@app.route("/match/<int:job_id>", methods=["GET"])
def match_resumes(job_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT embedding::float4[] AS job_emb, required_skills
        FROM job_descriptions
        WHERE id = %s
    """, (job_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return jsonify({"error": "Job not found"}), 404

    job_embedding_list = row["job_emb"]
    job_embedding = np.array(job_embedding_list, dtype=np.float32) if job_embedding_list else None
    job_req_skills = row["required_skills"] or []

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
        # call GPT for skill analysis
        gpt_json = call_gpt_for_analysis(job_req_skills, resume_skills, sim_score)
        skillAnalysis = gpt_json.get("skillAnalysis", {})
        recommendations = gpt_json.get("recommendations", [])

        results.append({
            "resume_id": r_id,
            "name": r_name,
            "email": r_email,
            "similarity_score": sim_score,
            "resume_skills": resume_skills,
            "job_required_skills": job_req_skills,
            "missing_skills": list(set(job_req_skills) - set(resume_skills)),
            "skillAnalysis": skillAnalysis,
            "recommendations": recommendations
        })

    # Sort by highest similarity
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return jsonify({"matches": results[:5]})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
