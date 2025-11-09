import os
import re
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# -------- Config --------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- Embedding ----------
def embed_text_local(texts):
    """Return list of normalized embedding vectors."""
    vecs = _model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


# ---------- General LLM helper ----------
def _groq_call(prompt, system_role="You are an expert HR assistant."):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 128,
        "stream": False
    }

    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Groq API error: {e}")
        return "Unknown"


# ---------- Extract Job Role ----------
def groq_predict_job_roles(title, description):
    prompt = f"""
Given the following SHL assessment title and description, list 2–3 common job roles that would typically take this assessment.
Return a comma-separated list like "Software Engineer, Data Analyst, HR Manager".

Title: {title}
Description: {description}
"""
    result = _groq_call(prompt, "You are an HR domain expert mapping SHL assessments to job roles.")
    roles = [r.strip() for r in re.split(r"[,/]", result) if r.strip()]
    return roles or ["General Roles"]


# ---------- Extract Test Type ----------
def groq_classify_test_types(title, description):
    prompt = f"""
Each SHL assessment belongs to one or more of these Test Types:
A: Ability & Aptitude
B: Biodata & Situational Judgement
C: Competencies
D: Development & 360
E: Assessment Exercises
K: Knowledge & Skills
P: Personality & Behavior
S: Simulations

Given the assessment title and description, predict the most relevant test type letters (comma-separated).
Example output: "A,K" or "P".

Title: {title}
Description: {description}
"""
    result = _groq_call(prompt, "You are an expert SHL assessment classifier.")
    codes = re.findall(r"[ABCDKEPS]", result.upper())
    return list(set(codes)) or ["Unknown"]
