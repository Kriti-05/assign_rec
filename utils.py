from sentence_transformers import SentenceTransformer
import requests, re
import streamlit as st  # Use Streamlit secrets instead of .env

# ---------------- Secrets ----------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------------- Sentence Transformer ----------------
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Embedding ----------------
def embed_text_local(texts):
    vecs = _model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]

# ---------------- Groq API Call ----------------
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

# ---------------- Predict Job Roles ----------------
def groq_predict_job_roles(title, description):
    prompt = f"Title: {title}\nDescription: {description}\nList 2–3 common job roles."
    result = _groq_call(prompt, "You are an HR domain expert.")
    roles = [r.strip() for r in re.split(r"[,/]", result) if r.strip()]
    return roles or ["General Roles"]

# ---------------- Classify Test Types ----------------
def groq_classify_test_types(title, description):
    prompt = f"Title: {title}\nDescription: {description}\nPredict test type letters."
    result = _groq_call(prompt, "You are an SHL assessment classifier.")
    codes = re.findall(r"[ABCDKEPS]", result.upper())
    return list(set(codes)) or ["Unknown"]
