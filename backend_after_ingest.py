from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# ---------------- ENV CONFIG ----------------
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    INDEX_NAME = st.secrets.get("PINECONE_INDEX", "shl-assessments")
    print("✅ Using Streamlit secrets")
else:
    from dotenv import load_dotenv
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX", "shl-assessments")
    print("✅ Using .env for local dev")

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# ---------------- PINECONE + EMBEDDINGS ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- HELPERS ----------------
def map_test_types(codes):
    TEST_TYPE_MAPPING = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement", 
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations"
    }
    return [TEST_TYPE_MAPPING.get(c, "Unknown") for c in codes]

def normalize_adaptive(adaptive):
    if adaptive.lower() == "unknown":
        return "No"
    return adaptive

# ---------------- RECOMMENDATION FUNCTION ----------------
def get_recommendations(query, k=5, test_type=[]):
    vec = embedder.encode([query])[0].tolist()
    res = index.query(vector=vec, top_k=k*2, include_metadata=True)
    matches = []

    for m in res.matches:
        meta = m.metadata
        score = m.score
        if any(t in meta.get("test_type", []) for t in test_type):
            score += 0.15
        matches.append((score, meta))

    ranked = sorted(matches, key=lambda x: x[0], reverse=True)
    recommendations = []
    for _, m in ranked[:k]:
        recommendations.append({
            "url": m["url"],
            "name": m["name"],
            "adaptive_support": normalize_adaptive(m.get("adaptive_support", "No")),
            "description": m.get("description", ""),
            "duration": m.get("duration", 0),
            "remote_support": m.get("remote_support", "No"),
            "test_type": map_test_types(m.get("test_type", []))
        })
    return recommendations

# ---------------- ENDPOINTS ----------------
@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    k = int(data.get("k", 5))
    test_type = data.get("test_type", [])

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        recs = get_recommendations(query, k=k, test_type=test_type)
        return jsonify({"recommended_assessments": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# ---------------- MAIN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
