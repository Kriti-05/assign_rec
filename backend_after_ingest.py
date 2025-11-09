# backend.py
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import re

load_dotenv()
app = Flask(__name__)

# ---------------- CONFIG ----------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "shl-assessments")

# ---------------- PINECONE + EMBEDDINGS ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- TEST TYPE MAPPING ----------------
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

# ---------------- HELPERS ----------------
def map_test_types(codes):
    """Map test type letters to full names."""
    return [TEST_TYPE_MAPPING.get(c, "Unknown") for c in codes]

def normalize_adaptive(adaptive):
    """If adaptive_support is unknown, return 'No'."""
    if adaptive.lower() == "unknown":
        return "No"
    return adaptive

# ---------------- RECOMMEND ENDPOINT ----------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")
    k = int(data.get("k", 5))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # 1️⃣ Embed query
    vec = embedder.encode([query])[0].tolist()

    # 2️⃣ Query Pinecone
    try:
        res = index.query(vector=vec, top_k=k*2, include_metadata=True)
        matches = []

        for m in res.matches:
            meta = m.metadata
            score = m.score

            # Boost based on test type overlap
            query_test_types = data.get("test_type", [])
            if any(t in meta.get("test_type", []) for t in query_test_types):
                score += 0.15

            matches.append((score, meta))

        # Sort by score and take top-k
        ranked = sorted(matches, key=lambda x: x[0], reverse=True)

        # Format response
        recommendations = []
        for _, m in ranked:
            recommendations.append({
                "url": m["url"],
                "name": m["name"],
                "adaptive_support": normalize_adaptive(m.get("adaptive_support", "No")),
                "description": m.get("description", ""),
                "duration": m.get("duration", 0),
                "remote_support": m.get("remote_support", "No"),
                "test_type": map_test_types(m.get("test_type", []))
            })

        return jsonify({"recommended_assessments": recommendations})

    except Exception as e:
        print("❌ Pinecone query failed:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health_check():
    """Return simple health status of the API."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
