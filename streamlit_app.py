import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from utils import embed_text_local, groq_predict_job_roles, groq_classify_test_types

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="SHL HR Recommender", page_icon="🧠", layout="wide")
st.title("🧠 SHL HR Recommender Demo")

# ---------------- Pinecone Setup ----------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets.get("PINECONE_INDEX", "shl-assessments")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Helpers ----------------
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

def map_test_types(codes):
    return [TEST_TYPE_MAPPING.get(c, "Unknown") for c in codes]

def normalize_adaptive(adaptive):
    if adaptive.lower() == "unknown":
        return "No"
    return adaptive

# ---------------- Recommendation Function ----------------
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

# ---------------- SINGLE QUERY MODE ----------------
st.header("🎯 Single Query Mode")
query = st.text_area("Enter job description or hiring query:")

if st.button("🔍 Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                recs = get_recommendations(query, k=5)
                st.success(f"✅ Got {len(recs)} recommendations")
                if recs:
                    df = pd.DataFrame(recs)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recommendations found.")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

st.divider()

# ---------------- BATCH CSV MODE ----------------
st.header("📂 Batch Mode: Upload CSV of Queries")
uploaded = st.file_uploader("Upload CSV (must have a 'Query' column)", type=["csv"])

if uploaded:
    df_queries = pd.read_csv(uploaded)
    if "Query" not in df_queries.columns:
        st.error("❌ CSV must contain a 'Query' column.")
    else:
        st.success(f"✅ Loaded {len(df_queries)} queries.")
        st.dataframe(df_queries.head())

        if st.button("🚀 Run Batch Recommendations"):
            results = []
            progress = st.progress(0)
            total = len(df_queries)

            for i, row in enumerate(df_queries.itertuples(), 1):
                q = str(row.Query)
                try:
                    recs = get_recommendations(q, k=5)
                    urls = [r["url"] for r in recs] if recs else ["No recommendation"]
                    results.append({"Query": q, "Assessment_url": ", ".join(urls)})
                except Exception as e:
                    results.append({"Query": q, "Assessment_url": f"Error: {e}"})

                progress.progress(i / total)

            out_df = pd.DataFrame(results)
            st.success("✅ Batch processing complete!")
            st.dataframe(out_df, use_container_width=True)

            csv_data = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )
