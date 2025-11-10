import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SHL HR Recommender", page_icon="🧠", layout="wide")
st.title("🧠 SHL HR Recommender Demo")

# ---------------- CONFIG ----------------
API_URL = st.secrets.get("API_URL", "http://localhost:5000")

# ---------------- HEALTH CHECK ----------------
st.header("🔧 API Health Check")
if st.button("Check API Health"):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            st.success(f"✅ API is healthy: {resp.json()}")
        else:
            st.error(f"❌ Health check failed: {resp.status_code}")
    except Exception as e:
        st.error(f"⚠️ Could not reach API: {e}")

st.divider()

# ---------------- SINGLE QUERY ----------------
st.header("🎯 Single Query Mode")
query = st.text_area("Enter job description or hiring query:")
if st.button("🔍 Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                payload = {"query": query, "k": 5}
                resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    recs = data.get("recommended_assessments", [])
                    st.success(f"✅ Got {len(recs)} recommendations")
                    if recs:
                        df = pd.DataFrame(recs)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No recommendations found.")
                else:
                    st.error(f"❌ Error: {resp.text}")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

st.divider()

# ---------------- BATCH CSV ----------------
st.header("📂 Batch Mode: Upload CSV")
uploaded = st.file_uploader("Upload CSV (must have 'Query' column)", type=["csv"])
if uploaded:
    df_queries = pd.read_csv(uploaded)
    if "Query" not in df_queries.columns:
        st.error("CSV must contain a 'Query' column")
    else:
        results = []
        for row in df_queries.itertuples():
            q = str(row.Query)
            try:
                r = requests.post(f"{API_URL}/recommend", json={"query": q, "k": 5}, timeout=10).json()
                links = [x["url"] for x in r.get("recommended_assessments", [])]
                results.append({"Query": q, "Assessment_url": ", ".join(links) if links else "No recommendation"})
            except:
                results.append({"Query": q, "Assessment_url": "Error"})
        out_df = pd.DataFrame(results)
        st.dataframe(out_df, use_container_width=True)
        csv_data = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download submission.csv", data=csv_data, file_name="submission.csv", mime="text/csv")
