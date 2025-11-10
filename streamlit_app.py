import streamlit as st
import pandas as pd
import requests
import threading
import time
from backend import app  # Import your Flask app

# ---------------- RUN FLASK IN BACKGROUND ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# Start Flask in background thread when Streamlit starts
if 'flask_started' not in st.session_state:
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    st.session_state.flask_started = True
    time.sleep(2)  # Give Flask time to start

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="SHL HR Recommender", page_icon="🧠", layout="wide")
st.title("🧠 SHL HR Recommender Demo")

# Use local Flask server
API_URL = "http://localhost:5000"

# ---------------- HEALTH CHECK ----------------
st.header("🔧 API Health Check")

if st.button("Check API Health"):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            st.success(f"✅ API is healthy: {resp.json()}")
        else:
            st.error(f"❌ Health check failed: {resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"⚠️ Could not reach API: {e}")

st.divider()

# ---------------- SINGLE QUERY MODE ----------------
st.header("🎯 Single Query Mode")
query = st.text_area("Enter job description or hiring query:")

if st.button("🔍 Get Recommendations for Single Query"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                resp = requests.post(f"{API_URL}/recommend", json={"query": query})
                if resp.status_code != 200:
                    st.error(f"❌ Error {resp.status_code}: {resp.text}")
                else:
                    data = resp.json()
                    recs = data.get("recommended_assessments", [])
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
                    r = requests.post(f"{API_URL}/recommend", json={"query": q}, timeout=10).json()
                    links = [x["url"] for x in r.get("recommended_assessments", [])]
                    if links:
                        for url in links:
                            results.append({"Query": q, "Assessment_url": url})
                    else:
                        results.append({"Query": q, "Assessment_url": "No recommendation"})
                except Exception as e:
                    results.append({"Query": q, "Assessment_url": f"Error: {e}"})

                progress.progress(i / total)

            # --- Create submission CSV ---
            out_df = pd.DataFrame(results)
            st.success("✅ Batch processing complete! Here's your formatted submission:")
            st.dataframe(out_df, use_container_width=True)

            st.markdown("""
            ### ✅ Submission CSV Format (Appendix 3)
            The file contains exactly two columns:
            - **Query**
            - **Assessment_url**
            """)

            # --- Download button ---
            csv_data = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )
