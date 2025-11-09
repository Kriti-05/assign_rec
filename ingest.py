# ingest_assessments.py
import os
import re
import pandas as pd
import requests
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from utils import embed_text_local, groq_predict_job_roles, groq_classify_test_types
import pytesseract
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "shl-assessments"

# ---------- INIT PINECONE ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"🆕 Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
print(f"✅ Connected to index: {INDEX_NAME}")

# ---------- LOAD CSV ----------
df = pd.read_csv("train.csv")

if "Query" not in df.columns or "Assessment_url" not in df.columns:
    raise ValueError("❌ train.csv must have columns: 'Query' and 'Assessment_url'")

# ---------- HELPER FUNCTIONS ----------

def extract_test_type_images(soup):
    """Extract test type letters from images using OCR."""
    section = soup.find(string=re.compile("Test Type", re.I))
    if not section:
        return []
    # Usually images are in the next div
    div = section.find_next("div")
    if not div:
        return []
    imgs = div.find_all("img")
    codes = []
    for img in imgs:
        src = img.get("src")
        if src:
            try:
                r = requests.get(src, timeout=5)
                im = Image.open(BytesIO(r.content))
                text = pytesseract.image_to_string(im, config="--psm 8").strip().upper()
                codes.extend(re.findall(r"[ABCDKEPS]", text))
            except:
                continue
    return list(set(codes))

def extract_remote_support(soup):
    """Detect remote support from green icon."""
    section = soup.find(string=re.compile("Remote Testing", re.I))
    if not section:
        return "Unknown"
    icon = section.find_next("img")
    if icon:
        src = icon.get("src", "")
        if "green" in src.lower():
            return "Yes"
        else:
            return "No"
    return "Unknown"

def fetch_assessment_metadata(url):
    """Scrape SHL assessment page and return structured metadata."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # 1️⃣ Name / Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown SHL Assessment"

        # 2️⃣ Description
        desc_header = soup.find(string=re.compile("Description", re.I))
        description = desc_header.find_next("p").get_text(strip=True) if desc_header else "No description"

        # 3️⃣ Job roles
        job_header = soup.find(string=re.compile("Job levels", re.I))
        if job_header:
            job_text = job_header.find_next().get_text(strip=True)
            job_roles = [j.strip() for j in job_text.split(",") if j.strip()]
        else:
            job_roles = ["General Roles"]

        # 4️⃣ Languages
        lang_header = soup.find(string=re.compile("Languages", re.I))
        if lang_header:
            lang_text = lang_header.find_next().get_text(strip=True)
            languages = [l.strip() for l in lang_text.split(",") if l.strip()]
        else:
            languages = ["English"]

        # 5️⃣ Duration
        dur_header = soup.find(string=re.compile("Approximate Completion Time", re.I))
        if dur_header:
            dur_match = re.search(r"(\d+)", dur_header)
            duration = int(dur_match.group(1)) if dur_match else 0
        else:
            duration = 0

        # 6️⃣ Remote support
        remote_support = extract_remote_support(soup)

        # 7️⃣ Test type
        test_type_codes = extract_test_type_images(soup)
        if not test_type_codes:
            test_type_codes = groq_classify_test_types(title, description)

        # Fallback job roles using LLM
        if not job_roles:
            job_roles = groq_predict_job_roles(title, description)

        return {
            "url": url,
            "name": title,
            "description": description,
            "duration": duration,
            "adaptive_support": "Unknown",
            "remote_support": remote_support,
            "test_type": test_type_codes,
            "job_roles": job_roles,
            "languages": languages
        }

    except Exception as e:
        print(f"⚠️ Failed to fetch metadata for {url}: {e}")
        return {
            "url": url,
            "name": "Generic SHL Assessment",
            "description": "General-purpose SHL test for professional roles.",
            "duration": 0,
            "adaptive_support": "Unknown",
            "remote_support": "Unknown",
            "test_type": ["Unknown"],
            "job_roles": ["General Roles"],
            "languages": ["English"]
        }

# ---------- UPSERT INTO PINECONE ----------
records = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    query = str(row["Query"]).strip()
    url = str(row["Assessment_url"]).strip()

    meta = fetch_assessment_metadata(url)
    embedding = embed_text_local([query])[0]

    records.append({
        "id": f"{i}",
        "values": embedding,
        "metadata": {
            "query": query,
            **meta
        }
    })

index.upsert(vectors=records)
print(f"✅ Upserted {len(records)} records into Pinecone '{INDEX_NAME}'")
