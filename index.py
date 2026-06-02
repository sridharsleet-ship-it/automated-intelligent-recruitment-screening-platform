import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------
# PAGE
# -----------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI Resume Screener")

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------
# SKILLS
# -----------------------
SKILL_DB = [
    "python","java","c","c++",
    "machine learning","deep learning",
    "nlp","sql","tensorflow",
    "pandas","numpy","streamlit",
    "data analysis","ai","ml"
]

# -----------------------
# INPUT
# -----------------------
col1, col2 = st.columns(2)

with col1:
    job_desc = st.text_area("📄 Job Description", height=200)

with col2:
    files = st.file_uploader(
        "📤 Upload Resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

# -----------------------
# FUNCTIONS
# -----------------------
def extract_text(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        return ""
    return text


def tfidf_score(resume, job):
    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform([resume, job])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return 0 if pd.isna(score) else score
    except:
        return 0


# -----------------------
# RUN
# -----------------------
if st.button("🚀 Run Screening"):

    if not job_desc or not files:
        st.warning("Add job description + resumes")
        st.stop()

    results = []
    previews = {}

    job_emb = model.encode(job_desc)

    for file in files:
        text = extract_text(file)

        if not text.strip():
            st.warning(f"{file.name} has no readable text")
            continue

        text = text[:2000]

        # STORE PREVIEW
        previews[file.name] = text[:300].replace("\n", " ")

        # BERT
        emb = model.encode(text)
        bert = cosine_similarity([emb], [job_emb])[0][0]
        if np.isnan(bert):
            bert = 0

        # TFIDF
        tfidf = tfidf_score(text, job_desc)

        # SKILLS
        missing = [
            s for s in SKILL_DB
            if s in job_desc.lower() and s not in text.lower()
        ]

        required = [s for s in SKILL_DB if s in job_desc.lower()]
        match = ((len(required) - len(missing)) / len(required) * 100) if required else 0

        results.append({
            "Candidate": file.name,
            "BERT (%)": round(bert * 100, 2),
            "TF-IDF (%)": round(tfidf * 100, 2),
            "Match (%)": round(match, 2),
            "Missing Skills": ", ".join(missing) if missing else "None"
        })

    if not results:
        st.error("No valid resumes")
        st.stop()

    # -----------------------
    # TABLE
    # -----------------------
    df = pd.DataFrame(results).sort_values(by="BERT (%)", ascending=False)
    df.insert(0, "Rank", range(1, len(df)+1))

    st.subheader("🏆 Ranking Table")
    st.dataframe(df, use_container_width=True)

    # -----------------------
    # DASHBOARD
    # -----------------------
    st.subheader("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(df))
    c2.metric("Top Score", f"{df['BERT (%)'].max()}%")
    c3.metric("Average", f"{round(df['BERT (%)'].mean(),2)}%")

    st.bar_chart(df.set_index("Candidate")["BERT (%)"])

    # -----------------------
    # PREVIEW (EXTRACTION)
    # -----------------------
    st.subheader("📄 Resume Extraction")

    for name, text in previews.items():
        with st.expander(name):
            st.write(text + "...")

    # -----------------------
    # DOWNLOAD
    # -----------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Results",
        csv,
        "results.csv",
        "text/csv"
    )
