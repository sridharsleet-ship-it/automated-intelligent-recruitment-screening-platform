import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Resume AI", layout="centered")

st.title("🤖 Resume Screening AI")
st.markdown("Upload resumes and match them with a job description")

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------
# INPUT
# -----------------------
job_desc = st.text_area("📄 Job Description")

files = st.file_uploader(
    "📤 Upload PDF Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------
# PDF TEXT EXTRACT
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

# -----------------------
# RUN BUTTON
# -----------------------
if st.button("🚀 Analyze"):

    if not job_desc or not files:
        st.warning("Please add job description and resumes")
        st.stop()

    results = []

    job_emb = model.encode(job_desc)

    for file in files:
        text = extract_text(file)

        if not text.strip():
            st.warning(f"{file.name} has no readable text")
            continue

        text = text[:2000]

        emb = model.encode(text)
        score = cosine_similarity([emb], [job_emb])[0][0]

        if np.isnan(score):
            score = 0

        results.append({
            "Name": file.name,
            "Score": round(score * 100, 2)
        })

    if not results:
        st.error("No valid resumes")
        st.stop()

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

    # -----------------------
    # RESULTS UI
    # -----------------------
    st.subheader("🏆 Ranking")

    for i, row in df.iterrows():
        st.markdown(f"""
        **{row['Name']}**  
        Score: `{row['Score']}%`
        """)
        st.progress(row["Score"] / 100)

    # -----------------------
    # TOP RESULT
    # -----------------------
    top = df.iloc[0]
    st.success(f"Best Match: {top['Name']} ({top['Score']}%)")

    # -----------------------
    # DOWNLOAD
    # -----------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="results.csv",
        mime="text/csv"
    )
