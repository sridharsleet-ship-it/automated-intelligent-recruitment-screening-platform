import streamlit as st
import pdfplumber
import pandas as pd
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("AI Resume Screener")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    return spacy.load("en_core_web_sm"), SentenceTransformer("all-MiniLM-L6-v2")

nlp, model = load_models()

# -----------------------------
# SKILL DATABASE
# -----------------------------
SKILL_DB = [
    "python", "java", "c", "c++",
    "machine learning", "deep learning",
    "nlp", "sql", "tensorflow",
    "pandas", "numpy", "streamlit",
    "data analysis", "ai", "ml"
]

# -----------------------------
# INPUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    job_desc = st.text_area("Job Description", height=150)

with col2:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def tfidf_score(resume, job):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([resume, job])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# -----------------------------
# PROCESS
# -----------------------------
if st.button("Run Screening"):

    if not uploaded_files or not job_desc.strip():
        st.warning("Upload resumes and enter job description")
        st.stop()

    results = []
    job_emb = model.encode(job_desc)

    # -----------------------------
    # PROCESS FILES
    # -----------------------------
    for file in uploaded_files:

        resume_text = extract_text(file)

        # Scores
        resume_emb = model.encode(resume_text)
        bert = cosine_similarity([resume_emb], [job_emb])[0][0]
        tfidf = tfidf_score(resume_text, job_desc)

        # Missing skills only
        missing = [
            s for s in SKILL_DB
            if s in job_desc.lower() and s not in resume_text.lower()
        ]

        # Match %
        required = [s for s in SKILL_DB if s in job_desc.lower()]
        match_percent = ((len(required) - len(missing)) / len(required) * 100) if required else 0

        # Store results
        results.append({
            "Candidate": file.name,
            "BERT Score (%)": round(bert*100, 2),
            "TF-IDF (%)": round(tfidf*100, 2),
            "Match (%)": round(match_percent, 2),
            "Missing Skills": ", ".join(missing) if missing else "None",
            "Preview": resume_text[:300].replace("\n", " ")
        })

    # -----------------------------
    # RESUME EXTRACTION PREVIEW
    # -----------------------------
    st.subheader("Resume Extraction")

    for r in results:
        col1, col2 = st.columns([1, 4])
        col1.write(f"**{r['Candidate']}**")
        col2.write(r["Preview"] + "...")

    # -----------------------------
    # TABLE (RANKED)
    # -----------------------------
    df = pd.DataFrame(results).sort_values(by="BERT Score (%)", ascending=False)

    # Add Rank
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Description
    st.subheader("Results")
    st.markdown("""
    Resumes are ranked based on semantic similarity (BERT Score) with the job description.  
    Missing skills indicate gaps between candidate profiles and job requirements.
    """)

    st.dataframe(df.drop(columns=["Preview"]), use_container_width=True)

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.subheader("Overview")

    scores = df["BERT Score (%)"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", len(df))
    c2.metric("Top Score", f"{scores.max()}%")
    c3.metric("Average", f"{round(scores.mean(),2)}%")

    chart_df = df[["Candidate", "BERT Score (%)"]].set_index("Candidate")
    st.bar_chart(chart_df)
    # -----------------------------
# DOWNLOAD RESULTS
# -----------------------------
st.subheader("Download Report")

csv = df.drop(columns=["Preview"]).to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="resume_screening_results.csv",
    mime="text/csv"
)