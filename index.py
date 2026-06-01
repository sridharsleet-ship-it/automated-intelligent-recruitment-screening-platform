import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in SKILL_DB if skill in text]))

# -----------------------------
# TEXT SIMILARITY
# -----------------------------
def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("📄 AI Resume Screening System")

job_desc = st.text_area("Enter Job Description")

files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# PROCESS
# -----------------------------
if files and job_desc:

    results = []

    for file in files:
        text = extract_text(file)

        resume_skills = extract_skills(text)
        job_skills = extract_skills(job_desc)

        missing = list(set(job_skills) - set(resume_skills))

        score = compute_similarity(text, job_desc)

        results.append({
            "Resume": file.name,
            "Match %": round(score * 100, 2),
            "Missing Skills": ", ".join(missing) if missing else "None"
        })

    df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)

    st.subheader("📊 Ranking Table")
    st.dataframe(df, use_container_width=True)

else:
    st.info("Upload resumes and enter job description.")
