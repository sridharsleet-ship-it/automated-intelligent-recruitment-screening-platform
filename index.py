import streamlit as st
import pdfplumber
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD NLP MODEL
# -----------------------------
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

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
# TEXT SIMILARITY (TF-IDF)
# -----------------------------
def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return score

# -----------------------------
# EXTRACT TEXT FROM PDF
# -----------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("📄 AI Resume Screening System")
st.markdown("Upload resumes and compare them against a job description to rank candidates based on relevance.")

job_desc = st.text_area("📌 Enter Job Description")

uploaded_files = st.file_uploader(
    "📤 Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_files and job_desc.strip():

    results = []

    for file in uploaded_files:
        try:
            resume_text = extract_text_from_pdf(file)

            # Skills
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_desc)

            # Missing Skills
            missing_skills = list(set(job_skills) - set(resume_skills))

            # Similarity Score
            score = compute_similarity(resume_text, job_desc)

            results.append({
                "Resume": file.name,
                "Match %": round(score * 100, 2),
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
            })

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    # -----------------------------
    # RANKING TABLE
    # -----------------------------
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Match %", ascending=False)

        st.subheader("📊 Candidate Ranking")
        st.dataframe(df, use_container_width=True)

    else:
        st.warning("No valid resumes processed.")

else:
    st.info("Please upload resumes and enter a job description.")
