import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BERT (lightweight)
from transformers import AutoTokenizer, AutoModel
import torch

# -----------------------------
# LOAD BERT MODEL (LIGHT)
# -----------------------------
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_bert()

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
# TF-IDF SCORE
# -----------------------------
def tfidf_score(resume, job):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# -----------------------------
# BERT SCORE
# -----------------------------
def bert_score(text1, text2):
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        emb1 = bert_model(**inputs1).last_hidden_state.mean(dim=1)
        emb2 = bert_model(**inputs2).last_hidden_state.mean(dim=1)

    return cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]

# -----------------------------
# PDF EXTRACTION
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
st.set_page_config(page_title="AI Resume Screening Pro", layout="wide")

st.title("🚀 AI Resume Screening System (Pro Version)")

job_desc = st.text_area("📌 Enter Job Description")

files = st.file_uploader(
    "📤 Upload Resumes (PDF)",
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

        tfidf = tfidf_score(text, job_desc)
        bert = bert_score(text, job_desc)

        final_score = (0.5 * tfidf) + (0.5 * bert)

        results.append({
            "Resume": file.name,
            "Final Score (%)": round(final_score * 100, 2),
            "TF-IDF (%)": round(tfidf * 100, 2),
            "BERT (%)": round(bert * 100, 2),
            "Missing Skills": ", ".join(missing) if missing else "None"
        })

    df = pd.DataFrame(results).sort_values(by="Final Score (%)", ascending=False)

    st.subheader("📊 Candidate Ranking Dashboard")
    st.dataframe(df, use_container_width=True)

else:
    st.info("Upload resumes and enter job description.")
