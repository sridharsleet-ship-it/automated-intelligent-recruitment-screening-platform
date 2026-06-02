import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI Resume Screener")

# -----------------------------
# LOAD MODEL (NO SPACY ✅)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# SKILLS
# -----------------------------
SKILL_DB = [
    "python","java","c","c++",
    "machine learning","deep learning",
    "nlp","sql","tensorflow",
    "pandas","numpy","streamlit",
    "data analysis","ai","ml"
]

# -----------------------------
# INPUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    job_desc = st.text_area("📄 Job Description", height=150)

with col2:
    uploaded_files = st.file_uploader(
        "📤 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

# -----------------------------
# FUNCTIONS
# -----------------------------
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
        return 0 if np.isnan(score) else score
    except:
        return 0

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("🚀 Run Screening"):

    if not uploaded_files or not job_desc.strip():
        st.warning("Upload resumes and enter job description")
        st.stop()

    results = []
    previews = {}
    full_texts = {}

    job_emb = model.encode(job_desc)

    progress = st.progress(0)
    status = st.empty()

    for i, file in enumerate(uploaded_files):

        status.text(f"Processing {file.name}...")
        progress.progress((i + 1) / len(uploaded_files))

        resume_text = extract_text(file)

        if not resume_text.strip():
            continue

        previews[file.name] = resume_text[:300].replace("\n", " ")
        full_texts[file.name] = resume_text

        # BERT
        resume_emb = model.encode(resume_text)
        bert = cosine_similarity([resume_emb], [job_emb])[0][0]
        if np.isnan(bert):
            bert = 0

        # TF-IDF
        tfidf = tfidf_score(resume_text, job_desc)

        # Skills
        missing = [
            s for s in SKILL_DB
            if s in job_desc.lower() and s not in resume_text.lower()
        ]

        required = [s for s in SKILL_DB if s in job_desc.lower()]
        match_percent = ((len(required) - len(missing)) / len(required) * 100) if required else 0

        results.append({
            "Candidate": file.name,
            "BERT Score (%)": round(bert * 100, 2),
            "TF-IDF (%)": round(tfidf * 100, 2),
            "Match (%)": round(match_percent, 2),
            "Missing Skills": ", ".join(missing) if missing else "None",
            "Preview": previews[file.name]
        })

    progress.empty()
    status.empty()

    df = pd.DataFrame(results).sort_values(by="BERT Score (%)", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # ✅ SAVE DATA (FIXES PROFILE BUG)
    st.session_state.df = df
    st.session_state.previews = previews
    st.session_state.full_texts = full_texts


# -----------------------------
# DISPLAY (PERSISTENT)
# -----------------------------
if "df" in st.session_state:

    df = st.session_state.df
    previews = st.session_state.previews
    full_texts = st.session_state.full_texts

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader("🏆 Ranking Table")
    st.dataframe(df.drop(columns=["Preview"]), use_container_width=True)

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.subheader("📊 Dashboard")

    scores = df["BERT Score (%)"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", len(df))
    c2.metric("Top Score", f"{scores.max()}%")
    c3.metric("Average", f"{round(scores.mean(),2)}%")

    st.bar_chart(df.set_index("Candidate")["BERT Score (%)"])

    # -----------------------------
    # ✅ FIXED CANDIDATE PROFILE
    # -----------------------------
    st.subheader("👤 Candidate Profile")

    selected = st.selectbox("Select Candidate", df["Candidate"])

    row = df[df["Candidate"] == selected].iloc[0]

    st.markdown(f"### {selected}")
    st.write(f"**BERT Score:** {row['BERT Score (%)']}%")
    st.write(f"**TF-IDF:** {row['TF-IDF (%)']}%")
    st.write(f"**Match:** {row['Match (%)']}%")
    st.write(f"**Missing Skills:** {row['Missing Skills']}")

    with st.expander("📄 Resume Preview"):
        st.write(previews[selected] + "...")

    with st.expander("📜 Full Resume"):
        st.write(full_texts[selected])

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    csv = df.drop(columns=["Preview"]).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Results",
        data=csv,
        file_name="resume_screening_results.csv",
        mime="text/csv"
    )
