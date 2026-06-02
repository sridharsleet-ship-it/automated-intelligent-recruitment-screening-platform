import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------
# PAGE
# -----------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI Resume Screener")

# -----------------------
# MODEL
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
# FEEDBACK ENGINE (NEW 🔥)
# -----------------------
def generate_feedback(match, missing):
    if match > 80:
        return "Strong candidate with high alignment to job requirements."
    elif match > 60:
        return "Good match but can improve in some areas."
    else:
        return "Low match. Candidate lacks several key skills."

def ai_reviewer(resume_text):
    resume_text = resume_text.lower()

    if "project" in resume_text and "experience" in resume_text:
        return "Well-structured resume with project and experience sections."
    elif "project" in resume_text:
        return "Good project work, but add more experience details."
    else:
        return "Resume lacks strong project/experience sections."

# -----------------------
# RUN
# -----------------------
if st.button("🚀 Run Screening"):

    if not job_desc or not files:
        st.warning("Add job description + resumes")
        st.stop()

    results = []
    previews = {}
    full_texts = {}

    job_emb = model.encode(job_desc)

    # LOADING UX 🔥
    progress = st.progress(0)
    status = st.empty()

    for i, file in enumerate(files):

        status.text(f"Processing {file.name}...")
        progress.progress((i+1)/len(files))

        text = extract_text(file)

        if not text.strip():
            continue

        text = text[:2000]
        previews[file.name] = text[:300].replace("\n", " ")
        full_texts[file.name] = text

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

        # FINAL SCORE
        final_score = (bert*100*0.6) + (tfidf*100*0.2) + (match*0.2)

        # FEEDBACK
        feedback = generate_feedback(match, missing)

        # AI REVIEW
        review = ai_reviewer(text)

        results.append({
            "Candidate": file.name,
            "BERT (%)": round(bert*100,2),
            "TF-IDF (%)": round(tfidf*100,2),
            "Match (%)": round(match,2),
            "Final Score": round(final_score,2),
            "Missing Skills": ", ".join(missing) if missing else "None",
            "Feedback": feedback,
            "AI Review": review
        })

        time.sleep(0.2)

    progress.empty()
    status.empty()

    df = pd.DataFrame(results).sort_values(by="Final Score", ascending=False)
    df.insert(0, "Rank", range(1, len(df)+1))

    # -----------------------
    # TABLE
    # -----------------------
    st.subheader("🏆 Ranking Table")
    st.dataframe(df, use_container_width=True)

    # -----------------------
    # DASHBOARD
    # -----------------------
    st.subheader("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", len(df))
    c2.metric("Top Score", f"{df['Final Score'].max()}%")
    c3.metric("Average", f"{round(df['Final Score'].mean(),2)}%")

    st.bar_chart(df.set_index("Candidate")["Final Score"])

    # -----------------------
    # PROFILE PAGE 🔥 (15)
    # -----------------------
    st.subheader("👤 Candidate Profiles")

    selected = st.selectbox("Select Candidate", df["Candidate"])

    row = df[df["Candidate"] == selected].iloc[0]

    st.markdown(f"### {selected}")
    st.write(f"**Final Score:** {row['Final Score']}%")
    st.write(f"**Feedback:** {row['Feedback']}")
    st.write(f"**AI Review:** {row['AI Review']}")
    st.write(f"**Missing Skills:** {row['Missing Skills']}")

    with st.expander("📄 Resume Preview"):
        st.write(previews[selected] + "...")

    with st.expander("📜 Full Resume Text"):
        st.write(full_texts[selected])

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
