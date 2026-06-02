import streamlit as st
import pdfplumber
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.markdown("## 🚀 Automated Intelligent Recruitment Screening Platform")
st.markdown("Upload resumes and get AI-powered ranking instantly.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")  # faster model

model = load_model()

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
    job_desc = st.text_area("📌 Job Description", height=150)

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
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def tfidf_score(resume, job):
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([resume, job])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# -----------------------------
# PROCESS
# -----------------------------
if st.button("🚀 Run Screening"):

    if not uploaded_files or not job_desc.strip():
        st.warning("⚠️ Upload resumes and enter job description")
        st.stop()

    # Limit resumes for performance
    if len(uploaded_files) > 15:
        st.warning("⚠️ Max 15 resumes allowed")
        st.stop()

    results = []
    previews = {}

    job_emb = model.encode(job_desc)

    # -----------------------------
    # PROCESS FILES
    # -----------------------------
    for file in uploaded_files:
        try:
            resume_text = extract_text(file)

            # Trim for speed
            resume_text = resume_text[:2000]

            previews[file.name] = resume_text[:400].replace("\n", " ")

            # BERT
            resume_emb = model.encode(resume_text)
            bert = cosine_similarity([resume_emb], [job_emb])[0][0]

            # TF-IDF
            tfidf = tfidf_score(resume_text, job_desc)

            # Missing skills
            missing = [
                s for s in SKILL_DB
                if s in job_desc.lower() and s not in resume_text.lower()
            ]

            # Match %
            required = [s for s in SKILL_DB if s in job_desc.lower()]
            match_percent = ((len(required) - len(missing)) / len(required) * 100) if required else 0

            # Feedback
            if bert > 0.8:
                feedback = "🟢 Strong match for the role"
            elif bert > 0.6:
                feedback = "🟡 Good match, but can improve"
            else:
                feedback = "🔴 Low match, needs improvement"

            results.append({
                "Candidate": file.name,
                "BERT Score (%)": round(bert * 100, 2),
                "TF-IDF (%)": round(tfidf * 100, 2),
                "Match (%)": round(match_percent, 2),
                "Missing Skills": ", ".join(missing) if missing else "None",
                "Feedback": feedback
            })

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    # -----------------------------
    # DATAFRAME
    # -----------------------------
    df = pd.DataFrame(results).sort_values(by="BERT Score (%)", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # -----------------------------
    # FILTER
    # -----------------------------
    st.subheader("🎯 Filter Results")

    min_score = st.slider("Minimum BERT Score", 0, 100, 50)
    df = df[df["BERT Score (%)"] >= min_score]

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.subheader("📊 Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", len(df))
    c2.metric("Top Score", f"{df['BERT Score (%)'].max()}%")
    c3.metric("Average Score", f"{round(df['BERT Score (%)'].mean(),2)}%")

    # -----------------------------
    # TOP CANDIDATE
    # -----------------------------
    if not df.empty:
        top_candidate = df.iloc[0]
        st.success(f"🏆 Top Candidate: {top_candidate['Candidate']} ({top_candidate['BERT Score (%)']}%)")

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader("🏆 Ranked Candidates")

    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # CHART
    # -----------------------------
    st.subheader("📈 Score Distribution")

    chart_df = df[["Candidate", "BERT Score (%)"]].set_index("Candidate")
    st.bar_chart(chart_df)

    # -----------------------------
    # SKILL GAPS
    # -----------------------------
    st.subheader("⚠️ Skill Gaps")

    for r in results:
        if r["Missing Skills"] != "None":
            st.warning(f"{r['Candidate']}: Missing {r['Missing Skills']}")

    # -----------------------------
    # RESUME PREVIEW
    # -----------------------------
    st.subheader("📄 Resume Preview")

    for name, text in previews.items():
        with st.expander(f"View {name}"):
            st.write(text + "...")

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    st.subheader("⬇️ Download Results")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="resume_results.csv",
        mime="text/csv"
    )
    
