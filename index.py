import streamlit as st

st.title("AI Resume Screener")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success("Resume uploaded successfully!")
    st.write("Now we will extract text from it in next step.")