import streamlit as st
import tempfile
from embed import build_faiss_index
from rag_chat import RAGChat

st.set_page_config(page_title="📄 Resume-JD RAG Chatbot", layout="centered")

st.title("📄 Resume + JD Chatbot using RAG")
st.markdown("Ask smart questions about your resume & job description using Gemini + FAISS!")

with st.sidebar:
    st.header("🔑 Gemini API Key")
    api_key = st.text_input("Enter your Google Gemini API key", type="password")

    uploaded_resume = st.file_uploader("Upload Resume PDF", type="pdf")
    uploaded_jd = st.file_uploader("Upload Job Description PDF", type="pdf")

if api_key and uploaded_resume and uploaded_jd:
    with st.spinner("Embedding documents... please wait."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
            tmp1.write(uploaded_resume.read())
            resume_path = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
            tmp2.write(uploaded_jd.read())
            jd_path = tmp2.name

        index, chunks = build_faiss_index(resume_path, jd_path)
        rag_bot = RAGChat(index, chunks, api_key)

    st.success("✅ Documents indexed successfully! You can now ask questions.")
    
    query = st.text_input("Ask a question about your Resume or JD:")
    if query:
        with st.spinner("Thinking..."):
            answer = rag_bot.ask(query)
        st.markdown(f"**🤖 Answer:** {answer}")

else:
    st.info("Please upload both PDFs and enter your Gemini API key to begin.")
