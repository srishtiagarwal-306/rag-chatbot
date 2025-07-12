import streamlit as st
import tempfile
from embed import build_faiss_index
from rag_chat import RAGChatBot
st.set_page_config(page_title="📄 Resume-JD RAG Chatbot", layout="centered")

st.title("📄 Resume + JD Chatbot using RAG")
st.markdown("Ask smart questions about your resume & job description using Gemini + FAISS!")

# --- Sidebar: API Key and File Uploads ---
with st.sidebar:
    st.header("🔑 Gemini API Key")

    default_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""
    use_default = st.checkbox("Use built-in Gemini API key", value=True)

    if use_default and default_key:
        api_key = default_key
        st.success("✅ Using built-in API key from secrets")
    else:
        api_key = st.text_input("Paste your Gemini API key", type="password")

    uploaded_resume = st.file_uploader("Upload Resume PDF", type="pdf")
    uploaded_jd = st.file_uploader("Upload Job Description PDF", type="pdf")

# --- Main Logic ---
if api_key and uploaded_resume and uploaded_jd:
    with st.spinner("🔄 Embedding documents and building FAISS index..."):
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
            tmp1.write(uploaded_resume.read())
            resume_path = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
            tmp2.write(uploaded_jd.read())
            jd_path = tmp2.name

        # Build FAISS index from both PDFs
        index, chunks = build_faiss_index(resume_path, jd_path)
        rag_bot = RAGChatBot(index, chunks, api_key)

    st.success("✅ Documents indexed! Ask your question below:")

    # --- Ask a Question ---
    query = st.text_input("🔍 Ask a question about your Resume or JD:")
    if query:
        with st.spinner("💬 Generating answer using Gemini..."):
            answer = rag_bot.ask(query)
        st.markdown("### 🤖 Answer:")
        st.write(answer)
else:
    st.info("📂 Please upload both PDFs to begin.")

