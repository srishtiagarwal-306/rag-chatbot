import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGChatbot:
    def __init__(self, api_key, k=5):
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")
        self.k = k
        self.index = faiss.read_index("faiss_index/index.faiss")
        with open("faiss_index/chunks.pkl", "rb") as f:
            self.chunks, self.sources = pickle.load(f)

    def retrieve(self, query):
        q_embed = model.encode([query])
        D, I = self.index.search(np.array(q_embed), self.k)
        return [self.chunks[i] for i in I[0]]

    def generate_answer(self, query, docs):
        context = "\n".join(docs)
        prompt = f"""You are a helpful assistant. Answer the question based only on the following text:

\"\"\"
{context}
\"\"\"

Question: {query}
Answer:"""
        response = self.gemini.generate_content(prompt)
        return response.text.strip()
