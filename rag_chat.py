import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGChat:
    def __init__(self, index, chunks, api_key, k=5):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.index = index
        self.chunks = chunks
        self.k = k

    def ask(self, query):
        q_embed = model.encode([query])
        D, I = self.index.search(np.array(q_embed), self.k)
        retrieved = [self.chunks[i] for i in I[0]]
        context = "\n".join(retrieved)

        prompt = f"""You are a helpful assistant. Based on the following documents, answer the question:

\"\"\"
{context}
\"\"\"

Question: {query}
Answer:"""
        response = self.model.generate_content(prompt)
        return response.text.strip()
