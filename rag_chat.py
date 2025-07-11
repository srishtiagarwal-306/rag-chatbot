import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer


class RAGChatBot:
    def __init__(self, index, chunks, api_key, model_name="gemini-1.5-flash", k=5):
        self.index = index
        self.chunks = chunks
        self.k = k
        self.model_name = model_name

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

        # Load embedding model once
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def ask(self, query):
        if not query:
            return "❌ Query is empty."

        query_embedding = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_embedding), self.k)
        retrieved_chunks = [self.chunks[i] for i in I[0]]

        context = "\n".join(retrieved_chunks)
        if len(context) > 3000:
            context = context[:3000]  # truncate if too long

        prompt = f"""You are a helpful assistant. Based on the following documents, answer the question:

\"\"\"
{context}
\"\"\"

Question: {query}
Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"❌ Gemini error: {str(e)}"
