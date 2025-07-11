import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from extract import extract_text_from_pdf, chunk_text

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(pdf_paths):
    all_chunks = []
    sources = []

    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        sources.extend([os.path.basename(path)] * len(chunks))

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")

    # Save chunks + source mapping
    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump((all_chunks, sources), f)

if __name__ == "__main__":
    pdfs = ["data/resume.pdf", "data/job_description.pdf"]
    create_faiss_index(pdfs)
