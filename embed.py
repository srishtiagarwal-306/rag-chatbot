import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from extract import extract_text_from_pdf, chunk_text

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(resume_path, jd_path):
    all_chunks = []
    sources = []

    for path in [resume_path, jd_path]:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        sources.extend([path] * len(chunks))

    embeddings = model.encode(all_chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, all_chunks
