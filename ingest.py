import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150     # characters

def read_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    """Return list of {text, page, source} per page."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text.strip():
            pages.append({"text": text, "page": i + 1, "source": pdf_path.name})
    return pages

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def build_corpus() -> List[Dict[str, Any]]:
    corpus = []
    for pdf in DATA_DIR.glob("*.pdf"):
        for page in read_pdf_text(pdf):
            for chunk in chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
                corpus.append({
                    "chunk": chunk,
                    "page": page["page"],
                    "source": page["source"],
                })
    return corpus

def main():
    corpus = build_corpus()
    if not corpus:
        raise RuntimeError("No text extracted. Check PDFs in /data.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["chunk"] for c in corpus]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine via normalized inner product
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"✅ Indexed {len(corpus)} chunks from {len(list(DATA_DIR.glob('*.pdf')))} PDFs")

if __name__ == "__main__":
    main()
