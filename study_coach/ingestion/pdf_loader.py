"""
ingestion/pdf_loader.py
=======================
Load and chunk PDFs using PyMuPDF (fitz) — free, offline.
Also handles plain text input.

Install: pip install pymupdf
"""

import re
from typing import List

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def load_pdf(pdf_path: str, doc_id: str = "doc",
             chunk_size: int = 250, overlap: int = 40) -> List[dict]:
    """
    Extract text from PDF and return chunks ready for indexing.

    Args:
        pdf_path   : path to PDF file
        doc_id     : document identifier
        chunk_size : words per chunk
        overlap    : words of overlap between chunks

    Returns:
        List of { id, text, doc_id, chunk_id, page, word_count }
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            all_text.append({"text": text.strip(), "page": page_num + 1})

    doc.close()

    if not all_text:
        return []

    # Combine all pages then chunk
    full_text = "\n\n".join(p["text"] for p in all_text)
    return _chunk_text(full_text, doc_id=doc_id,
                       chunk_size=chunk_size, overlap=overlap)


def load_text(text: str, doc_id: str = "notes",
              chunk_size: int = 250, overlap: int = 40) -> List[dict]:
    """
    Chunk plain text into indexable chunks.
    """
    if not text or not text.strip():
        return []
    return _chunk_text(text.strip(), doc_id=doc_id,
                       chunk_size=chunk_size, overlap=overlap)


def _chunk_text(text: str, doc_id: str,
                chunk_size: int, overlap: int) -> List[dict]:
    """
    Sentence-aware overlapping chunker.
    """
    sentences = _split_sentences(text)
    chunks = []
    current = []
    chunk_id = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        current.append(sent)

        if sum(len(s.split()) for s in current) >= chunk_size:
            chunk_text = " ".join(current).strip()
            if len(chunk_text.split()) > 15:
                chunks.append({
                    "id": f"{doc_id}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1

            # Overlap: keep tail sentences
            tail, tail_words = [], 0
            for s in reversed(current):
                sw = len(s.split())
                if tail_words + sw <= overlap:
                    tail.insert(0, s)
                    tail_words += sw
                else:
                    break
            current = tail

    # Final chunk
    if current:
        chunk_text = " ".join(current).strip()
        if len(chunk_text.split()) > 10:
            chunks.append({
                "id": f"{doc_id}_chunk_{chunk_id}",
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "word_count": len(chunk_text.split())
            })

    return chunks


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, protecting common abbreviations."""
    abbrs = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "e.g.", "i.e.",
             "vs.", "Fig.", "et al.", "approx."]
    protected = text
    for i, a in enumerate(abbrs):
        protected = protected.replace(a, f"__A{i}__")

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    result = []
    for p in parts:
        for i, a in enumerate(abbrs):
            p = p.replace(f"__A{i}__", a)
        # Also split on double newlines (section breaks)
        sub = [x.strip() for x in p.split("\n\n") if x.strip()]
        result.extend(sub)
    return result
