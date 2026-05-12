"""
embeddings/chroma_store.py
===========================
ChromaDB vector store with sentence-transformers embeddings.
Both run 100% offline — no API keys needed.

Install: pip install chromadb sentence-transformers
"""

import os
from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION   = "study_materials"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # free, ~80MB, offline

_embedder   = None
_client     = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def index_chunks(chunks: List[dict]) -> int:
    """Embed and upsert chunks into ChromaDB. Returns count indexed."""
    if not chunks:
        return 0
    col = _get_collection()
    emb = _get_embedder()

    texts  = [c["text"] for c in chunks]
    ids    = [c["id"]   for c in chunks]
    metas  = [{"doc_id": c["doc_id"], "chunk_id": str(c["chunk_id"])} for c in chunks]
    embeds = emb.encode(texts, show_progress_bar=False).tolist()

    col.upsert(ids=ids, documents=texts, embeddings=embeds, metadatas=metas)
    return len(chunks)


def retrieve(query: str, top_k: int = 4) -> List[dict]:
    """Semantic search. Returns list of { text, doc_id, chunk_id, score }."""
    col = _get_collection()
    emb = _get_embedder()

    if col.count() == 0:
        return []

    q_emb = emb.encode([query]).tolist()
    res   = col.query(
        query_embeddings=q_emb,
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"]
    )

    results = []
    for i in range(len(res["documents"][0])):
        results.append({
            "text":     res["documents"][0][i],
            "doc_id":   res["metadatas"][0][i].get("doc_id", "unknown"),
            "chunk_id": res["metadatas"][0][i].get("chunk_id", "0"),
            "score":    round(1 - res["distances"][0][i], 4)
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def get_stats() -> dict:
    try:
        return {"count": _get_collection().count()}
    except Exception:
        return {"count": 0}


def clear_collection():
    global _collection
    try:
        col = _get_collection()
        ids = col.get()["ids"]
        if ids:
            col.delete(ids=ids)
        _collection = None
    except Exception:
        pass
