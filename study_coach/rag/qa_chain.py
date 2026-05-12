"""
rag/qa_chain.py
================
RAG-powered Q&A using Groq (free API).
Detects topic from question for memory tracking.
"""

import os
import re
from typing import List
from groq import Groq
from embeddings.chroma_store import retrieve

SYSTEM_PROMPT = """You are a friendly, knowledgeable study coach helping a student understand their course material.

Rules:
1. Answer ONLY using the provided context from the student's notes.
2. If the answer isn't in the context, say "I don't see this in your notes — try uploading more material."
3. Use clear, simple language. Break down complex concepts.
4. Give examples where helpful.
5. Keep answers focused and under 200 words unless the question requires more detail.
6. End with a one-line study tip related to the topic."""

TOPIC_KEYWORDS = {
    "supervised learning":   ["supervised", "labeled", "regression", "classification"],
    "unsupervised learning":  ["unsupervised", "clustering", "k-means", "pca"],
    "neural networks":        ["neural", "deep learning", "cnn", "rnn", "layer", "backprop"],
    "overfitting":            ["overfit", "regularization", "dropout", "generalization"],
    "model evaluation":       ["accuracy", "precision", "recall", "f1", "roc", "confusion matrix"],
    "reinforcement learning": ["reinforcement", "reward", "agent", "policy"],
    "feature engineering":    ["feature", "engineering", "selection", "extraction"],
    "bias variance":          ["bias", "variance", "tradeoff", "underfitting"],
    "general ml":             ["machine learning", "algorithm", "model", "train", "test"],
}


def detect_topic(question: str) -> str:
    """Detect the topic of a question for memory tracking."""
    q_lower = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return topic
    return "general"


def answer_question(question: str, user_id: str = "student",
                    model: str = "llama-3.3-70b-versatile",
                    top_k: int = 4) -> dict:
    """
    Answer a study question using RAG + Groq.

    Returns: { answer, sources, topic, error }
    """
    result = {"answer": "", "sources": [], "topic": "", "error": None}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        result["error"] = "Groq API key not set. Get one free at groq.com"
        return result

    # Retrieve relevant chunks
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        result["answer"] = "I don't see any relevant material in your notes. Try uploading more content!"
        result["topic"] = detect_topic(question)
        return result

    # Build context
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Excerpt {i+1} from '{c['doc_id']}' — relevance {c['score']:.0%}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Student's Notes:
{context}

---
Student Question: {question}

Answer clearly and helpfully based on the notes above."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.3
        )
        result["answer"]  = response.choices[0].message.content.strip()
        result["sources"] = chunks
        result["topic"]   = detect_topic(question)

    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "auth" in err.lower():
            result["error"] = "Invalid Groq API key."
        elif "rate" in err.lower():
            result["error"] = "Rate limit — wait a moment and retry."
        else:
            result["error"] = f"Groq error: {err}"

    return result
