"""
api/main.py
===========
FastAPI REST API for the Study Coach.
Run: uvicorn api.main:app --reload

Endpoints:
  POST /ingest          — index study materials
  POST /ask             — answer a question
  POST /quiz/generate   — generate quiz questions
  POST /quiz/submit     — submit quiz answers + update memory
  GET  /progress/{uid}  — get student progress
  DELETE /progress/{uid} — reset student progress
"""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from ingestion.pdf_loader import load_pdf, load_text
from embeddings.chroma_store import index_chunks, retrieve, get_stats
from rag.qa_chain import answer_question
from quiz.quiz_generator import generate_quiz
from memory.session_memory import (
    update_topic_score, get_session_summary,
    get_weak_areas, reset_user, export_progress
)

app = FastAPI(
    title="AI Study Coach API",
    description="Personalized RAG study assistant with voice, quiz, and memory",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Models ─────────────────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    doc_id: str
    text: str

class AskRequest(BaseModel):
    question: str
    user_id: str = "student"
    top_k: int = 4
    model: str = "llama-3.3-70b-versatile"

class QuizRequest(BaseModel):
    user_id: str = "student"
    topic: Optional[str] = None
    num_questions: int = 5
    model: str = "llama-3.3-70b-versatile"
    auto_weak_areas: bool = True

class QuizSubmitRequest(BaseModel):
    user_id: str
    topic: str
    answers: List[dict]  # [{ question_index, user_answer, correct_answer }]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    stats = get_stats()
    return {"service": "AI Study Coach API", "chunks_indexed": stats["count"], "docs": "/docs"}


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text is empty.")
    chunks = load_text(req.text, doc_id=req.doc_id)
    count  = index_chunks(chunks)
    return {"doc_id": req.doc_id, "chunks_indexed": count}


@app.post("/ingest/pdf")
async def ingest_pdf(
    doc_id: str,
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        chunks = load_pdf(tmp_path, doc_id=doc_id)
        count  = index_chunks(chunks)
        return {"doc_id": doc_id, "chunks_indexed": count}
    finally:
        os.unlink(tmp_path)


@app.post("/ask")
def ask(req: AskRequest, x_groq_api_key: Optional[str] = Header(None)):
    if x_groq_api_key:
        os.environ["GROQ_API_KEY"] = x_groq_api_key
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(401, "Groq API key required.")

    result = answer_question(
        question=req.question,
        user_id=req.user_id,
        model=req.model,
        top_k=req.top_k
    )
    if result.get("error"):
        raise HTTPException(500, result["error"])
    return result


@app.post("/quiz/generate")
def generate_quiz_endpoint(
    req: QuizRequest,
    x_groq_api_key: Optional[str] = Header(None)
):
    if x_groq_api_key:
        os.environ["GROQ_API_KEY"] = x_groq_api_key
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(401, "Groq API key required.")

    topic = req.topic
    if not topic and req.auto_weak_areas:
        weak = get_weak_areas(req.user_id)
        topic = weak[0] if weak else None

    result = generate_quiz(topic=topic, num_questions=req.num_questions, model=req.model)
    if result.get("error"):
        raise HTTPException(500, result["error"])
    return result


@app.post("/quiz/submit")
def submit_quiz(req: QuizSubmitRequest):
    results = []
    for ans in req.answers:
        correct = str(ans.get("user_answer", "")).strip().lower() == \
                  str(ans.get("correct_answer", "")).strip().lower()
        update_topic_score(req.user_id, req.topic, correct)
        results.append({**ans, "is_correct": correct})

    return {
        "results": results,
        "score": sum(1 for r in results if r["is_correct"]),
        "total": len(results),
        "progress": get_session_summary(req.user_id)
    }


@app.get("/progress/{user_id}")
def get_progress(user_id: str):
    return get_session_summary(user_id)


@app.get("/progress/{user_id}/export")
def export_progress_endpoint(user_id: str):
    return {"data": export_progress(user_id)}


@app.delete("/progress/{user_id}")
def delete_progress(user_id: str):
    reset_user(user_id)
    return {"message": f"Progress reset for {user_id}"}
