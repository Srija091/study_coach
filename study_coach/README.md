# 🎓 Personalized AI Study Coach

> Upload notes → Ask questions (text or voice) → Get quizzed → Track mastery per topic.
> **100% free. Runs locally. No cloud costs.**

Built with **PyMuPDF** · **sentence-transformers** · **ChromaDB** · **Groq free API** · **Whisper (local)** · **diskcache**

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

# For voice input (optional):
# macOS:  brew install ffmpeg
# Linux:  sudo apt install ffmpeg
```

### 2. Get a free Groq API key
Sign up at **https://groq.com** — no credit card, takes 60 seconds.

### 3. Run the web app
```bash
streamlit run app.py
```

### 4. Or run the FastAPI backend
```bash
uvicorn api.main:app --reload
# Docs → http://localhost:8000/docs
```

---

## How It Works

```
Your PDF / Notes
      ↓
[PyMuPDF] ── extract text ──→ [Chunker] ── overlapping chunks
                                    ↓
                     [sentence-transformers] ── local embeddings
                                    ↓
                             [ChromaDB] ── persistent vector store
                                    ↓
Student asks question (text or 🎤 voice via Whisper)
                                    ↓
                     [Semantic Retrieval] ── top-k cosine search
                                    ↓
                         [Groq LLM] ── Llama-3.3-70B (free)
                                    ↓
              Answer + Source Citations + Topic Detection
                                    ↓
                    [diskcache Memory] ── mastery tracking
                                    ↓
              Quiz generation → adaptive to weak areas
                                    ↓
              📊 Progress Dashboard ── per-topic mastery bars
```

---

## Features

| Feature | Tool | Cost |
|---|---|---|
| PDF ingestion | PyMuPDF | Free |
| Text embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, offline |
| Vector search | ChromaDB | Free, persistent |
| LLM answers | Groq free API (Llama-3.3-70B) | Free |
| Voice input | OpenAI Whisper (local) | Free, offline |
| Memory / mastery tracking | diskcache | Free, no setup |
| UI | Streamlit | Free |
| REST API | FastAPI | Free |

---

## Project Structure

```
study_coach/
├── app.py                      # Streamlit UI (4 tabs)
├── api/
│   └── main.py                 # FastAPI REST backend
├── ingestion/
│   └── pdf_loader.py           # PyMuPDF PDF + text chunker
├── embeddings/
│   └── chroma_store.py         # ChromaDB + sentence-transformers
├── rag/
│   └── qa_chain.py             # Groq RAG Q&A + topic detection
├── voice/
│   └── transcriber.py          # Local Whisper voice transcription
├── memory/
│   └── session_memory.py       # diskcache per-student mastery tracking
├── quiz/
│   └── quiz_generator.py       # Adaptive MCQ generation via Groq
├── requirements.txt
└── README.md
```

---

## Resume Bullets

```
• Built a full-stack AI study coach with RAG over uploaded PDFs
  (ChromaDB + sentence-transformers), local voice input via Whisper,
  and adaptive quiz generation targeting weak areas — zero API cost
  except Groq free tier.

• Designed persistent session memory with diskcache tracking per-topic
  mastery scores across sessions; weak area detection automatically
  focuses quiz generation on topics below 60% mastery threshold.

• Deployed as a dual-interface system (Streamlit + FastAPI) supporting
  PDF/text ingestion, voice Q&A, adaptive MCQ quizzes, and a real-time
  progress dashboard with mastery bars per topic.
```

---

## License
MIT — free to use, modify, and build on.
