"""
Personalized AI Study Coach
============================
100% Free Stack:
  PyMuPDF            → PDF ingestion
  sentence-transformers → local embeddings (offline)
  ChromaDB           → free vector store
  Groq free API      → LLM answers + quiz generation
  Whisper (local)    → free offline voice transcription
  diskcache          → persistent memory (replaces Redis)
  Streamlit          → UI

Run: streamlit run app.py
"""

import streamlit as st
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.pdf_loader import load_pdf, load_text
from embeddings.chroma_store import index_chunks, retrieve, get_stats, clear_collection
from rag.qa_chain import answer_question
from voice.transcriber import transcribe_audio, is_whisper_available
from memory.session_memory import (
    update_topic_score, get_weak_areas, get_all_topics,
    get_session_summary, reset_user
)
from quiz.quiz_generator import generate_quiz, TOPIC_HINTS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Coach",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
  .title  { font-size:1.9rem; font-weight:700; color:#1a3a5c; }
  .sub    { color:#6c757d; font-size:.93rem; margin-bottom:1.2rem; }
  .answer-box { background:#f0f6ff; border-left:4px solid #1a3a5c;
                border-radius:6px; padding:1rem 1.2rem; font-size:.95rem; line-height:1.7; }
  .source-box { background:#f8f9fa; border-radius:6px;
                padding:.55rem .9rem; margin:.3rem 0; font-size:.8rem; color:#444; }
  .quiz-q     { background:#fff8e1; border-left:4px solid #f9a825;
                border-radius:6px; padding:.8rem 1rem; margin:.5rem 0; font-size:.95rem; }
  .correct    { background:#e8f5e9; border-left:4px solid #43a047;
                border-radius:6px; padding:.6rem .9rem; font-size:.9rem; }
  .wrong      { background:#fdecea; border-left:4px solid #e53935;
                border-radius:6px; padding:.6rem .9rem; font-size:.9rem; }
  .mastery-bar{ height:8px; border-radius:4px; background:#e9ecef; margin:3px 0 8px; }
  .stat-num   { font-size:1.5rem; font-weight:700; color:#1a3a5c; }
  .weak-tag   { display:inline-block; background:#fdecea; color:#b71c1c;
                border-radius:12px; padding:2px 10px; font-size:.78rem;
                font-weight:600; margin:2px; }
  .strong-tag { display:inline-block; background:#e8f5e9; color:#1b5e20;
                border-radius:12px; padding:2px 10px; font-size:.78rem;
                font-weight:600; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🎓 AI Study Coach</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload notes → Ask questions (text or voice) → Get quizzed → Track mastery. 100% free.</p>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    groq_key = st.text_input("Groq API Key", type="password",
                              placeholder="gsk_...",
                              help="Free at groq.com — no credit card")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    user_id = st.text_input("Your Name / Student ID", value="student_1",
                             help="Used to track your personal progress")

    groq_model = st.selectbox("Model", [
        "llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"
    ])

    st.markdown("---")
    st.markdown("**📚 Indexed Materials**")
    stats = get_stats()
    st.markdown(f'<p class="stat-num">{stats["count"]}</p>'
                f'<p style="font-size:.8rem;color:#6c757d">chunks indexed</p>',
                unsafe_allow_html=True)

    if st.button("🗑️ Clear Materials"):
        clear_collection()
        st.success("Cleared.")
        st.rerun()

    st.markdown("---")
    st.markdown("**🧠 Your Progress**")
    summary = get_session_summary(user_id)
    st.markdown(f'<p class="stat-num">{summary["total_topics"]}</p>'
                f'<p style="font-size:.8rem;color:#6c757d">topics tracked</p>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="stat-num">{summary["avg_mastery"]}%</p>'
                f'<p style="font-size:.8rem;color:#6c757d">avg mastery</p>',
                unsafe_allow_html=True)

    if st.button("🔄 Reset My Progress"):
        reset_user(user_id)
        st.success("Progress reset.")
        st.rerun()

    st.markdown("---")
    st.markdown("**Setup**")
    st.code("pip install -r requirements.txt", language="bash")

    if not is_whisper_available():
        st.warning("Voice disabled. Install: `pip install openai-whisper`")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "💬 Ask", "🧪 Quiz Me", "📊 Progress"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload Study Materials
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Your Study Materials")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📄 Upload PDF or Text File**")
        uploaded_file = st.file_uploader(
            "Upload PDF or .txt", type=["pdf", "txt"], label_visibility="collapsed"
        )
        doc_id = st.text_input("Document name", placeholder="e.g. chapter-3-ml",
                                value="")

        if uploaded_file and st.button("📥 Index File", type="primary"):
            if not doc_id:
                doc_id = uploaded_file.name.replace(" ", "_").lower()
            with st.spinner("Loading and indexing..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix="." + uploaded_file.name.split(".")[-1]
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if uploaded_file.name.endswith(".pdf"):
                    chunks = load_pdf(tmp_path, doc_id=doc_id)
                else:
                    text = open(tmp_path).read()
                    chunks = load_text(text, doc_id=doc_id)

                os.unlink(tmp_path)
                indexed = index_chunks(chunks)
                st.success(f"✅ Indexed **{indexed} chunks** from '{doc_id}'")

    with col2:
        st.markdown("**📝 Paste Notes Directly**")
        pasted_text = st.text_area("Paste your notes here", height=200,
                                    placeholder="Paste lecture notes, textbook excerpts...")
        paste_id = st.text_input("Notes name", value="my-notes",
                                  key="paste_id")

        if st.button("📥 Index Notes", type="primary", key="idx_paste"):
            if not pasted_text.strip():
                st.warning("Please paste some text first.")
            else:
                with st.spinner("Indexing..."):
                    chunks = load_text(pasted_text, doc_id=paste_id)
                    indexed = index_chunks(chunks)
                    st.success(f"✅ Indexed **{indexed} chunks** from '{paste_id}'")

    # Demo content loader
    with st.expander("📋 Load demo content — Machine Learning basics"):
        demo_text = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Types of Machine Learning:

1. Supervised Learning
In supervised learning, the algorithm is trained on labeled data. The model learns to map inputs to outputs based on example input-output pairs. Common algorithms include linear regression, decision trees, support vector machines, and neural networks. Applications include email spam detection, image classification, and medical diagnosis.

2. Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled responses. Clustering algorithms like K-means group similar data points. Dimensionality reduction techniques like PCA reduce the number of features. Applications include customer segmentation and anomaly detection.

3. Reinforcement Learning
An agent learns to make decisions by receiving rewards or penalties. The goal is to maximize cumulative reward. Used in game playing (AlphaGo), robotics, and autonomous vehicles.

Key Concepts:

Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new data. It can be prevented using regularization, dropout, cross-validation, and more training data.

Underfitting happens when a model is too simple to capture the patterns in data. Both training and test performance are poor.

The bias-variance tradeoff describes the balance between underfitting (high bias) and overfitting (high variance). A good model minimizes both.

Feature engineering is the process of using domain knowledge to create features that make machine learning algorithms work better.

Neural Networks and Deep Learning:
Neural networks are inspired by the human brain and consist of layers of interconnected nodes. Deep learning uses neural networks with many layers. Convolutional Neural Networks (CNNs) are used for image processing. Recurrent Neural Networks (RNNs) handle sequential data like text and time series.

Model Evaluation:
Common metrics include accuracy, precision, recall, F1-score, and AUC-ROC. Cross-validation helps estimate model performance on unseen data. The confusion matrix shows true/false positives and negatives.
        """
        if st.button("Load Demo Content"):
            with st.spinner("Indexing demo content..."):
                chunks = load_text(demo_text, doc_id="ml-basics")
                indexed = index_chunks(chunks)
                st.success(f"✅ Indexed {indexed} chunks of ML basics demo content!")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Ask Questions
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Ask Your Study Coach")

    if get_stats()["count"] == 0:
        st.warning("No materials indexed yet. Go to the Upload tab first.")
    else:
        # Voice input
        voice_text = ""
        if is_whisper_available():
            st.markdown("**🎤 Ask by Voice**")
            audio_file = st.file_uploader(
                "Upload audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a"],
                key="voice_upload", label_visibility="collapsed"
            )
            if audio_file:
                with st.spinner("Transcribing with Whisper..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix="." + audio_file.name.split(".")[-1]
                    ) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    voice_text = transcribe_audio(tmp_path)
                    os.unlink(tmp_path)
                if voice_text:
                    st.success(f"Transcribed: *{voice_text}*")

        # Text input
        st.markdown("**💬 Ask by Text**")
        question = st.text_input(
            "Your question",
            value=voice_text,
            placeholder="What is the difference between overfitting and underfitting?",
            label_visibility="collapsed"
        )

        # Quick example questions
        examples = [
            "What is supervised learning?",
            "Explain overfitting and how to prevent it",
            "What is the bias-variance tradeoff?",
            "How do neural networks work?",
            "What metrics are used to evaluate models?",
        ]
        st.caption("Quick questions:")
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                question = ex

        top_k = st.slider("Chunks to retrieve", 2, 8, 4, key="qa_topk")

        if st.button("🔍 Ask Coach", type="primary", disabled=not question):
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Enter your Groq API key in the sidebar. Free at groq.com")
            else:
                with st.spinner("Thinking..."):
                    result = answer_question(
                        question=question,
                        user_id=user_id,
                        model=groq_model,
                        top_k=top_k
                    )

                if result.get("error"):
                    st.error(result["error"])
                else:
                    # Answer
                    st.markdown("**📖 Answer:**")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True
                    )

                    # Sources
                    if result.get("sources"):
                        st.markdown("**📎 From your notes:**")
                        for s in result["sources"]:
                            st.markdown(
                                f'<div class="source-box">📄 <strong>{s["doc_id"]}</strong> · '
                                f'relevance {s["score"]:.0%}<br>{s["text"][:180]}...</div>',
                                unsafe_allow_html=True
                            )

                    # Update memory with detected topic
                    if result.get("topic"):
                        st.caption(f"📌 Topic tracked: **{result['topic']}**")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Quiz Mode
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Quiz Mode")

    if get_stats()["count"] == 0:
        st.warning("No materials indexed yet. Go to the Upload tab first.")
    else:
        col_q1, col_q2 = st.columns([2, 1])
        with col_q1:
            weak = get_weak_areas(user_id)
            if weak:
                st.info(f"🎯 Focusing on your weak areas: {', '.join(weak)}")
                focus_topic = st.selectbox("Or pick a topic", ["Auto (weak areas)"] + list(TOPIC_HINTS.keys()))
            else:
                focus_topic = st.selectbox("Pick a topic to quiz on", ["Auto"] + list(TOPIC_HINTS.keys()))

        with col_q2:
            num_questions = st.selectbox("Questions", [3, 5, 10], index=1)

        if st.button("🎲 Generate Quiz", type="primary"):
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Enter your Groq API key in the sidebar.")
            else:
                topic = None if focus_topic in ("Auto (weak areas)", "Auto") else focus_topic
                if topic is None and weak:
                    topic = weak[0]

                with st.spinner("Generating quiz questions..."):
                    quiz = generate_quiz(
                        topic=topic,
                        num_questions=num_questions,
                        model=groq_model
                    )

                if quiz.get("error"):
                    st.error(quiz["error"])
                elif not quiz.get("questions"):
                    st.warning("Could not generate questions. Try indexing more content first.")
                else:
                    st.session_state["quiz"] = quiz["questions"]
                    st.session_state["quiz_topic"] = topic or "general"
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_submitted"] = False

        # Display quiz
        if "quiz" in st.session_state and st.session_state["quiz"]:
            quiz_qs = st.session_state["quiz"]
            topic_name = st.session_state.get("quiz_topic", "general")

            st.markdown(f"**Quiz: {topic_name.replace('-', ' ').title()}** — {len(quiz_qs)} questions")
            st.markdown("---")

            for i, q in enumerate(quiz_qs):
                st.markdown(
                    f'<div class="quiz-q"><strong>Q{i+1}.</strong> {q["question"]}</div>',
                    unsafe_allow_html=True
                )
                options = q.get("options", [])
                if options:
                    selected = st.radio(
                        f"q{i}_options",
                        options,
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                    st.session_state["quiz_answers"][i] = selected
                else:
                    answer = st.text_input(f"Your answer", key=f"q_{i}_text")
                    st.session_state["quiz_answers"][i] = answer

            if st.button("✅ Submit Quiz", type="primary"):
                st.session_state["quiz_submitted"] = True

            # Show results
            if st.session_state.get("quiz_submitted"):
                st.markdown("---")
                st.markdown("### 📊 Results")
                correct_count = 0

                for i, q in enumerate(quiz_qs):
                    user_ans = st.session_state["quiz_answers"].get(i, "")
                    correct_ans = q.get("answer", "")
                    is_correct = _check_answer(user_ans, correct_ans)

                    if is_correct:
                        correct_count += 1
                        st.markdown(
                            f'<div class="correct">✅ Q{i+1}: Correct! — {q["question"]}<br>'
                            f'<em>{correct_ans}</em></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="wrong">❌ Q{i+1}: {q["question"]}<br>'
                            f'Your answer: <em>{user_ans}</em><br>'
                            f'Correct: <em>{correct_ans}</em></div>',
                            unsafe_allow_html=True
                        )

                    update_topic_score(user_id, topic_name, is_correct)

                score_pct = int(correct_count / len(quiz_qs) * 100)
                if score_pct >= 80:
                    st.success(f"🎉 Score: {correct_count}/{len(quiz_qs)} ({score_pct}%) — Great job!")
                elif score_pct >= 50:
                    st.warning(f"📚 Score: {correct_count}/{len(quiz_qs)} ({score_pct}%) — Keep studying!")
                else:
                    st.error(f"💪 Score: {correct_count}/{len(quiz_qs)} ({score_pct}%) — Review this topic!")

                if st.button("🔄 Try Again"):
                    del st.session_state["quiz"]
                    del st.session_state["quiz_submitted"]
                    st.rerun()


def _check_answer(user_ans: str, correct_ans: str) -> bool:
    """Fuzzy answer matching — handles MCQ letter or full text."""
    if not user_ans or not correct_ans:
        return False
    u = user_ans.strip().lower()
    c = correct_ans.strip().lower()
    if u == c:
        return True
    # MCQ: user picks "A. Linear regression" — check if starts with same letter
    if len(u) > 0 and len(c) > 0 and u[0] == c[0]:
        return True
    # Partial match
    return u in c or c in u


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Progress Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"📊 Progress Dashboard — {user_id}")

    topics = get_all_topics(user_id)

    if not topics:
        st.info("No quiz data yet. Take a quiz in the Quiz Me tab to track your progress!")
    else:
        summary = get_session_summary(user_id)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Topics Studied", summary["total_topics"])
        m2.metric("Avg Mastery", f"{summary['avg_mastery']}%")
        m3.metric("Total Questions", summary["total_attempts"])
        m4.metric("Correct Answers", summary["total_correct"])

        # Weak / strong areas
        weak = get_weak_areas(user_id)
        st.markdown("---")
        col_w, col_s = st.columns(2)
        with col_w:
            st.markdown("**🔴 Weak Areas (< 60%)**")
            if weak:
                for w in weak:
                    st.markdown(f'<span class="weak-tag">📖 {w}</span>', unsafe_allow_html=True)
            else:
                st.success("No weak areas! You're crushing it 🎉")
        with col_s:
            strong = [t for t, d in topics.items() if d.get("mastery", 0) >= 80]
            st.markdown("**🟢 Strong Areas (≥ 80%)**")
            if strong:
                for s in strong:
                    st.markdown(f'<span class="strong-tag">⭐ {s}</span>', unsafe_allow_html=True)
            else:
                st.info("Keep quizzing to build strong areas!")

        # Per-topic mastery bars
        st.markdown("---")
        st.markdown("**📈 Topic Mastery Breakdown**")
        for topic, data in sorted(topics.items(), key=lambda x: x[1].get("mastery", 0)):
            mastery = data.get("mastery", 0)
            attempts = data.get("attempts", 0)
            correct = data.get("correct", 0)
            color = "#43a047" if mastery >= 80 else ("#f9a825" if mastery >= 50 else "#e53935")
            st.markdown(
                f"**{topic}** — {mastery}% mastery ({correct}/{attempts} correct)"
            )
            st.markdown(
                f'<div class="mastery-bar"><div style="width:{mastery}%;height:100%;'
                f'background:{color};border-radius:4px;"></div></div>',
                unsafe_allow_html=True
            )
