"""
quiz/quiz_generator.py
=======================
Generates adaptive quiz questions using Groq (free).
Focuses on weak areas detected from student memory.
Supports MCQ and short-answer formats.
"""

import os
import json
import re
from groq import Groq
from embeddings.chroma_store import retrieve

# Topic → search hint for retrieval
TOPIC_HINTS = {
    "supervised learning":    "supervised learning labeled data regression classification",
    "unsupervised learning":  "unsupervised clustering k-means dimensionality reduction",
    "neural networks":        "neural network deep learning layers backpropagation",
    "overfitting":            "overfitting underfitting regularization generalization",
    "model evaluation":       "accuracy precision recall f1 score confusion matrix roc",
    "reinforcement learning": "reinforcement learning reward agent policy environment",
    "feature engineering":    "feature engineering selection extraction transformation",
    "bias variance":          "bias variance tradeoff underfitting overfitting complexity",
    "general ml":             "machine learning algorithm model training data"
}

QUIZ_SYSTEM_PROMPT = """You are a quiz generator for a study coach app.
Generate multiple-choice quiz questions based on the provided study material.

STRICT OUTPUT FORMAT — respond with ONLY valid JSON, no other text:
{
  "questions": [
    {
      "question": "What is ...?",
      "options": ["A. option1", "B. option2", "C. option3", "D. option4"],
      "answer": "A. option1",
      "explanation": "Brief explanation of why this is correct."
    }
  ]
}

Rules:
- Generate exactly the number of questions requested
- Each question must have exactly 4 options (A, B, C, D)
- The answer field must exactly match one of the options
- Questions should test understanding, not just memorization
- Vary difficulty: mix easy recall, application, and analysis questions
- Base questions ONLY on the provided context
- No markdown, no code fences — pure JSON only"""


def generate_quiz(
    topic: str = None,
    num_questions: int = 5,
    model: str = "llama-3.3-70b-versatile"
) -> dict:
    """
    Generate an adaptive quiz focused on a topic.

    Args:
        topic         : topic to quiz on (uses TOPIC_HINTS for retrieval)
        num_questions : number of questions to generate
        model         : Groq model

    Returns:
        { questions: [...], topic, error }
    """
    result = {"questions": [], "topic": topic or "general", "error": None}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        result["error"] = "Groq API key not set. Get one free at groq.com"
        return result

    # Get search query for this topic
    search_query = TOPIC_HINTS.get(topic, topic or "key concepts overview")

    # Retrieve relevant content to base questions on
    chunks = retrieve(search_query, top_k=6)
    if not chunks:
        result["error"] = "No study material found. Upload notes first!"
        return result

    # Build context for question generation
    context = "\n\n---\n\n".join(
        f"[From '{c['doc_id']}']\n{c['text']}"
        for c in chunks[:4]
    )

    topic_display = (topic or "the material").replace("-", " ").title()
    user_prompt = f"""Study Material:
{context}

---
Generate exactly {num_questions} multiple-choice questions about: {topic_display}

Return ONLY the JSON object — no other text."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.4
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        parsed = json.loads(raw)
        questions = parsed.get("questions", [])

        # Validate questions
        valid = []
        for q in questions:
            if (q.get("question") and
                    q.get("options") and len(q["options"]) == 4 and
                    q.get("answer")):
                valid.append(q)

        result["questions"] = valid[:num_questions]

        if not valid:
            result["error"] = "Generated questions were invalid. Try again."

    except json.JSONDecodeError:
        # Fallback: try to extract questions from non-JSON response
        result["questions"] = _fallback_parse(raw, num_questions)
        if not result["questions"]:
            result["error"] = "Could not parse quiz questions. Try a different model."
    except Exception as e:
        err = str(e)
        if "rate" in err.lower():
            result["error"] = "Rate limit — wait a moment and retry."
        elif "api_key" in err.lower():
            result["error"] = "Invalid Groq API key."
        else:
            result["error"] = f"Error: {err}"

    return result


def _fallback_parse(raw: str, num_questions: int) -> list:
    """
    Last-resort parser if JSON parsing fails.
    Tries to extract Q&A pairs from plain text.
    """
    questions = []
    lines = raw.split("\n")
    current_q = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Looks like a question
        if line.endswith("?") and len(line) > 15:
            if current_q:
                questions.append(current_q)
            current_q = {"question": line, "options": [], "answer": "", "explanation": ""}
        # Looks like an option
        elif current_q and re.match(r'^[A-D][\.\)]\s+', line):
            current_q["options"].append(line)
            if not current_q["answer"] and len(current_q["options"]) == 1:
                current_q["answer"] = line

    if current_q:
        questions.append(current_q)

    # Pad options if needed
    for q in questions:
        while len(q["options"]) < 4:
            q["options"].append(f"{'ABCD'[len(q['options'])]}. (option)")

    return questions[:num_questions]
