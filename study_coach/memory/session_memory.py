"""
memory/session_memory.py
=========================
Persistent per-student memory using diskcache.
Tracks topics studied, quiz scores, mastery %, weak areas.

Replaces Redis — pure Python, zero setup, free forever.
Install: pip install diskcache
"""

import diskcache as dc
import json

CACHE_DIR = "./study_cache"
_cache = None


def _get_cache() -> dc.Cache:
    global _cache
    if _cache is None:
        _cache = dc.Cache(CACHE_DIR)
    return _cache


# ── Topic scoring ──────────────────────────────────────────────────────────────

def update_topic_score(user_id: str, topic: str, correct: bool) -> dict:
    """
    Record a quiz answer for a topic.
    Updates attempts, correct count, and mastery %.

    Returns updated topic data.
    """
    cache = _get_cache()
    key = f"user:{user_id}:topics"

    data = cache.get(key, default={})
    if not isinstance(data, dict):
        data = {}

    if topic not in data:
        data[topic] = {"attempts": 0, "correct": 0, "mastery": 0}

    data[topic]["attempts"] += 1
    if correct:
        data[topic]["correct"] += 1

    # Mastery = weighted recent performance
    # Simple: correct / attempts * 100
    attempts = data[topic]["attempts"]
    correct_count = data[topic]["correct"]
    data[topic]["mastery"] = round(correct_count / attempts * 100)

    cache[key] = data
    return data[topic]


def get_weak_areas(user_id: str, threshold: int = 60) -> list:
    """
    Return topics where mastery is below threshold%.
    These are what the quiz should focus on.
    """
    topics = get_all_topics(user_id)
    weak = [
        topic for topic, data in topics.items()
        if data.get("mastery", 100) < threshold
        and data.get("attempts", 0) >= 2  # only flag after 2+ attempts
    ]
    return sorted(weak, key=lambda t: topics[t].get("mastery", 0))


def get_strong_areas(user_id: str, threshold: int = 80) -> list:
    """Return topics where mastery is at or above threshold%."""
    topics = get_all_topics(user_id)
    return [
        topic for topic, data in topics.items()
        if data.get("mastery", 0) >= threshold
    ]


def get_all_topics(user_id: str) -> dict:
    """Return all tracked topics and their data for a user."""
    cache = _get_cache()
    key = f"user:{user_id}:topics"
    return cache.get(key, default={})


def get_session_summary(user_id: str) -> dict:
    """Return a summary of a student's overall progress."""
    topics = get_all_topics(user_id)

    if not topics:
        return {
            "total_topics": 0,
            "avg_mastery": 0,
            "total_attempts": 0,
            "total_correct": 0,
            "weak_areas": [],
            "strong_areas": []
        }

    masteries = [d.get("mastery", 0) for d in topics.values()]
    total_attempts = sum(d.get("attempts", 0) for d in topics.values())
    total_correct  = sum(d.get("correct", 0) for d in topics.values())

    return {
        "total_topics":   len(topics),
        "avg_mastery":    round(sum(masteries) / len(masteries)),
        "total_attempts": total_attempts,
        "total_correct":  total_correct,
        "weak_areas":     get_weak_areas(user_id),
        "strong_areas":   get_strong_areas(user_id)
    }


def log_question_asked(user_id: str, question: str, topic: str) -> None:
    """Track questions a student has asked (for personalization)."""
    cache = _get_cache()
    key = f"user:{user_id}:questions"
    history = cache.get(key, default=[])
    history.append({"question": question, "topic": topic})
    # Keep last 100 questions
    cache[key] = history[-100:]


def get_question_history(user_id: str) -> list:
    """Return list of questions the student has asked."""
    return _get_cache().get(f"user:{user_id}:questions", default=[])


def reset_user(user_id: str) -> None:
    """Clear all progress data for a student."""
    cache = _get_cache()
    for key in [
        f"user:{user_id}:topics",
        f"user:{user_id}:questions"
    ]:
        try:
            del cache[key]
        except KeyError:
            pass


def export_progress(user_id: str) -> str:
    """Export student progress as JSON string."""
    return json.dumps({
        "user_id": user_id,
        "topics":  get_all_topics(user_id),
        "summary": get_session_summary(user_id)
    }, indent=2)
