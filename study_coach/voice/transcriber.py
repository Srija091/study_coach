"""
voice/transcriber.py
=====================
Offline voice transcription using OpenAI Whisper (local).
Runs completely free — no API calls, no cost.

Install: pip install openai-whisper
Also needs: pip install ffmpeg-python  (or install ffmpeg system-wide)
  macOS:  brew install ffmpeg
  Linux:  sudo apt install ffmpeg
  Windows: https://ffmpeg.org/download.html
"""

_whisper_model = None
_whisper_available = None


def is_whisper_available() -> bool:
    """Check if openai-whisper is installed."""
    global _whisper_available
    if _whisper_available is None:
        try:
            import whisper
            _whisper_available = True
        except ImportError:
            _whisper_available = False
    return _whisper_available


def _get_model(size: str = "base"):
    """
    Load Whisper model (cached after first load).
    Sizes and download sizes:
      tiny   ~75MB   — fastest, least accurate
      base   ~145MB  — good balance (recommended)
      small  ~465MB  — more accurate
      medium ~1.5GB  — very accurate
      large  ~3GB    — most accurate
    """
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"Loading Whisper '{size}' model (downloads on first run)...")
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file to text using local Whisper.

    Args:
        audio_path : path to audio file (mp3, wav, m4a, ogg, flac)
        model_size : whisper model size (tiny/base/small/medium/large)

    Returns:
        Transcribed text string, or empty string on failure.
    """
    if not is_whisper_available():
        return ""

    try:
        model = _get_model(size=model_size)
        result = model.transcribe(audio_path, fp16=False)
        text = result.get("text", "").strip()
        return text
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""


def transcribe_with_timestamps(audio_path: str, model_size: str = "base") -> list:
    """
    Transcribe audio and return segments with timestamps.
    Useful for lecture recordings.

    Returns:
        List of { start, end, text }
    """
    if not is_whisper_available():
        return []

    try:
        model = _get_model(size=model_size)
        result = model.transcribe(audio_path, fp16=False)
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end":   round(seg["end"], 2),
                "text":  seg["text"].strip()
            })
        return segments
    except Exception as e:
        print(f"Whisper error: {e}")
        return []
