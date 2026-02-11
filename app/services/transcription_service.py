from __future__ import annotations

from typing import Any

from langdetect import detect


def detect_language(text: str) -> str:
    """Best-effort language detection with a safe fallback to English."""
    if not text:
        return "en"

    try:
        return detect(text)
    except Exception:
        return "en"


def transcribe_audio(whisper_model: Any, file_path: str) -> str:
    """
    Transcribe the given audio file with Whisper if a model is available.

    Returns a human-readable error message if the model is not initialized
    or if transcription fails.
    """
    if whisper_model is None:
        return "Whisper model not available"

    try:
        result = whisper_model.transcribe(file_path, language=None, fp16=False)
        return result.get("text", "").strip()
    except Exception as exc:
        # Keep a user-friendly message; detailed logging happens at caller level
        return f"Erreur lors de la transcription: {exc}"

