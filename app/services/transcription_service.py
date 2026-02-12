from __future__ import annotations

from typing import Any
import os

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
    Transcribe audio (≤5 min) with Whisper and return full text.
    Combines all segments to prevent truncation.
    """
    if whisper_model is None:
        return "Whisper model not available"

    if not os.path.exists(file_path):
        return "Fichier audio introuvable"

    try:
        result = whisper_model.transcribe(file_path, fp16=False)

        # Combine all segments to ensure full transcription
        segments = result.get("segments")
        if segments:
            full_text = "".join(segment["text"] for segment in segments)
        else:
            full_text = result.get("text", "")

        return full_text.strip()

    except Exception as exc:
        return f"Erreur lors de la transcription: {exc}"

