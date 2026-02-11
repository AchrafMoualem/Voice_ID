from __future__ import annotations

from typing import Any

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from nltk.tokenize import sent_tokenize


def summarize_text(
    summarizer: Any,
    transcription: str,
    language: str,
    max_words: int = 100,
) -> str:
    """
    Generate a short summary from transcription text using an LSA summarizer.

    Returns a user-friendly status message if summarization is not available.
    """
    if summarizer is None:
        return "Résumé indisponible"

    if not transcription or len(transcription.strip()) <= 50:
        return "Résumé indisponible"

    try:
        # Segment text into sentences for more robust summarization
        sentences = sent_tokenize(transcription)
        cleaned_transcription = ". ".join(sentences)

        parser = PlaintextParser.from_string(
            cleaned_transcription,
            Tokenizer(language),
        )
        raw_summary = " ".join(
            str(s) for s in summarizer(parser.document, sentences_count=1)
        )

        words = raw_summary.split()
        if not words:
            return "Résumé indisponible"

        summary = " ".join(words[:max_words])
        if len(words) > max_words:
            summary += "..."
        return summary
    except Exception as exc:
        return f"Erreur lors du résumé: {exc}"

