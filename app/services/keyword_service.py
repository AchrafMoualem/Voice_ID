from __future__ import annotations

from typing import Any, List


def extract_keywords(
    kw_model: Any,
    transcription: str,
    language: str,
    min_length: int = 10,
    top_n: int = 5,
) -> List[str]:
    """
    Extract keywords from transcription text using KeyBERT.

    Returns a list of keyword strings or a short status message when
    extraction is not possible.
    """
    if kw_model is None:
        return ["KeyBERT non disponible"]

    if not transcription or len(transcription.strip()) <= min_length:
        return ["Pas assez de texte"]

    try:
        stopwords_lang = "french" if language == "fr" else "english"
        keywords = kw_model.extract_keywords(
            transcription,
            keyphrase_ngram_range=(1, 2),
            stop_words=stopwords_lang,
            top_n=top_n,
        )
        return [kw[0] for kw in keywords]
    except Exception as exc:
        return [f"Erreur lors de l'extraction: {exc}"]

