from __future__ import annotations

from typing import List, Optional, Tuple

import predict as pred_module


# Human-readable speaker names used by the UI
SPEAKER_NAMES = {
    "0": "Achraf",
    "1": "Akish",
    "2": "Angelina",
    "3": "Ariana",
    "4": "Darwin",
    "5": "Jamila",
    "6": "Lana",
    "7": "Sam",
    "8": "Shamrock",
    "9": "Simon",
}


def predict_speaker(
    file_path: str,
    model_path: Optional[str] = None,
) -> Tuple[Optional[str], List[float]]:
    """
    Run the speaker identification model on the given audio file.

    This is a thin wrapper around the lower-level `predict.predict_audio`
    function, keeping core model logic isolated while exposing a simple
    interface for the web layer.
    """
    label, probabilities = pred_module.predict_audio(
        file_path,
        model_path=model_path,
    )
    return label, probabilities

