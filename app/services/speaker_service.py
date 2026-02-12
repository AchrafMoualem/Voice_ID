from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import numpy as np
import predict as pred_module


# Build dynamic mappings from the saved `label_mapping.npy` when available.
# `pred_module` already loads `label_to_index` and `index_to_label` at import-time.
try:
    INDEX_TO_LABEL: Dict[int, str] = getattr(pred_module, "index_to_label")
    # Ensure keys are ints and sorted
    INDEX_TO_LABEL = {int(k): v for k, v in INDEX_TO_LABEL.items()}
except Exception:
    INDEX_TO_LABEL = {}


# `SPEAKER_LABELS` is an ordered list of speaker names (index -> name)
if INDEX_TO_LABEL:
    SPEAKER_LABELS: List[str] = [INDEX_TO_LABEL[i] for i in sorted(INDEX_TO_LABEL.keys())]
    # Also expose numeric-string -> name mapping for compatibility
    SPEAKER_NAMES = {str(i): INDEX_TO_LABEL[i] for i in sorted(INDEX_TO_LABEL.keys())}
else:
    SPEAKER_LABELS = []
    SPEAKER_NAMES = {}


def predict_speaker(file_path: str) -> Tuple[Optional[str], List[float]]:
    """Run the speaker identification model and return (label, probabilities).

    The returned `label` is the speaker name when the model/reporting uses
    the original directory names as labels (as produced by training). If the
    model returns an index instead, downstream code should consult
    `SPEAKER_LABELS` or `SPEAKER_NAMES` for human-readable names.
    """
    return pred_module.predict_audio(file_path)
