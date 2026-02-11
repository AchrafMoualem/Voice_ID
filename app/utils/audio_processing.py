from __future__ import annotations

import os
from typing import Optional


def delete_file_safely(path: str, logger: Optional[object] = None) -> None:
    """
    Remove a temporary file, logging a warning instead of raising if deletion
    fails.
    """
    if not path or not os.path.exists(path):
        return

    try:
        os.remove(path)
        if logger is not None:
            logger.info("Temporary file deleted: %s", path)
    except Exception as exc:  # pragma: no cover - defensive
        if logger is not None:
            logger.warning("Unable to delete temporary file %s: %s", path, exc)

