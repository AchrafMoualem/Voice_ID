import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """
    Base Flask configuration.

    Centralizes application-wide settings and model paths so they are not
    hard-coded inside routes or services.
    """

    # Flask core settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    # Upload / request limits
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "50")) * 1024 * 1024

    # Speaker identification model path
    SPEAKER_MODEL_PATH = os.environ.get(
        "SPEAKER_MODEL_PATH",
        str(BASE_DIR / "models" / "final_model.h5"),
    )

    # Paths for preprocessing artifacts (kept for compatibility with training code)
    LABEL_MAPPING_PATH = os.environ.get(
        "LABEL_MAPPING_PATH",
        str(BASE_DIR / "models" / "label_mapping.npy"),
    )
    MEAN_PATH = os.environ.get(
        "MEAN_PATH",
        str(BASE_DIR / "models" / "mean.npy"),
    )
    STD_PATH = os.environ.get(
        "STD_PATH",
        str(BASE_DIR / "models" / "std.npy"),
    )

