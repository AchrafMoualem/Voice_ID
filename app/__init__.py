import os

from flask import Flask
import whisper
from keybert import KeyBERT
from sumy.summarizers.lsa import LsaSummarizer
import nltk

from config import Config


os.environ["TF_USE_LEGACY_KERAS"] = "0"

# Ensure required NLTK resources are available at startup
nltk.download("punkt", quiet=True)


def create_app(config_class: type[Config] = Config) -> Flask:
    """
    Application factory for the Speaker Identification web app.

    This initializes:
    - Flask application and configuration
    - Global NLP models (Whisper, KeyBERT, LSA summarizer)
    - Routes / blueprints
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config_class)

    # === Make model/artifact paths available to non-Flask modules ===
    os.environ.setdefault("SPEAKER_MODEL_PATH", str(app.config.get("SPEAKER_MODEL_PATH")))
    os.environ.setdefault("LABEL_MAPPING_PATH", str(app.config.get("LABEL_MAPPING_PATH")))
    os.environ.setdefault("MEAN_PATH", str(app.config.get("MEAN_PATH")))
    os.environ.setdefault("STD_PATH", str(app.config.get("STD_PATH")))

    # Import services after env vars are set so modules like `predict.py` pick them up
    from app import services

    # ==== Load global models once at startup ====
    whisper_model = None
    kw_model = None
    summarizer = None

    try:
        whisper_model = whisper.load_model("base")
        app.logger.info("Whisper model loaded successfully")
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.error("Failed to load Whisper model: %s", exc)

    try:
        kw_model = KeyBERT()
        app.logger.info("KeyBERT model loaded successfully")
    except Exception as exc:  # pragma: no cover
        app.logger.error("Failed to load KeyBERT model: %s", exc)

    try:
        summarizer = LsaSummarizer()
        app.logger.info("LSA Summarizer loaded successfully")
    except Exception as exc:  # pragma: no cover
        app.logger.error("Failed to load LSA summarizer: %s", exc)

    # Attach models to app context so services/routes can access them
    app.config["WHISPER_MODEL"] = whisper_model
    app.config["KEYBERT_MODEL"] = kw_model
    app.config["TEXT_SUMMARIZER"] = summarizer


    app.config["TRANSCRIBE_AUDIO"] = lambda file_path: services.transcription_service.transcribe_audio(
        whisper_model, file_path
    )

    # ==== Register blueprints / routes ====
    from .routes import main_bp

    app.register_blueprint(main_bp)

    return app

