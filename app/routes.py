import os
import tempfile
from typing import Any, Dict, List, Tuple

from flask import Blueprint, current_app, jsonify, render_template, request

from .services.speaker_service import SPEAKER_NAMES, predict_speaker
from .services.transcription_service import detect_language, transcribe_audio
from .services.keyword_service import extract_keywords
from .services.summarization_service import summarize_text
from .utils.audio_processing import delete_file_safely


main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home() -> str:
    """Landing page."""
    return render_template("home.html")


@main_bp.route("/app")
def index() -> str:
    """Main application UI."""
    return render_template("index.html")


@main_bp.route("/predict", methods=["POST"])
def handle_prediction():
    """
    Core prediction endpoint.

    - Saves uploaded audio to a temporary file
    - Runs speaker identification
    - Transcribes with Whisper
    - Detects language, extracts keywords, and generates a summary
    """
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "Aucun fichier audio fourni"}), 400

    file = request.files["audio"]
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"temp_audio_{os.getpid()}_{file.filename}",
    )

    try:
        file.save(temp_path)
        current_app.logger.info("Audio saved temporarily at %s", temp_path)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({"error": "Fichier vide ou non sauvegardé"}), 500

        # 1) Speaker prediction
        model_path = current_app.config.get("SPEAKER_MODEL_PATH")
        label, probabilities = predict_speaker(temp_path, model_path=model_path)
        if label is None:
            return jsonify({"error": "Prédiction échouée"}), 500

        # 2) Transcription (Whisper)
        whisper_model = current_app.config.get("WHISPER_MODEL")
        transcription = transcribe_audio(whisper_model, temp_path)
        

        # 3) Language detection
        language = detect_language(transcription) if transcription else "en"

        # 4) Keyword extraction
        kw_model = current_app.config.get("KEYBERT_MODEL")
        keyword_list = extract_keywords(kw_model, transcription, language)

        # 5) Summarization
        summarizer = current_app.config.get("TEXT_SUMMARIZER")
        summary = summarize_text(summarizer, transcription, language)

        # 6) Build JSON response
        confidence = float(max(probabilities)) if probabilities else 0.0
        probabilities_list = [float(p) for p in probabilities] if probabilities else []
        speaker_label = SPEAKER_NAMES.get(str(label), f"{label}")

        response_data: Dict[str, Any] = {
            "predicted_label": speaker_label,
            "confidence": confidence,
            "probabilities": probabilities_list,
            "speaker_names": list(SPEAKER_NAMES.values()),
            "transcription": transcription,
            "keywords": keyword_list,
            "resume": summary,
        }

        current_app.logger.info(
            "Prediction response: %s",
            {k: (v[:50] + "...") if k == "transcription" and isinstance(v, str) else v
             for k, v in response_data.items()},
        )
        return jsonify(response_data)

    except Exception as exc:  # pragma: no cover - defensive logging
        current_app.logger.exception("Unexpected server error during prediction: %s", exc)
        return jsonify({"error": f"Server error: {exc}"}), 500

    finally:
        delete_file_safely(temp_path, logger=current_app.logger)


@main_bp.app_errorhandler(413)
def too_large(e):  # type: ignore[override]
    return jsonify({"error": "Fichier trop volumineux"}), 413


@main_bp.app_errorhandler(500)
def internal_error(e):  # type: ignore[override]
    return jsonify({"error": "Erreur interne du serveur"}), 500

