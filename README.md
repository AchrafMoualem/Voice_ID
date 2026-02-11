## Speaker Identification & Semantic Analysis

This project is a Flask-based web application for:

- **Speaker identification** using a TensorFlow/Keras CNN model and MFCC features.
- **Speech transcription** using OpenAI Whisper.
- **Keyword extraction** using KeyBERT.
- **Automatic summarization** using Sumy (LSA) and NLTK sentence tokenization.

### Project structure

- `app/`
  - `__init__.py`: Flask application factory, global model initialisation (Whisper, KeyBERT, Summarizer).
  - `routes.py`: HTTP routes and JSON API for `/`, `/app`, `/predict`.
  - `services/`
    - `speaker_service.py`: Thin wrapper around the TensorFlow speaker model.
    - `transcription_service.py`: Whisper transcription and language detection helpers.
    - `keyword_service.py`: Keyword extraction with KeyBERT.
    - `summarization_service.py`: Text summarisation with Sumy (LSA).
  - `utils/`
    - `audio_processing.py`: Helpers for safe temporary file cleanup.
  - `templates/`
    - `home.html`: Landing page.
    - `index.html`: Main app UI (upload / record audio, show results).
- `models/`: Placeholder package for trained model artifacts.
- `predict.py`: Core audio preprocessing + speaker prediction logic.
- `scripts/`:
  - `train.py`: Offline training / experimentation scripts for the speaker model.
  - `wav_transform.py`: Utility for converting datasets to WAV.
- `config.py`: Central configuration (model paths, limits, etc.).
- `run.py`: Entrypoint to run the Flask app via `python run.py`.
- `app.py`: Backwards-compatible shim that also runs the Flask app.

### Running the app

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the trained model (`models/final_model.h5`) and preprocessing artifacts
   (`label_mapping.npy`, `mean.npy`, `std.npy`) are present in the project root.
4. Start the server:

   ```bash
   python run.py
   ```

5. Open `http://localhost:5000/` in your browser.

### Notes

- Heavy models (Whisper, KeyBERT, LSA summarizer, TensorFlow speaker model)
  are loaded **once at startup** and reused across requests.
- Training- and dataset-related scripts (`scripts/train.py`, `scripts/wav_transform.py`)
  are kept for experimentation and model retraining and are not used by the web app.

