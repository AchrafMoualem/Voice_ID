import numpy as np
import librosa
import os
import subprocess
import tempfile
import tensorflow as tf
import soundfile as sf

# ==== PATH MANAGEMENT ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow overriding via environment variables (set by the Flask app at startup)
MODEL_PATH = os.environ.get(
    "SPEAKER_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "final_model.h5"),
)

LABELS_PATH = os.environ.get(
    "LABEL_MAPPING_PATH",
    os.path.join(BASE_DIR, "models", "label_mapping.npy"),
)

MEAN_PATH = os.environ.get(
    "MEAN_PATH",
    os.path.join(BASE_DIR, "models", "mean.npy"),
)

STD_PATH = os.environ.get(
    "STD_PATH",
    os.path.join(BASE_DIR, "models", "std.npy"),
)

# ==== PARAMETERS ====
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
MAX_PAD_LEN = 100

# ==== LOAD LABELS ====
label_to_index = np.load(LABELS_PATH, allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}

# ==== LOAD NORMALIZATION ONCE ====
MEAN = np.load(MEAN_PATH)
STD = np.load(STD_PATH)

# ==== LOAD MODEL ONCE ====
try:
    SPEAKER_MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Speaker model loaded from: {MODEL_PATH}")
except Exception as exc:
    print(f"❌ Failed to load speaker model at startup: {exc}")
    SPEAKER_MODEL = None


# ==== REMOVE SILENCE ====
def remove_silence(y, sr, top_db=20):
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
        return np.concatenate([y[start:end] for start, end in intervals]) if len(intervals) > 0 else y
    except Exception as e:
        print(f"❌ Silence removal error: {e}")
        return y


# ==== AUDIO NORMALIZATION ====
def normalize_audio_volume(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y = remove_silence(y, sr, top_db=20)
        y = y / (np.max(np.abs(y)) + 1e-9)
        sf.write(file_path, y, sr)
        return file_path
    except Exception as e:
        print(f"❌ Audio normalization failed: {e}")
        return file_path


# ==== FEATURE EXTRACTION ====
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None


# ==== NORMALIZATION ====
def normalize(X):
    return (X - MEAN) / STD


# ==== CONVERT WEBM TO WAV USING TEMPFILE ====
def convert_webm_to_wav(webm_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        command = [
            "ffmpeg",
            "-y",
            "-i",
            webm_path,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            wav_path,
        ]

        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return wav_path

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        return None


# ==== PREDICT FROM FILE ====
def predict_audio(file_path):
    wav_path = file_path
    temp_file_created = False

    try:
        # ==== CONVERT IF NEEDED ====
        if file_path.endswith(".webm"):
            wav_path = convert_webm_to_wav(file_path)
            if not wav_path:
                return None, []
            temp_file_created = True

        # ==== NORMALIZE AUDIO ====
        wav_path = normalize_audio_volume(wav_path)

        if SPEAKER_MODEL is None:
            print("❌ Speaker model not loaded.")
            return None, []

        # ==== FEATURE EXTRACTION ====
        mfcc = extract_features(wav_path)
        if mfcc is None:
            return None, []

        X = mfcc[np.newaxis, ..., np.newaxis]
        X = normalize(X)

        preds = SPEAKER_MODEL.predict(X)
        pred_index = np.argmax(preds, axis=1)[0]
        predicted_label = index_to_label[pred_index]

        return predicted_label, preds[0].tolist()

    except Exception as e:
        print(f"💥 Prediction error: {e}")
        return None, []

    finally:
        # ==== GUARANTEED CLEANUP ====
        if temp_file_created and wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                print(f"🧹 Temporary file deleted: {wav_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup failed: {cleanup_error}")


# ==== MAIN TEST ====
if __name__ == "__main__":
    test_file = os.path.join(BASE_DIR, "test_audio.webm")
    label, probs = predict_audio(test_file)

    if label:
        print(f"🎤 Predicted: {label}")
        print(f"📊 Probabilities: {probs}")
    else:
        print("❌ Prediction failed.")


