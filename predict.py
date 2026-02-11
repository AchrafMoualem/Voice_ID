import numpy as np
import librosa
import os
import subprocess
import uuid
import tensorflow as tf
import soundfile as sf

# ==== PATH MANAGEMENT ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "label_mapping.npy")
MEAN_PATH = os.path.join(BASE_DIR, "mean.npy")
STD_PATH = os.path.join(BASE_DIR, "std.npy")
model_path = os.path.join(BASE_DIR, "models", "final_model.h5")


# ==== PARAMETERS ====
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
MAX_PAD_LEN = 100

# ==== LOAD LABELS ====
label_to_index = np.load(LABELS_PATH, allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}

# ==== LOAD MODEL ONCE ====
try:
    SPEAKER_MODEL = tf.keras.models.load_model(model_path)
    print(f"✅ Speaker model loaded from: {model_path}")
except Exception as exc:  # pragma: no cover
    print(f"❌ Failed to load speaker model at startup: {exc}")
    SPEAKER_MODEL = None

# ==== AUDIO CONVERSION ====
def convert_webm_to_wav(webm_path):
    wav_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4().hex}.wav")
    command = ["ffmpeg", "-y", "-i", webm_path, "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        return None

# ==== REMOVE SILENCE ====
def remove_silence(y, sr, top_db=20):
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
        return np.concatenate([y[start:end] for start, end in intervals]) if intervals.any() else y
    except Exception as e:
        print(f"❌ Erreur suppression silence: {e}")
        return y

# ==== AUDIO NORMALIZATION ====
def normalize_audio_volume(file_path, output_path=None):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y = remove_silence(y, sr, top_db=20)
        y = y / np.max(np.abs(y) + 1e-9)
        if not output_path:
            output_path = file_path
        sf.write(output_path, y, sr)
        print(f"🔊 Audio nettoyé et normalisé : {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Échec normalisation : {e}")
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
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    return (X - mean) / std

# ==== PREDICT FROM FILE ====
def predict_audio(file_path, model_path=model_path):
    try:
        if file_path.endswith('.webm'):
            wav_path = convert_webm_to_wav(file_path)
            if not wav_path:
                return None, []
        else:
            wav_path = file_path

        wav_path = normalize_audio_volume(wav_path)

        model = SPEAKER_MODEL
        # Allow explicit override while still avoiding per-request loads
        if model is None or (model_path and model_path != model_path):
            print(f"📁 Loading speaker model from: {model_path}")
            model = tf.keras.models.load_model(model_path)

        if model is None:
            print("❌ Speaker model is not available.")
            return None, []

        mfcc = extract_features(wav_path)
        if mfcc is None:
            return None, []

        X = mfcc[np.newaxis, ..., np.newaxis]
        X = normalize(X)
        preds = model.predict(X)
        pred_index = np.argmax(preds, axis=1)[0]
        predicted_label = index_to_label[pred_index]

        if wav_path != file_path and os.path.exists(wav_path):
            os.remove(wav_path)

        return predicted_label, preds[0].tolist()

    except Exception as e:
        print(f"💥 Prediction error: {e}")
        return None, []

# ==== MAIN TEST ====
if __name__ == "__main__":
    test_file = os.path.join(BASE_DIR, "test_audio.webm")  # ou .wav
    label, probs = predict_audio(test_file)
    if label:
        print(f"🎤 Predicted: {label}")
        print(f"📊 Probabilities: {probs}")
    else:
        print("❌ Prediction failed.")

