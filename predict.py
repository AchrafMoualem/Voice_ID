'''import numpy as np
import librosa
import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf
import sounddevice as sd
import sys

# ==== PATH MANAGEMENT ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "label_mapping.npy")
MEAN_PATH = os.path.join(BASE_DIR, "mean.npy")
STD_PATH = os.path.join(BASE_DIR, "std.npy")
MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")

# ==== PARAMETERS ====
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
MAX_PAD_LEN = 100

# ==== LOAD LABELS ====
label_to_index = np.load(LABELS_PATH, allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}

# ==== FEATURE EXTRACTION ====
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_from_array(audio):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing live audio: {e}")
        return None

# ==== NORMALIZATION ====
def normalize(X):
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    return (X - mean) / std

# ==== PREDICT FROM FILE ====
def predict_audio(file_path, model_path=MODEL_PATH):
    try:
        print(f"📁 Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)

        print(f"🔎 Extracting MFCC from: {file_path}")
        mfcc = extract_features(file_path)
        if mfcc is None:
            print("❌ MFCC extraction failed.")
            return None, []

        print(f"✅ MFCC shape: {mfcc.shape}")

        X = mfcc[np.newaxis, ..., np.newaxis]
        print(f"📐 Input shape to model: {X.shape}")

        X = normalize(X)
        print(f"⚙️ Normalization applied.")

        preds = model.predict(X)
        print(f"📊 Predictions: {preds}")

        pred_index = np.argmax(preds, axis=1)[0]
        predicted_label = index_to_label[pred_index]

        print(f"✅ Predicted label: {predicted_label}")
        return predicted_label, preds[0].tolist()

    except Exception as e:
        print(f"💥 Error predicting from file: {e}")
        return None, []

# ==== PREDICT FROM LIVE MICROPHONE ====
def predict_live_microphone(model_path=MODEL_PATH):
    try:
        print(f"🎙️ Speak now for {DURATION} seconds...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        model = tf.keras.models.load_model(model_path)
        mfcc = extract_features_from_array(audio)
        if mfcc is None:
            return None, []

        X = mfcc[np.newaxis, ..., np.newaxis]
        X = normalize(X)
        preds = model.predict(X)
        pred_index = np.argmax(preds, axis=1)[0]
        predicted_label = index_to_label[pred_index]

        return predicted_label, preds[0].tolist()

    except Exception as e:
        print(f"Error during live prediction: {e}")
        return None, []

# ==== MAIN ====
if __name__ == "__main__":
    audio_path = os.path.join(BASE_DIR, "test_audio.wav")  # change to a valid test file
    prediction_file = predict_audio(audio_path)
    if prediction_file is not None:
        print(f"📁 Prediction from file: {prediction_file}")
    else:
        print("❌ Failed to predict from file.")



    # Option 2: Predict from live microphone
    #predicted_label, probabilities = predict_live_microphone(model_path)
    #if predicted_label is not None:
        #confidence = max(probabilities)
        #print(f"🎤 Prediction from microphone: {predicted_label}")
        #print(f"📊 Confidence: {confidence:.2f}")
        #print(f"📈 Probabilities: {probabilities}")
    #else:
        #print("❌ Failed to predict from microphone.")'''


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
MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")

# ==== PARAMETERS ====
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
MAX_PAD_LEN = 100

# ==== LOAD LABELS ====
label_to_index = np.load(LABELS_PATH, allow_pickle=True).item()
index_to_label = {v: k for k, v in label_to_index.items()}

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
def predict_audio(file_path, model_path=MODEL_PATH):
    try:
        if file_path.endswith('.webm'):
            wav_path = convert_webm_to_wav(file_path)
            if not wav_path:
                return None, []
        else:
            wav_path = file_path

        wav_path = normalize_audio_volume(wav_path)

        model = tf.keras.models.load_model(model_path)
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

