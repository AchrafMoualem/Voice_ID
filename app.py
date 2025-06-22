from flask import Flask, request, jsonify, render_template
import predict as pred_module
import os
import tempfile
import whisper
from keybert import KeyBERT
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from langdetect import detect
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')  # Télécharger le tokenizer au démarrage

# ==== Initialisation Flask ====
app = Flask(__name__)
os.environ["TF_USE_LEGACY_KERAS"] = "0"

# ==== Noms des locuteurs ====
speaker_names = {
    "0": "Achraf", "1": "Akish", "2": "Angelina", "3": "Ariana", "4": "Darwin",
    "5": "Jamila", "6": "Lana", "7": "Sam", "8": "Shamrock", "9": "Simon"
}

# ==== Chargement des modèles ====
try:
    whisper_model = whisper.load_model("base")
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    whisper_model = None

try:
    kw_model = KeyBERT()
    print("✅ KeyBERT model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load KeyBERT model: {e}")
    kw_model = None

try:
    summarizer = LsaSummarizer()
    print("✅ LSA Summarizer loaded successfully")
except Exception as e:
    print(f"❌ Failed to load LSA Summarizer: {e}")
    summarizer = None

# ==== Routes ====
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def index():
    return render_template('index.html')

# ==== Détection de langue ====
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ==== Prédiction ====
@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'audio' not in request.files or request.files['audio'].filename == '':
        return jsonify({'error': 'Aucun fichier audio fourni'}), 400

    file = request.files['audio']
    temp_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.getpid()}_{file.filename}")

    try:
        file.save(temp_path)
        print(f"📂 Audio sauvegardé temporairement: {temp_path}")

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Fichier vide ou non sauvegardé'}), 500

        # 🎯 Prédiction du locuteur
        try:
            label, probabilities = pred_module.predict_audio(
                temp_path,
                model_path=r"C:\Users\hp\desktop\speacker_identification\Final_Model_2.h5"
            )
            print("🎯 Résultat de predict_audio:", label, probabilities)
            if label is None:
                return jsonify({'error': 'Prédiction échouée'}), 500
        except Exception as e:
            return jsonify({'error': f'Erreur prédiction: {str(e)}'}), 500

        # 📜 Transcription avec Whisper
        transcription = ""
        if whisper_model:
            try:
                whisper_result = whisper_model.transcribe(temp_path, language=None, fp16=False)
                transcription = whisper_result.get("text", "").strip()
                print(f"📜 Transcription: {transcription[:100]}...")
            except Exception as e:
                print(f"❌ Erreur transcription: {e}")
                transcription = "Erreur lors de la transcription"
        else:
            transcription = "Whisper model not available"

        # 🌍 Détection de langue
        language = detect_language(transcription) if transcription else "en"

        # 🗝️ Extraction des mots-clés avec KeyBERT
        keyword_list = []
        if kw_model and transcription and len(transcription.strip()) > 10:
            try:
                stopwords_lang = 'french' if language == 'fr' else 'english'
                keywords = kw_model.extract_keywords(
                    transcription,
                    keyphrase_ngram_range=(1, 2),
                    stop_words=stopwords_lang,
                    top_n=5
                )
                keyword_list = [kw[0] for kw in keywords]
                print("🔑 Mots-clés:", keyword_list)
            except Exception as e:
                print(f"❌ Erreur mots-clés: {e}")
                keyword_list = ["Erreur lors de l'extraction"]
        else:
            keyword_list = ["Pas assez de texte"]

        # 🧠 Résumé automatique avec SUMY (LSA) + segmentation NLTK
        summary = ""
        max_words = 100  # Nombre de mots max dans le résumé

        if summarizer and transcription and len(transcription.strip()) > 50:
            try:
                # ➕ Segmentation avec NLTK
                sentences = sent_tokenize(transcription)
                cleaned_transcription = ". ".join(sentences)

                parser = PlaintextParser.from_string(cleaned_transcription, Tokenizer(language))
                raw_summary = " ".join(str(s) for s in summarizer(parser.document, sentences_count=1))

                # ✂️ Limiter le résumé à X mots
                words = raw_summary.split()
                summary = " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
            except Exception as e:
                print(f"❌ Erreur résumé: {e}")
                summary = "Erreur lors du résumé"
        else:
            summary = "Résumé indisponible"

        # 📝 Affichage console
        print("📝 Résumé bref :", summary)

        # 📈 Réponse finale
        confidence = float(max(probabilities)) if probabilities else 0.0
        probabilities_list = [float(p) for p in probabilities] if probabilities else []
        speaker_label = speaker_names.get(str(label), f"{label}")

        response_data = {
            'predicted_label': speaker_label,
            'confidence': confidence,
            'probabilities': probabilities_list,
            'speaker_names': list(speaker_names.values()),
            'transcription': transcription,
            'keywords': keyword_list,
            'resume': summary
        }

        print("📤 Réponse envoyée:", {
            k: v if k != 'transcription' else f"{v[:50]}..." for k, v in response_data.items()
        })

        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"💥 Erreur générale : {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ Fichier temporaire supprimé: {temp_path}")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer le fichier temporaire: {e}")

# ==== Gestion des erreurs ====
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Fichier trop volumineux'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

# ==== Lancement de l'app ====
if __name__ == '__main__':
    print("🚀 Démarrage du serveur Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)
