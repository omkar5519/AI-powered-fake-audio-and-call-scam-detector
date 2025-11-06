import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
import os
import datetime
import pandas as pd
import requests
import whisper
import gdown
import zipfile

# ---------------- CONFIG ----------------
SCAM_KEYWORDS = [
    "otp", "bank", "lottery", "payment", "insurance", "refund",
    "reward", "transaction", "prize", "investment", "claim",
    "credit card", "account", "loan", "id verification", "upi", "gift"
]

# ---------------- PATHS ----------------
MODEL_ZIP_URL = "https://drive.google.com/uc?id=1V83mY45Vein7T4YzaBLHj10HSlCTpZt3"  # Your Google Drive file ID
MODEL_ZIP_PATH = "files/audio_cnn_rnn_model_tf.zip"
MODEL_FOLDER = "files/audio_cnn_rnn_model_tf"
ENCODER_PATH = "files/label_encoder.pkl"
UPLOAD_DIR = "uploads"

# --- Download & unzip model ---
def download_and_extract_model():
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs("files", exist_ok=True)
        st.info("Downloading model...")
        gdown.download(MODEL_ZIP_URL, MODEL_ZIP_PATH, quiet=False)
        st.info("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_FOLDER)
        st.success("Model ready!")

# --- Load model, encoder, and Whisper ---
@st.cache_resource
def load_model_and_encoder():
    download_and_extract_model()
    model = tf.keras.models.load_model(MODEL_FOLDER, compile=False)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# --- Feature extraction ---
def extract_features(audio_path, sr=16000, n_mfcc=40, duration=3):
    y, sr = librosa.load(audio_path, sr=sr)
    max_len = sr * duration
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)
    scam_plane = np.zeros_like(mfcc)
    features = np.concatenate([mfcc, scam_plane], axis=-1)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=1)
    return features.astype(np.float32)

# --- Scam keyword detection ---
def analyze_transcription(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
        text = result["text"].lower()
        found_keywords = [kw for kw in SCAM_KEYWORDS if kw in text]
        scam_score = min(len(found_keywords), 5) / 5.0 * 100
        return text.strip(), found_keywords, scam_score
    except Exception:
        return "", [], 0

# --- Streamlit UI ---
st.title("🎙️ Real / Fake / Scam Voice Detector")
st.caption("Detects AI voices, scam calls, and real speech using Deep Learning + Whisper")

# Load models
model, encoder = load_model_and_encoder()
whisper_model = load_whisper()

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    audio_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    try:
        # --- Prediction ---
        features = extract_features(audio_path)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction) * 100)

        # --- Whisper + Scam Analysis ---
        with st.spinner("🔍 Analyzing speech with Whisper..."):
            text, found_keywords, scam_score = analyze_transcription(audio_path)

        st.success(f"🧩 Prediction: **{predicted_label}** ({confidence:.2f}% confidence)")
        st.progress(int(confidence))

        # --- Display transcription ---
        if text:
            st.subheader("🗣️ Transcription")
            st.write(text)
        else:
            st.warning("No speech detected or Whisper failed to transcribe.")

        # --- Display scam analysis ---
        if found_keywords:
            st.warning(f"⚠️ Scam keywords detected: {', '.join(found_keywords)}")
        st.metric("💰 Scam Risk Score", f"{scam_score:.1f}%")

        # --- Save prediction locally ---
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "filename": [uploaded_file.name],
            "prediction": [predicted_label],
            "confidence": [confidence],
            "scam_score": [scam_score],
            "timestamp": [timestamp]
        }

        df_new = pd.DataFrame(data)
        if os.path.exists("predictions.csv"):
            df_existing = pd.read_csv("predictions.csv")
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv("predictions.csv", index=False)
        st.info("✅ Prediction saved to **predictions.csv**")

        # --- Show history ---
        st.subheader("📜 Prediction History")
        st.dataframe(df_all)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
