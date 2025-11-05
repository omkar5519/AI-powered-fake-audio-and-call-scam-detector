import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
import os
import datetime
import pandas as pd
from pymongo import MongoClient
import whisper

# ---------------- CONFIG ----------------
SCAM_KEYWORDS = [
    "otp", "bank", "lottery", "payment", "insurance", "refund",
    "reward", "transaction", "prize", "investment", "claim",
    "credit card", "account", "loan", "id verification", "upi", "gift"
]

# ---------------- STREAMLIT SETUP ----------------
st.title("🎙️ Real / Fake / Scam Voice Detector")
st.caption("Detects AI voices, scam calls, and real speech using Deep Learning + Whisper")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

# ---------------- HELPER FUNCTIONS ----------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("files/audio_cnn_rnn_model.h5")
    with open("files/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # or "small" for faster inference

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

def analyze_transcription(audio_path, whisper_model):
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
        text = result["text"].lower()
        found_keywords = [kw for kw in SCAM_KEYWORDS if kw in text]
        scam_score = min(len(found_keywords), 5) / 5.0 * 100
        return text.strip(), found_keywords, scam_score
    except Exception as e:
        return "", [], 0

def save_to_mongo(data):
    try:
        client = MongoClient(os.environ.get("MONGODB_URI"))
        db = client["fake_audio_db"]
        collection = db["predictions"]
        collection.insert_one(data)
        return True
    except Exception as e:
        return False

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    audio_path = os.path.join("uploads", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔄 Loading models..."):
        model, encoder = load_model_and_encoder()
        whisper_model = load_whisper_model()

    # Prediction
    try:
        features = extract_features(audio_path)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction) * 100)

        # Whisper transcription + scam detection
        with st.spinner("🔍 Analyzing speech with Whisper..."):
            text, found_keywords, scam_score = analyze_transcription(audio_path, whisper_model)

        # Display results
        st.success(f"🧩 Prediction: **{predicted_label}** ({confidence:.2f}% confidence)")
        st.progress(int(confidence))

        if text:
            st.subheader("🗣️ Transcription")
            st.write(text)
        if found_keywords:
            st.warning(f"⚠️ Scam keywords detected: {', '.join(found_keywords)}")
        st.metric("💰 Scam Risk Score", f"{scam_score:.1f}%")

        # Save locally
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "filename": uploaded_file.name,
            "prediction": predicted_label,
            "confidence": confidence,
            "scam_score": scam_score,
            "transcription": text,
            "found_keywords": found_keywords,
            "timestamp": timestamp
        }

        df_new = pd.DataFrame([data])
        if os.path.exists("predictions.csv"):
            df_existing = pd.read_csv("predictions.csv")
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv("predictions.csv", index=False)

        # Save to MongoDB Atlas safely
        if save_to_mongo(data):
            st.info("📦 Prediction saved to MongoDB Atlas successfully!")
        else:
            st.warning("⚠️ Could not save to MongoDB Atlas. Check your connection.")

        st.subheader("📜 Prediction History")
        st.dataframe(df_all)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
