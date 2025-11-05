"""
train_model.py
---------------------------------
Train a CNN-BiLSTM model to classify:
    - real human voice
    - fake / AI / cloned voice
    - scam voice (real human voice with scam keywords)
    - background (noise / non-speech)

Optimized for Apple Silicon (M1/M2) with tensorflow-macos + tensorflow-metal.
"""

import os
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten,
    TimeDistributed, Bidirectional, LSTM, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf
import pickle
import random
import whisper

# ---------------- CONFIG ----------------
DATASET_PATH = "files/dataset"   # subfolders: real/, fake/, scam/, background/
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
MODEL_SAVE_PATH = "files/audio_cnn_rnn_model.h5"
ENCODER_SAVE_PATH = "files/label_encoder.pkl"

SCAM_KEYWORDS = [
    "otp", "bank", "lottery", "payment", "insurance", "refund", "reward",
    "transaction", "prize", "investment", "claim", "credit card", "account"
]

# ---------------- GPU SETTINGS ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU memory growth enabled")
    except Exception as e:
        print("⚠️ GPU config failed:", e)

# ---------------- AUDIO LOADER ----------------
def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and normalize audio file to fixed duration and sample rate."""
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)

        samples = samples / (2 ** (8 * audio.sample_width - 1))
        samples = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=sr)

        max_len = sr * duration
        if len(samples) > max_len:
            samples = samples[:max_len]
        else:
            samples = np.pad(samples, (0, max_len - len(samples)))

        return samples
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path, scam_weight=0):
    """Extract MFCCs and ensure exactly 2 feature channels."""
    audio = load_audio(file_path)
    if audio is None:
        return None

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mfcc = np.expand_dims(mfcc, axis=-1)

        # always add scam score as 2nd channel
        scam_plane = np.ones_like(mfcc) * scam_weight
        mfcc = np.concatenate([mfcc, scam_plane], axis=-1)

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"⚠️ Feature extraction failed for {file_path}: {e}")
        return None

# ---------------- SCAM DETECTION ----------------
print("🧩 Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")

def get_scam_score(file_path):
    """Transcribe audio and detect scam keyword presence."""
    try:
        result = whisper_model.transcribe(file_path, fp16=False)
        text = result["text"].lower()
        score = sum(kw in text for kw in SCAM_KEYWORDS)
        return min(score, 5) / 5.0  # normalize 0–1
    except Exception:
        return 0

# ---------------- DATASET LOADER ----------------
def load_dataset(dataset_path):
    """Load and extract features from dataset."""
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))
    print("📂 Loading dataset...")

    for label in classes:
        class_folder = os.path.join(dataset_path, label)
        if not os.path.isdir(class_folder):
            continue

        files = [
            os.path.join(root, f)
            for root, _, fs in os.walk(class_folder)
            for f in fs
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))
        ]

        # balance background up to 1200 samples
        if label.lower() == "background" and len(files) > 1200:
            files = random.sample(files, 1200)

        print(f"📁 {label}: {len(files)} files")

        for file_path in tqdm(files, desc=f"Extracting {label}"):
            scam_weight = get_scam_score(file_path) if label.lower() == "scam" else 0
            feat = extract_features(file_path, scam_weight)
            if feat is not None:
                X.append(feat)
                y.append(label)

    print(f"✅ Loaded {len(X)} samples across {len(set(y))} classes.")
    print("🔍 Example feature shape:", X[0].shape)
    return np.array(X), np.array(y)

# ---------------- MODEL ARCHITECTURE ----------------
def build_cnn_bilstm_model(input_shape, num_classes):
    """CNN + BiLSTM hybrid model."""
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------- TRAINING ----------------
if __name__ == "__main__":
    print("🚀 Starting training pipeline...")
    X, y = load_dataset(DATASET_PATH)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    print("🧠 Building CNN-BiLSTM model...")
    model = build_cnn_bilstm_model(X_train.shape[1:], y_cat.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]

    print("🎯 Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=callbacks
    )

    print("💾 Saving model and label encoder...")
    os.makedirs("files", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    with open(ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"✅ Model saved at: {MODEL_SAVE_PATH}")
    print(f"✅ Encoder saved at: {ENCODER_SAVE_PATH}")
    print("🎉 Training complete!")
