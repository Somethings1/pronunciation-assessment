import tensorflow as tf
import numpy as np
import librosa
import io
import json
import os
from server.utils.audio_features import extract_full_features


class StressEvaluator:
    def __init__(self, model_path, scaler_path):
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

        print(f"Loading Stress Model from {model_path}...")
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("✅ Stress Model loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load Stress model: {e}")

        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                data = json.load(f)
                self.scaler_mean = np.array(data["mean"])
                self.scaler_std = np.array(data["std"])

    def softmax(self, x):
        """Hàm biến điểm số thấp lè tè thành xác suất % dễ nhìn"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, audio_bytes, word_text):
        if not self.model:
            return {"error": "Model chưa load"}

        # 1. Load Audio
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        except Exception:
            return {"error": "File audio lỗi"}

        y_trimmed, index = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=128)

        if len(y_trimmed) < 1600:
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        y_normalized = librosa.util.normalize(y_trimmed)

        # 2. Extract Features
        features = extract_full_features(y_normalized, sr, word_text)
        num_syl = features.shape[1]
        if num_syl == 0:
            return {"error": "Không tách được âm tiết"}

        # 3. Normalize Features (StandardScaler)
        if self.scaler_mean is not None:
            features = (features - self.scaler_mean) / (self.scaler_std + 1e-7)

        # 4. Inference
        pred = self.model.predict(features, verbose=0)
        scores = np.squeeze(pred)
        if scores.ndim == 0:
            scores = np.array([scores])

        # Fix lỗi length mismatch
        if len(scores) > num_syl:
            scores = scores[:num_syl]
        elif len(scores) < num_syl:
            scores = np.pad(scores, (0, num_syl - len(scores)))

        probs = self.softmax(scores)

        stress_index = int(np.argmax(probs))

        return {
            "word": word_text,
            "detected_syllables_count": num_syl,
            "stress_index": stress_index,
            "stress_probability": float(probs[stress_index]),
            "raw_scores": probs.tolist()
        }
