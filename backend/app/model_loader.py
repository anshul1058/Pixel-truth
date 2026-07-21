import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/baseline_v0.keras")
IMG_SIZE = (224, 224)

# Confirmed from training: train_ds.class_names == ['fake', 'real']
# So model output close to 0 -> fake, close to 1 -> real
CLASS_NAMES = ["fake", "real"]

_model = None

def get_model():
    global _model
    if _model is None:
        if MODEL_SOURCE == "local":
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH)
        elif MODEL_SOURCE == "vertex":
            # Placeholder for Part 2 (GCP) — call a Vertex AI Endpoint instead
            raise NotImplementedError("Vertex AI serving not wired up yet — set MODEL_SOURCE=local for now")
        else:
            raise ValueError(f"Unknown MODEL_SOURCE: {MODEL_SOURCE}")
    return _model

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)          # raw 0-255, matches training pipeline
    arr = np.expand_dims(arr, axis=0)                # add batch dimension -> (1, 224, 224, 3)
    return arr

def predict(image_bytes: bytes):
    model = get_model()
    arr = preprocess_image(image_bytes)
    raw_score = float(model.predict(arr, verbose=0)[0][0])   # sigmoid output, 0-1

    predicted_idx = 1 if raw_score > 0.5 else 0
    label = CLASS_NAMES[predicted_idx]
    confidence = raw_score if predicted_idx == 1 else 1 - raw_score

    return label, round(confidence, 4)