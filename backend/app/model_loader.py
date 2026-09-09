import os
from huggingface_hub import hf_hub_download

MODEL_REPO = os.getenv("MODEL_REPO", "Bombek1/ai-image-detector-siglip-dinov2")
MODEL_FILE = "pytorch_model.pt"

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[PixelTruth] Loading model from HuggingFace: {MODEL_REPO}")

        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        print(f"[PixelTruth] Weights downloaded to: {weights_path}")

        from app.model import AIImageDetector
        _model = AIImageDetector(weights_path, device="cpu")

        print(f"[PixelTruth] Model loaded successfully on {_model.device}.")
    return _model


def predict(image_bytes: bytes):
    model = get_model()
    result = model.predict(image_bytes)
    return result["prediction"], round(result["confidence"], 4)
