from google import genai
from PIL import Image
from io import BytesIO
import os
import time
import random

client = genai.Client()

OUT_DIR = "../data/raw_v3/fake_gemini"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Diagnostic: list models actually available to this API key/project ---
print("=== Checking available models ===")
try:
    for m in client.models.list():
        name = getattr(m, "name", "")
        if "image" in name.lower():
            print(name)
except Exception as e:
    print(f"Could not list models: {e}")
print("===================================\n")

N_IMAGES = 150
MODEL_ID = "gemini-2.5-flash-image"  # stable, non-preview ID — free tier confirmed for this one

SUBJECTS = [
    "a person standing in an office", "a person walking on a city street",
    "a portrait of a young professional", "a group of friends at a cafe",
    "a woman reading a book at home", "a man cooking in a kitchen",
    "a child playing in a park", "an elderly couple walking a dog",
]
SCENES = [
    "a modern living room", "a busy city intersection", "a quiet beach at sunset",
    "a mountain hiking trail", "a cozy coffee shop interior", "a university campus",
    "a suburban backyard", "a downtown skyline at dusk",
]
OBJECTS = [
    "a bowl of fresh fruit on a table", "a laptop on a wooden desk",
    "a bicycle parked against a wall", "a bouquet of flowers in a vase",
    "a plate of food at a restaurant", "a car parked on a street",
]

ALL_PROMPTS = SUBJECTS + SCENES + OBJECTS

saved = 0
attempts = 0
max_attempts = N_IMAGES * 2

while saved < N_IMAGES and attempts < max_attempts:
    attempts += 1
    prompt = f"A photorealistic photo of {random.choice(ALL_PROMPTS)}, natural lighting, high detail"

    try:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        image_bytes = next(
            (p.inline_data.data for p in resp.candidates[0].content.parts
             if getattr(p, "inline_data", None)),
            None,
        )
        if image_bytes:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img.save(f"{OUT_DIR}/gemini_{saved}.png")
            saved += 1
            print(f"[{saved}/{N_IMAGES}] saved: {prompt[:60]}")
        else:
            print(f"No image returned for prompt: {prompt[:60]}")

    except Exception as e:
        print(f"Error on attempt {attempts}: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("Hit quota/rate limit — stopping early rather than burning more attempts.")
            break
        time.sleep(3)

    time.sleep(1)

print(f"\nDone. Saved {saved}/{N_IMAGES} images to {OUT_DIR}")
