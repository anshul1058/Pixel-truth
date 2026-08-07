from datasets import load_dataset
import os
from tqdm import tqdm

OUT_DIR = "../data/raw_v3/fake_midjourney"
os.makedirs(OUT_DIR, exist_ok=True)

N_MIDJOURNEY = 3000

print("Downloading Midjourney-generated images (parquet branch)...")
mj = load_dataset(
    "ehristoforu/midjourney-images",
    revision="refs/convert/parquet",
    split="train"
)
mj = mj.shuffle(seed=42).select(range(min(N_MIDJOURNEY, len(mj))))

saved = 0
for i, example in enumerate(tqdm(mj, desc="Saving midjourney")):
    try:
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(f"{OUT_DIR}/midjourney_{i}.jpg")
        saved += 1
    except Exception as e:
        print(f"Skipped index {i}: {e}")

print(f"Saved {saved} Midjourney images")
