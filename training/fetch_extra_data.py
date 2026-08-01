from datasets import load_dataset
from itertools import islice
import os
from tqdm import tqdm

REAL_OUT = "../data/raw_v2/real"
FAKE_OUT = "../data/raw_v2/fake"
os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

N_REAL = 6000
N_FAKE = 6000


def save_images(dataset, out_folder, prefix, limit):
    saved = 0
    dataset = dataset.shuffle(seed=42).select(range(min(limit, len(dataset))))
    for i, example in enumerate(tqdm(dataset, desc=f"Saving {prefix}")):
        try:
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(f"{out_folder}/{prefix}_{i}.jpg")
            saved += 1
        except Exception as e:
            print(f"Skipped {prefix} index {i}: {e}")
    return saved


# ---------------------------------------------------------
# REAL PHOTOS — COCO (parquet-based, no legacy script issue)
# ---------------------------------------------------------
print("=== Downloading real photos (COCO) ===")
if not os.listdir(REAL_OUT):
    coco = load_dataset("royokong/coco_test", split="test")
    real_saved = save_images(coco, REAL_OUT, "coco", N_REAL)
    print(f"Saved {real_saved} real photos\n")
else:
    print(f"Skipping — {REAL_OUT} already has files. Delete the folder to re-fetch.\n")


# ---------------------------------------------------------
# FAKE IMAGES — DiffusionDB (streamed, no bulk download)
# ---------------------------------------------------------
print("=== Downloading diffusion-generated fakes (DiffusionDB, streamed) ===")
if not os.listdir(FAKE_OUT):
    diffdb_stream = load_dataset(
        "poloclub/diffusiondb",
        "default",
        revision="refs/convert/parquet",
        split="train",
        streaming=True
    )

    saved = 0
    for i, example in enumerate(tqdm(islice(diffdb_stream, N_FAKE), total=N_FAKE, desc="Saving diffusion")):
        try:
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(f"{FAKE_OUT}/diffusion_{i}.jpg")
            saved += 1
        except Exception as e:
            print(f"Skipped index {i}: {e}")
    print(f"Saved {saved} fake images\n")
else:
    print(f"Skipping — {FAKE_OUT} already has files. Delete the folder to re-fetch.\n")


print("=== Done ===")
print("Real:", len(os.listdir(REAL_OUT)) if os.path.exists(REAL_OUT) else 0)
print("Fake:", len(os.listdir(FAKE_OUT)) if os.path.exists(FAKE_OUT) else 0)