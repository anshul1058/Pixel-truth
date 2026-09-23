#!/usr/bin/env python3
"""Download models and training data for PixelTruth."""

import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def install_deps():
    """Install required packages."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "huggingface_hub", "datasets", "tqdm", "Pillow"])


def download_model():
    """Download the trained model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    model_dir = os.path.join(ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    repo = "Bombek1/ai-image-detector-siglip-dinov2"
    filename = "pytorch_model.pt"

    dest = os.path.join(model_dir, filename)
    if os.path.exists(dest):
        print(f"[OK] Model already exists: {dest}")
        return

    print(f"Downloading model from {repo}...")
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir)
    print(f"[OK] Model saved to: {path}")


def download_cifake():
    """Download CIFAKE dataset (real + fake images)."""
    from datasets import load_dataset
    from PIL import Image

    out_dir = os.path.join(ROOT, "data", "raw")
    real_dir = os.path.join(out_dir, "real")
    fake_dir = os.path.join(out_dir, "fake")

    if os.path.exists(real_dir) and os.listdir(real_dir):
        print(f"[SKIP] CIFAKE already downloaded: {out_dir}")
        return

    print("Downloading CIFAKE dataset...")
    ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i, example in enumerate(ds["train"]):
        img = example["image"]
        label = example["label"]
        folder = real_dir if label == 0 else fake_dir
        img.save(os.path.join(folder, f"img_{i}.png"))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/60000 images saved...")

    print(f"[OK] CIFAKE saved to {out_dir}")


def download_coco_diffusion():
    """Download COCO real photos + DiffusionDB fakes."""
    from datasets import load_dataset
    from itertools import islice
    from tqdm import tqdm

    real_dir = os.path.join(ROOT, "data", "raw_v2", "real")
    fake_dir = os.path.join(ROOT, "data", "raw_v2", "fake")
    n = 6000

    if os.path.exists(real_dir) and os.listdir(real_dir):
        print(f"[SKIP] COCO already downloaded: {real_dir}")
    else:
        print("Downloading COCO real photos...")
        os.makedirs(real_dir, exist_ok=True)
        coco = load_dataset("royokong/coco_test", split="test")
        coco = coco.shuffle(seed=42).select(range(min(n, len(coco))))
        for i, ex in enumerate(tqdm(coco, desc="COCO")):
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(real_dir, f"coco_{i}.jpg"))
        print(f"[OK] COCO saved ({len(os.listdir(real_dir))} images)")

    if os.path.exists(fake_dir) and os.listdir(fake_dir):
        print(f"[SKIP] DiffusionDB already downloaded: {fake_dir}")
    else:
        print("Downloading DiffusionDB fakes...")
        os.makedirs(fake_dir, exist_ok=True)
        stream = load_dataset(
            "poloclub/diffusiondb", "default",
            revision="refs/convert/parquet", split="train", streaming=True,
        )
        for i, ex in enumerate(tqdm(islice(stream, n), total=n, desc="DiffusionDB")):
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(fake_dir, f"diffusion_{i}.jpg"))
        print(f"[OK] DiffusionDB saved ({len(os.listdir(fake_dir))} images)")


def download_midjourney():
    """Download Midjourney-generated images."""
    from datasets import load_dataset
    from tqdm import tqdm

    out_dir = os.path.join(ROOT, "data", "raw_v3", "fake_midjourney")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"[SKIP] Midjourney already downloaded: {out_dir}")
        return

    print("Downloading Midjourney images...")
    os.makedirs(out_dir, exist_ok=True)
    mj = load_dataset(
        "ehristoforu/midjourney-images",
        revision="refs/convert/parquet", split="train",
    )
    mj = mj.shuffle(seed=42).select(range(min(3000, len(mj))))
    for i, ex in enumerate(tqdm(mj, desc="Midjourney")):
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(out_dir, f"midjourney_{i}.jpg"))
    print(f"[OK] Midjourney saved ({len(os.listdir(out_dir))} images)")


def download_all_data():
    """Download all training datasets."""
    download_cifake()
    download_coco_diffusion()
    download_midjourney()
    print("\n[DONE] All datasets downloaded to data/")


def main():
    parser = argparse.ArgumentParser(description="PixelTruth setup: download models and data")
    parser.add_argument("target", nargs="?", default="all",
                        choices=["model", "data", "all"],
                        help="what to download (default: all)")
    args = parser.parse_args()

    install_deps()

    if args.target in ("model", "all"):
        download_model()
    if args.target in ("data", "all"):
        download_all_data()


if __name__ == "__main__":
    main()
