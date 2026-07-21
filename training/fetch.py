# pyrefly: ignore [missing-import]
from datasets import load_dataset
import os

dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")

# Check label mapping first before assuming 0=real, 1=fake
print(dataset["train"].features)

os.makedirs("../data/raw/real", exist_ok=True)
os.makedirs("../data/raw/fake", exist_ok=True)

for i, example in enumerate(dataset["train"]):
    img = example["image"]
    label = example["label"]
    folder = "../data/raw/real" if label == 0 else "../data/raw/fake"
    img.save(f"{folder}/img_{i}.png")

print("Done saving images locally.")