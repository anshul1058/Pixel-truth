import os, shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (224, 224)
RAW_DIR = "../data/raw"
OUT_DIR = "../data/processed"

def load_paths(label):
    folder = os.path.join(RAW_DIR, label)
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def resize_and_save(paths, label, split):
    out_folder = os.path.join(OUT_DIR, split, label)
    os.makedirs(out_folder, exist_ok=True)
    for p in tqdm(paths, desc=f"{split}/{label}"):
        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            img.save(os.path.join(out_folder, os.path.basename(p)))
        except Exception as e:
            print(f"Skipped {p}: {e}")

for label in ["real", "fake"]:
    paths = load_paths(label)
    train, temp = train_test_split(paths, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    resize_and_save(train, label, "train")
    resize_and_save(val, label, "val")
    resize_and_save(test, label, "test")

print("Preprocessing done.")