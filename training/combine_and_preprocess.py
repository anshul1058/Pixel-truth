import os, random
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (224, 224)
OUT_DIR = "../data/processed_v3"

random.seed(42)

def list_files(folder):
    if not os.path.exists(folder):
        return []
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

# Real: COCO only
real_paths = list_files("../data/raw_v2/real")
random.shuffle(real_paths)

# Fake: force EQUAL counts from each generator family, not random pooled sampling
cifake_paths = list_files("../data/raw/fake")
diffdb_paths = list_files("../data/raw_v2/fake")
random.shuffle(cifake_paths)
random.shuffle(diffdb_paths)

# Balance: take min(available diffusion count, half of real count) from each source
per_source_target = len(real_paths) // 2
n_diffdb = min(per_source_target, len(diffdb_paths))
n_cifake = min(per_source_target, len(cifake_paths))

fake_paths = cifake_paths[:n_cifake] + diffdb_paths[:n_diffdb]
random.shuffle(fake_paths)

# Trim real to match total fake count exactly, keeping classes balanced
real_paths = real_paths[:len(fake_paths)]

print(f"Real: {len(real_paths)} | Fake: {len(fake_paths)} (CIFAKE: {n_cifake}, DiffusionDB: {n_diffdb})")

for label, paths in [("real", real_paths), ("fake", fake_paths)]:
    train, temp = train_test_split(paths, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    resize_and_save(train, label, "train")
    resize_and_save(val, label, "val")
    resize_and_save(test, label, "test")

print("Combined preprocessing done (v3, source-balanced).")
