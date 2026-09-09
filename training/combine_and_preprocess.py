import os, random
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (224, 224)
OUT_DIR = "../data/processed_v7"
GEMINI_OVERSAMPLE = 4  # repeat each Gemini image N times in TRAIN split only

random.seed(42)

def list_files_tagged(folder, tag):
    if not os.path.exists(folder):
        print(f"WARNING: folder not found: {folder}")
        return []
    return [(os.path.join(folder, f), tag) for f in os.listdir(folder)]

def resize_and_save(items, label, split):
    out_folder = os.path.join(OUT_DIR, split, label)
    os.makedirs(out_folder, exist_ok=True)
    counters = {}
    for p, tag in tqdm(items, desc=f"{split}/{label}"):
        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            counters[tag] = counters.get(tag, 0) + 1
            new_name = f"{tag}_{counters[tag]}_{os.path.basename(p)}"
            img.save(os.path.join(out_folder, new_name))
        except Exception as e:
            print(f"Skipped {p}: {e}")

real_items = list_files_tagged("../data/raw_v2/real", "coco") + \
             list_files_tagged("../data/raw_genimage/real_pool", "genimgreal")
random.shuffle(real_items)

gan_items = list_files_tagged("../data/raw/fake", "cifake") + \
            list_files_tagged("../data/raw_genimage/gan_pool", "genimggan")
sd_items = list_files_tagged("../data/raw_v2/fake", "diffdb") + \
           list_files_tagged("../data/raw_genimage/sd_pool", "genimgsd")
mj_items = list_files_tagged("../data/raw_v3/fake_midjourney", "mj") + \
           list_files_tagged("../data/raw_genimage/mj_pool", "genimgmj")
gemini_items = list_files_tagged("../data/raw_v3/fake_gemini", "gemini")
random.shuffle(gan_items)
random.shuffle(sd_items)
random.shuffle(mj_items)
random.shuffle(gemini_items)

# Keep GAN/SD/MJ at the same big scale as before (v5) — don't shrink them
per_source_target = 3666
n_gan = min(per_source_target, len(gan_items))
n_sd = min(per_source_target, len(sd_items))
n_mj = min(per_source_target, len(mj_items))

# Split gemini 70/15/15 first, then oversample ONLY the train portion
gem_train, gem_temp = train_test_split(gemini_items, test_size=0.3, random_state=42)
gem_val, gem_test = train_test_split(gem_temp, test_size=0.5, random_state=42)
gem_train_oversampled = gem_train * GEMINI_OVERSAMPLE

gan_train, gan_temp = train_test_split(gan_items[:n_gan], test_size=0.3, random_state=42)
gan_val, gan_test = train_test_split(gan_temp, test_size=0.5, random_state=42)
sd_train, sd_temp = train_test_split(sd_items[:n_sd], test_size=0.3, random_state=42)
sd_val, sd_test = train_test_split(sd_temp, test_size=0.5, random_state=42)
mj_train, mj_temp = train_test_split(mj_items[:n_mj], test_size=0.3, random_state=42)
mj_val, mj_test = train_test_split(mj_temp, test_size=0.5, random_state=42)

fake_train = gan_train + sd_train + mj_train + gem_train_oversampled
fake_val = gan_val + sd_val + mj_val + gem_val
fake_test = gan_test + sd_test + mj_test + gem_test
random.shuffle(fake_train); random.shuffle(fake_val); random.shuffle(fake_test)

real_train, real_temp = train_test_split(real_items, test_size=0.3, random_state=42)
real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
real_train = real_train[:len(fake_train)]
real_val = real_val[:len(fake_val)]
real_test = real_test[:len(fake_test)]

print(f"Train: real={len(real_train)} fake={len(fake_train)} (gemini x{GEMINI_OVERSAMPLE}={len(gem_train_oversampled)})")
print(f"Val:   real={len(real_val)} fake={len(fake_val)} (gemini={len(gem_val)})")
print(f"Test:  real={len(real_test)} fake={len(fake_test)} (gemini={len(gem_test)})")

resize_and_save(real_train, "real", "train")
resize_and_save(fake_train, "fake", "train")
resize_and_save(real_val, "real", "val")
resize_and_save(fake_val, "fake", "val")
resize_and_save(real_test, "real", "test")
resize_and_save(fake_test, "fake", "test")

print("Combined preprocessing done (v7, big dataset + oversampled Gemini).")
