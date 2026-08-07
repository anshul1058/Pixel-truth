import os, random
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (224, 224)
OUT_DIR = "../data/processed_v5"

random.seed(42)

def list_files_tagged(folder, tag):
    if not os.path.exists(folder):
        print(f"WARNING: folder not found: {folder}")
        return []
    return [(os.path.join(folder, f), tag) for f in os.listdir(folder)]

def resize_and_save(items, label, split):
    out_folder = os.path.join(OUT_DIR, split, label)
    os.makedirs(out_folder, exist_ok=True)
    for p, tag in tqdm(items, desc=f"{split}/{label}"):
        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            new_name = f"{tag}_{os.path.basename(p)}"
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
random.shuffle(gan_items)
random.shuffle(sd_items)
random.shuffle(mj_items)

per_source_target = len(real_items) // 3
n_gan = min(per_source_target, len(gan_items))
n_sd = min(per_source_target, len(sd_items))
n_mj = min(per_source_target, len(mj_items))

fake_items = gan_items[:n_gan] + sd_items[:n_sd] + mj_items[:n_mj]
random.shuffle(fake_items)

real_items = real_items[:len(fake_items)]

print(f"Real: {len(real_items)} | Fake: {len(fake_items)} "
      f"(GAN: {n_gan}, Diffusion/SD: {n_sd}, Midjourney: {n_mj})")

for label, items in [("real", real_items), ("fake", fake_items)]:
    train, temp = train_test_split(items, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    resize_and_save(train, label, "train")
    resize_and_save(val, label, "val")
    resize_and_save(test, label, "test")

print("Combined preprocessing done (v5, collision-safe).")
