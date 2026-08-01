import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

MODEL_PATH = "../models/finetuned_v1.keras"
TEST_DIR = "../data/processed_v3/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False  # keep order so we can map predictions back to filenames
)

class_names = test_ds.class_names
print("Class names:", class_names)

# Collect true labels and predictions
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().flatten().tolist())
    y_pred_probs.extend(preds.flatten().tolist())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n=== Overall Test Set Metrics ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("Confusion Matrix (rows=true, cols=predicted):")
print(f"           {class_names[0]:>10} {class_names[1]:>10}")
cm = confusion_matrix(y_true, y_pred)
for i, row in enumerate(cm):
    print(f"{class_names[i]:>10} {row[0]:>10} {row[1]:>10}")

overall_acc = (y_pred == y_true).mean()
print(f"\nOverall test accuracy: {overall_acc:.4f}")

# Break down accuracy by fake-image source (CIFAKE vs DiffusionDB), since combine_and_preprocess.py
# renamed files with distinguishing prefixes we can check
print("\n=== Breakdown by fake source (filename prefix) ===")
fake_dir = os.path.join(TEST_DIR, "fake")
filenames = sorted(os.listdir(fake_dir))  # matches shuffle=False iteration order

cifake_correct, cifake_total = 0, 0
diffdb_correct, diffdb_total = 0, 0

fake_idx_start = list(y_true).index(class_names.index("fake")) if "fake" in class_names else None
# Simpler: just re-run predictions per-file directly for the breakdown, avoids index-matching complexity
from PIL import Image

def predict_single(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    score = float(model.predict(arr, verbose=0)[0][0])
    return 1 if score > 0.5 else 0  # 1 = real, 0 = fake (per class_names order)

fake_label_idx = class_names.index("fake")

for fname in filenames:
    path = os.path.join(fake_dir, fname)
    pred = predict_single(path)
    correct = (pred == fake_label_idx)
    if fname.startswith("img_"):  # original CIFAKE naming
        cifake_total += 1
        cifake_correct += correct
    elif fname.startswith("diffusion_"):
        diffdb_total += 1
        diffdb_correct += correct

if cifake_total:
    print(f"CIFAKE (GAN) fakes correctly caught:      {cifake_correct}/{cifake_total} ({cifake_correct/cifake_total:.2%})")
if diffdb_total:
    print(f"DiffusionDB (diffusion) fakes caught:      {diffdb_correct}/{diffdb_total} ({diffdb_correct/diffdb_total:.2%})")

print("\nDone.")
