import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image

MODEL_PATH = "../models/finetuned_v5.keras"
TEST_DIR = "../data/processed_v7/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False
)

class_names = test_ds.class_names
print("Class names:", class_names)

y_true, y_pred_probs = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().flatten().tolist())
    y_pred_probs.extend(preds.flatten().tolist())

y_true = np.array(y_true)
y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

print("\n=== Overall Test Set Metrics ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix (rows=true, cols=predicted):")
print(f"           {class_names[0]:>10} {class_names[1]:>10}")
for i, row in enumerate(cm):
    print(f"{class_names[i]:>10} {row[0]:>10} {row[1]:>10}")
print(f"\nOverall test accuracy: {(y_pred == y_true).mean():.4f}")

print("\n=== Breakdown by fake source (filename prefix, before 2nd underscore) ===")
fake_dir = os.path.join(TEST_DIR, "fake")
filenames = sorted(os.listdir(fake_dir))
fake_label_idx = class_names.index("fake")

def predict_single(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    score = float(model.predict(arr, verbose=0)[0][0])
    return 1 if score > 0.5 else 0

counts = {}
for fname in filenames:
    tag = fname.split("_")[0]  # coco/cifake/diffdb/mj/genimgreal/genimggan/genimgsd/genimgmj/gemini
    path = os.path.join(fake_dir, fname)
    pred = predict_single(path)
    correct = int(pred == fake_label_idx)
    if tag not in counts:
        counts[tag] = [0, 0]
    counts[tag][0] += correct
    counts[tag][1] += 1

for tag, (c, t) in sorted(counts.items()):
    print(f"{tag:>12}: {c}/{t} ({c/t:.2%})")

print("\nDone.")
