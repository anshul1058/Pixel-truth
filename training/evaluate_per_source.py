import os
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

IMG_SIZE = (384, 384)
MODEL_PATH = "../models/gemini_detector_v1.keras"
DATA_DIR = "../data/processed_v7/val"

def get_source_from_filename(filename):
    fname = filename.lower()
    if 'gemini' in fname:
        return 'gemini'
    elif 'midjourney' in fname or 'mj_' in fname or 'genimgmj' in fname:
        return 'midjourney'
    elif 'cifake' in fname:
        return 'cifake'
    elif 'diffdb' in fname or 'diffusion' in fname:
        return 'stable_diffusion'
    elif 'genimgsd' in fname:
        return 'stable_diffusion'
    return 'unknown'

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def evaluate_per_source(model, data_dir):
    sources = {}
    
    for class_name in ['fake', 'real']:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            source = get_source_from_filename(fname)
            if source == 'unknown' and class_name == 'fake':
                continue
            if source == 'unknown':
                source = 'real'
            
            if source not in sources:
                sources[source] = {'y_true': [], 'y_pred': [], 'y_scores': []}
            
            img_path = os.path.join(class_dir, fname)
            try:
                arr = preprocess_image(img_path)
                score = float(model.predict(arr, verbose=0)[0][0])
                pred = 1 if score > 0.5 else 0
                true_label = 1 if class_name == 'real' else 0
                
                sources[source]['y_true'].append(true_label)
                sources[source]['y_pred'].append(pred)
                sources[source]['y_scores'].append(score)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
    results = {}
    for source, data in sources.items():
        y_true = np.array(data['y_true'])
        y_pred = np.array(data['y_pred'])
        y_scores = np.array(data['y_scores'])
        
        if len(y_true) == 0:
            continue
            
        acc = np.mean(y_true == y_pred)
        auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[source] = {
            'count': len(y_true),
            'accuracy': float(acc),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
        }
        
        print(f"\n{source.upper()} ({len(y_true)} samples):")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return results

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")
    
    print(f"\nEvaluating on {DATA_DIR}...")
    results = evaluate_per_source(model, DATA_DIR)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for source, metrics in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f"{source:20s} | AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | N={metrics['count']}")
    
    output_path = MODEL_PATH.replace(".keras", "_per_source_eval.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()