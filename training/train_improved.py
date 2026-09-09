import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import EfficientNetB3, ConvNeXtBase
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = (384, 384)
BATCH_SIZE = 16
DATA_DIR = "../data/processed_v7_balanced"
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 15
MODEL_SAVE_PATH = "../models/gemini_detector_v1.keras"

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

def create_balanced_dataset(data_dir, subset, img_size, batch_size, balance_gemini=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/{subset}",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=(subset == "train")
    )
    
    class_names = ds.class_names
    print(f"{subset} class names: {class_names}")
    
    if subset == "train" and balance_gemini:
        real_ds = ds.filter(lambda x, y: tf.equal(tf.squeeze(y), 1))
        fake_ds = ds.filter(lambda x, y: tf.equal(tf.squeeze(y), 0))
        
        real_count = tf.data.experimental.cardinality(real_ds).numpy() * batch_size
        fake_count = tf.data.experimental.cardinality(fake_ds).numpy() * batch_size
        
        print(f"Real images: ~{real_count}, Fake images: ~{fake_count}")
    
    return ds, class_names

def augment_image(image, label):
    # image shape: (batch, H, W, C) - apply augmentations per image in batch
    def augment_single(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        img = tf.image.random_hue(img, 0.02)
        
        img = tf.cast(img, tf.uint8)
        img = tf.image.random_jpeg_quality(img, min_jpeg_quality=30, max_jpeg_quality=100)
        img = tf.cast(img, tf.float32)
        
        if tf.random.uniform(()) < 0.3:
            img = tf.image.random_crop(img, size=[IMG_SIZE[0], IMG_SIZE[1], 3])
            img = tf.image.resize(img, IMG_SIZE)
        
        return img
    
    image = tf.map_fn(augment_single, image, fn_output_signature=tf.float32)
    return image, label

def mixup(image, label, alpha=0.2):
    batch_size = tf.shape(image)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)
    
    lam = tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha, beta=alpha)
    lam = tf.minimum(lam, 1.0)
    
    mixed_image = lam * image + (1 - lam) * image2
    mixed_label = lam * tf.cast(label, tf.float32)[:, tf.newaxis, tf.newaxis, tf.newaxis] + \
                  (1 - lam) * tf.cast(label2, tf.float32)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    return mixed_image, tf.squeeze(mixed_label, axis=[1, 2])

def build_model(backbone_name='efficientnetb3', img_size=IMG_SIZE):
    if backbone_name == 'efficientnetb3':
        base = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    elif backbone_name == 'convnext_base':
        base = ConvNeXtBase(include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    base.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base

def compile_model(model, learning_rate=1e-3, label_smoothing=0.05):
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    return model

def get_callbacks(model_path):
    return [
        callbacks.ModelCheckpoint(
            model_path, monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=False, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        callbacks.CSVLogger(model_path.replace(".keras", "_training.log")),
    ]

def main():
    print("=" * 60)
    print("PixelTruth - Improved AI Image Detector Training")
    print("=" * 60)
    
    train_ds, class_names = create_balanced_dataset(DATA_DIR, "train", IMG_SIZE, BATCH_SIZE)
    val_ds, _ = create_balanced_dataset(DATA_DIR, "val", IMG_SIZE, BATCH_SIZE, balance_gemini=False)
    
    print(f"\nClass names: {class_names}")
    print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
    
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    print("\n=== Building Model ===")
    model, base = build_model('efficientnetb3', IMG_SIZE)
    model = compile_model(model, learning_rate=1e-3, label_smoothing=0.05)
    model.summary()
    
    print("\n=== Stage 1: Training Head (Base Frozen) ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(MODEL_SAVE_PATH),
        verbose=1
    )
    
    print("\n=== Stage 2: Fine-tuning Top Layers ===")
    base.trainable = True
    fine_tune_at = len(base.layers) - 40
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Fine-tuning from layer {fine_tune_at} onwards ({len(base.layers) - fine_tune_at} layers)")
    
    model = compile_model(model, learning_rate=5e-5, label_smoothing=0.05)
    model.summary()
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
        initial_epoch=EPOCHS_HEAD,
        callbacks=get_callbacks(MODEL_SAVE_PATH),
        verbose=1
    )
    
    print("\n=== Stage 3: Full Fine-tuning (Low LR) ===")
    for layer in base.layers:
        layer.trainable = True
    
    model = compile_model(model, learning_rate=1e-5, label_smoothing=0.0)
    model.summary()
    
    history3 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE + 5,
        initial_epoch=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
        callbacks=get_callbacks(MODEL_SAVE_PATH.replace(".keras", "_full.keras")),
        verbose=1
    )
    
    print(f"\n=== Training Complete ===")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    
    final_model_path = MODEL_SAVE_PATH.replace(".keras", "_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()