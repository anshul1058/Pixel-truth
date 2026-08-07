import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "../data/processed_v5"

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary")
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary")

print("Class names:", train_ds.class_names)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)

    # random_jpeg_quality needs a single uint8 image (rank 3), not a batch (rank 4),
    # so apply it per-image inside the batch via map_fn, casting dtype around it
    image = tf.cast(image, tf.uint8)
    image = tf.map_fn(
        lambda img: tf.image.random_jpeg_quality(img, min_jpeg_quality=40, max_jpeg_quality=100),
        image,
        fn_output_signature=tf.uint8
    )
    image = tf.cast(image, tf.float32)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print("\n=== Stage 1: training head (base frozen) ===")
model.fit(train_ds, validation_data=val_ds, epochs=6)

base.trainable = True
FINE_TUNE_AT = len(base.layers) - 30
for layer in base.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

print("\n=== Stage 2: fine-tuning top layers ===")
model.fit(train_ds, validation_data=val_ds, epochs=6)

model.save("../models/finetuned_v3.keras")
print("Model saved to ../models/finetuned_v3.keras")
