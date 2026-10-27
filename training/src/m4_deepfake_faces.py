# %% [markdown]
# # M4 · Deepfake Face Detection (real vs AI-generated faces)
# **Model:** CNN (Keras), trained from scratch on 128×128 face images
# **Dataset:** 140k Real and Fake Faces: 70k real faces (Flickr-Faces-HQ) + 70k fake faces (StyleGAN)
#
# This replaces the CIFAKE model: CIFAKE contains 32×32 pictures of objects/animals, so it fails on real
# photos of people. This model learns from faces, which is what a deepfake detector should see.
#
# **How to run on Kaggle**
# 1. **Add Input** → search *"140k Real and Fake Faces"* (by xhlulu) → add.
# 2. Settings → **Accelerator: GPU T4**.
# 3. *Run All* (~20-25 min) → download `m4_deepfake_cnn.keras` from Output →
#    replace the old file in `app/models/baseline/`.

# %%
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score

print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

IMG_SIZE = 128
BATCH = 64
EPOCHS = 12
TRAIN_PER_CLASS = 20_000   # 40k training images in total; increase if you have time
EVAL_PER_CLASS = 5_000
CLASS_NAMES = ["REAL", "FAKE"]  # REAL = 0, FAKE = 1
random.seed(42)


def find_root():
    for root in ["/kaggle/input", "/content", "."]:
        hits = glob.glob(f"{root}/**/train/real", recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("'train/real' folder not found. Did you add '140k Real and Fake Faces'?")


ROOT = find_root()
print("Dataset root:", ROOT, "| splits:", os.listdir(ROOT))

# %% [markdown]
# ## 1. Build balanced file lists (train / valid / test)
# The dataset already has separate train, valid and test folders, so test faces are never seen in training.

# %%
def file_list(split, per_class):
    paths, labels = [], []
    for label, cls in enumerate(["real", "fake"]):
        files = sorted(glob.glob(os.path.join(ROOT, split, cls, "*")))
        random.shuffle(files)
        files = files[:per_class]
        paths += files
        labels += [label] * len(files)
        print(f"{split:5s} {cls:4s}: {len(files)} images")
    return paths, labels


train_p, train_y = file_list("train", TRAIN_PER_CLASS)
val_p, val_y = file_list("valid", EVAL_PER_CLASS)
test_p, test_y = file_list("test", EVAL_PER_CLASS)


def load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, tf.cast(label, tf.float32)


def make_ds(paths, labels, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_p, train_y, shuffle=True)
val_ds = make_ds(val_p, val_y, shuffle=False)
test_ds = make_ds(test_p, test_y, shuffle=False)

# %%
images, labels = next(iter(train_ds))
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(CLASS_NAMES[int(labels[i])])
    plt.axis("off")

# %% [markdown]
# ## 2. CNN model
# 4 blocks of `Conv2D → BatchNorm → MaxPool`, then global average pooling and a dense head.
# The last conv layer is named **`last_conv`** for Grad-CAM. `Rescaling` and light augmentation are inside the
# model, so the app can send raw images of any size (it resizes them to 128×128).

# %%
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.RandomFlip("horizontal")(x)          # active only during training
x = layers.RandomBrightness(0.1, value_range=(0, 1))(x)
for filters in (32, 64, 128):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # P(FAKE)
model = keras.Model(inputs, outputs, name="aices_deepfake_face_cnn")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

# %%
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)],
)

# %%
h = history.history
plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1); plt.plot(h["accuracy"], label="train"); plt.plot(h["val_accuracy"], label="val"); plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(h["loss"], label="train"); plt.plot(h["val_loss"], label="val"); plt.title("Loss"); plt.legend()

# %% [markdown]
# ## 3. Test set evaluation (numbers for the report)

# %%
y_true = np.array(test_y)
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_true, y_prob), 3))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES);

# %% [markdown]
# ## 4. Grad-CAM demo: where did the model look?

# %%
grad_model = keras.Model(model.inputs, [model.get_layer("last_conv").output, model.output])


def gradcam(img_batch):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    return heat / (heat.max() + 1e-8)


images, labels = next(iter(test_ds.unbatch().shuffle(500, seed=1).batch(6)))
plt.figure(figsize=(12, 4.5))
for i in range(6):
    img = images[i:i + 1]
    heat = tf.image.resize(gradcam(img)[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    p = float(model.predict(img, verbose=0)[0][0])
    plt.subplot(2, 6, i + 1); plt.imshow(img[0].numpy().astype("uint8")); plt.axis("off")
    plt.title(f"true {CLASS_NAMES[int(labels[i])]}")
    plt.subplot(2, 6, i + 7); plt.imshow(img[0].numpy().astype("uint8")); plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off"); plt.title(f"P(fake)={p:.2f}")

# %%
model.save("m4_deepfake_cnn.keras")
print("Saved m4_deepfake_cnn.keras → replace the old file in app/models/baseline/")
