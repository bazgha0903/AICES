# %% [markdown]
# # M4-B · General AI-Generated Image Detection (any photo, not only faces)
# **Model:** EfficientNet-B0 (transfer learning from ImageNet), fine-tuned, 224×224 input
# **Dataset:** AI vs. Human-Generated Images (Women in AI / Kaggle 2025): real Shutterstock photos paired with
# AI-generated versions of the same scene: people, animals, food, objects, landscapes, art.
#
# The face model (M4) only knows human faces. This model learns the general difference between camera photos
# and AI-generated images, so the app uses it for every image without a face.
#
# **How to run on Kaggle**
# 1. **Add Input** → search *"AI vs. Human-Generated Images"* (by alessandrasala79) → add.
# 2. Settings → **Accelerator: GPU T4** and **Internet: ON** (to download the ImageNet weights).
# 3. *Run All* (~40-50 min) → download `m4b_ai_image_general.keras` from Output →
#    copy it to `app/models/baseline/`.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

IMG_SIZE = 224
BATCH = 64
PER_CLASS = 15_000          # 30k images in total (the dataset has ~80k); increase if you have time
EPOCHS_HEAD, EPOCHS_FINE = 4, 6
CLASS_NAMES = ["REAL", "FAKE"]  # REAL = human photo (0), FAKE = AI-generated (1)


def find_csv():
    hits = [p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)]
    if not hits:
        raise FileNotFoundError("train.csv not found. Did you add 'AI vs. Human-Generated Images'?")
    return hits[0]


CSV = find_csv()
ROOT = os.path.dirname(CSV)
df = pd.read_csv(CSV)
print("CSV:", CSV, "| columns:", list(df.columns), "| rows:", len(df))

# Column names: file path + label (0 = human, 1 = AI)
path_col = next(c for c in df.columns if "file" in c.lower() or "path" in c.lower() or "image" in c.lower())
label_col = next(c for c in df.columns if "label" in c.lower())
df["path"] = df[path_col].apply(lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p))
df["label"] = df[label_col].astype(int)
df = df[df["path"].apply(os.path.exists)]
print("Images found on disk:", len(df))
print(df["label"].value_counts().rename({0: "REAL (human)", 1: "FAKE (AI)"}))

# %% [markdown]
# ## 1. Balanced sample and train / validation / test split (80 / 10 / 10, stratified)

# %%
df = (df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), PER_CLASS), random_state=42))
        .reset_index(drop=True))
train_df, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=42)
print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")


def load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.uint8), tf.cast(label, tf.float32)   # uint8 so the cache fits in RAM


def make_ds(frame, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((frame["path"].values, frame["label"].values))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).cache()   # decode once, reuse every epoch
    if shuffle:
        ds = ds.shuffle(4096, seed=42)
    return ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = make_ds(train_df, True), make_ds(val_df, False), make_ds(test_df, False)

# %%
images, labels = next(iter(train_ds))
plt.figure(figsize=(12, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(CLASS_NAMES[int(labels[i])])
    plt.axis("off")

# %% [markdown]
# ## 2. Model: EfficientNet-B0 + small classification head
# Transfer learning: EfficientNet-B0 already learned general visual features on ImageNet (1.2M photos).
# We add a sigmoid head that outputs P(AI-generated).
# - **Stage 1:** the backbone is frozen, only the head is trained.
# - **Stage 2:** the top of the backbone is unfrozen and fine-tuned with a small learning rate.
#
# EfficientNet normalises pixels itself, so the model takes raw 0-255 images (same as the app sends).
# The last conv activation (`top_activation`) is used for Grad-CAM.

# %%
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.RandomFlip("horizontal")(inputs)       # active only during training
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)   # P(FAKE)
model = keras.Model(inputs, outputs, name="aices_ai_image_effnetb0")

METRICS = ["accuracy", keras.metrics.AUC(name="auc")]
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=METRICS)
print("Trainable params (stage 1):", sum(int(np.prod(w.shape)) for w in model.trainable_weights))

# %%
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# %% [markdown]
# ### Stage 2: fine-tune the top layers of EfficientNet

# %%
base.trainable = True
for layer in base.layers[:-60]:                    # keep the lower layers frozen
    layer.trainable = False
for layer in base.layers:                          # BatchNorm stays frozen (standard practice)
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=METRICS)
h2 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
)

# %%
acc = h1.history["accuracy"] + h2.history["accuracy"]
val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
plt.plot(acc, label="train"); plt.plot(val_acc, label="val")
plt.axvline(len(h1.history["accuracy"]) - 0.5, ls="--", c="gray"); plt.title("Accuracy (dashed = start of fine-tuning)")
plt.legend();

# %% [markdown]
# ## 3. Test set evaluation (numbers for the report)

# %%
y_true = test_df["label"].values
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
acc, f1, auc = (y_pred == y_true).mean(), f1_score(y_true, y_pred, average="macro"), roc_auc_score(y_true, y_prob)
print(f"accuracy {acc:.3f} | macro F1 {f1:.3f} | ROC-AUC {auc:.3f}")
print("For metrics.json →", {"accuracy": round(acc, 3), "f1": round(f1, 3), "auc": round(auc, 3)})
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES);

# %% [markdown]
# ## 4. Grad-CAM demo: where did the model look?

# %%
grad_model = keras.Model(model.inputs, [model.get_layer("top_activation").output, model.output])


def gradcam(img_batch, prob):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        score = preds[:, 0] if prob >= 0.5 else 1.0 - preds[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    return heat / (heat.max() + 1e-8)


images, labels = next(iter(test_ds.unbatch().shuffle(500, seed=1).batch(6)))
plt.figure(figsize=(14, 5))
for i in range(6):
    img = images[i:i + 1]
    p = float(model.predict(img, verbose=0)[0][0])
    heat = tf.image.resize(gradcam(img, p)[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    plt.subplot(2, 6, i + 1); plt.imshow(img[0].numpy().astype("uint8")); plt.axis("off")
    plt.title(f"true {CLASS_NAMES[int(labels[i])]}")
    plt.subplot(2, 6, i + 7); plt.imshow(img[0].numpy().astype("uint8")); plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off"); plt.title(f"P(fake)={p:.2f}")

# %%
model.save("m4b_ai_image_general.keras")
print("Saved m4b_ai_image_general.keras → copy it to app/models/baseline/")
