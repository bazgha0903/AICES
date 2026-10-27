# %% [markdown]
# # M4 · Deepfake Image Detection
# **Model:** Simple CNN (Keras), trained from scratch
# **Dataset:** CIFAKE: 60k REAL + 60k FAKE (Stable Diffusion) images, 32×32
#
# **How to run on Kaggle**
# 1. **Add Input** → search *"CIFAKE Real and AI-Generated Synthetic Images"* → add.
# 2. Settings → **Accelerator: GPU T4** (free).
# 3. *Run All* (~15-20 min) → download `m4_deepfake_cnn.keras` from Output → `app/models/baseline/`.
#
# Want face deepfakes instead? Use the *"140k Real and Fake Faces"* dataset, set `IMG_SIZE = 64` and point
# `TRAIN_DIR` / `TEST_DIR` at the directory containing its `real` / `fake` folders.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score

print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

IMG_SIZE = 32
BATCH = 128
EPOCHS = 15
CLASS_NAMES = ["REAL", "FAKE"]  # REAL = 0, FAKE = 1


def find_dir(split):
    for root in ["/kaggle/input", "/content", "."]:
        hits = glob.glob(f"{root}/**/{split}/REAL", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError(f"'{split}/REAL' folder not found. Did you add CIFAKE?")


TRAIN_DIR, TEST_DIR = find_dir("train"), find_dir("test")
print(TRAIN_DIR, TEST_DIR)

# %% [markdown]
# ## 1. Load data (10% of train held out for validation)

# %%
common = dict(labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
              image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH)
train_ds = keras.utils.image_dataset_from_directory(TRAIN_DIR, validation_split=0.1, subset="training",
                                                    seed=42, **common)
val_ds = keras.utils.image_dataset_from_directory(TRAIN_DIR, validation_split=0.1, subset="validation",
                                                  seed=42, **common)
test_ds = keras.utils.image_dataset_from_directory(TEST_DIR, shuffle=False, **common)
AUTOTUNE = tf.data.AUTOTUNE
train_ds, val_ds, test_ds = (d.cache().prefetch(AUTOTUNE) for d in (train_ds, val_ds, test_ds))

# %%
images, labels = next(iter(train_ds))
plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(CLASS_NAMES[int(labels[i][0])])
    plt.axis("off")

# %% [markdown]
# ## 2. CNN model
# 3 blocks of `Conv2D → MaxPool`, then `Dense`. The last conv layer is named **`last_conv`** because
# Grad-CAM (in the app) uses it. `Rescaling` is inside the model, so the app sends raw images.

# %%
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.RandomFlip("horizontal")(x)  # active only during training
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # P(FAKE)
model = keras.Model(inputs, outputs, name="aices_deepfake_cnn")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

# %%
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)

# %%
pd_hist = history.history
plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1); plt.plot(pd_hist["accuracy"], label="train"); plt.plot(pd_hist["val_accuracy"], label="val"); plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(pd_hist["loss"], label="train"); plt.plot(pd_hist["val_loss"], label="val"); plt.title("Loss"); plt.legend()

# %% [markdown]
# ## 3. Test set evaluation (numbers for the report)

# %%
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
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


images, labels = next(iter(test_ds.unbatch().shuffle(500).batch(6)))
plt.figure(figsize=(12, 4))
for i in range(6):
    img = images[i:i + 1]
    heat = tf.image.resize(gradcam(img)[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    p = float(model.predict(img, verbose=0)[0][0])
    plt.subplot(2, 6, i + 1); plt.imshow(img[0].numpy().astype("uint8")); plt.axis("off")
    plt.title(f"true {CLASS_NAMES[int(labels[i][0])]}")
    plt.subplot(2, 6, i + 7); plt.imshow(img[0].numpy().astype("uint8")); plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off"); plt.title(f"P(fake)={p:.2f}")

# %%
model.save("m4_deepfake_cnn.keras")
print("Saved m4_deepfake_cnn.keras → app/models/baseline/")
