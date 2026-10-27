# %% [markdown]
# # M4 · AI-Generated Image Detection · CLIP features + Logistic Regression
# **Idea (UnivFD, Ojha et al., CVPR 2023):** a frozen, pretrained **CLIP ViT-L/14** image encoder turns every
# image into a 768-number feature vector. CLIP has seen ~400M images of every kind (photos, art, cartoons), so
# these features capture the general look of an image. A **Logistic Regression** trained on top learns which
# directions in that feature space mean "AI-generated". Research shows this generalises to unseen generators
# much better than a CNN trained from scratch.
#
# **Dataset:** *AI vs. Human-Generated Images* (alessandrasala79), plus any other dataset you add that has
# `real/` and `fake/` folders.
# **Robustness:** half of the training images are randomly JPEG-compressed and resized (like screenshots /
# WhatsApp images) before feature extraction.
#
# **How to run on Kaggle**
# 1. **Add Input** → *AI vs. Human-Generated Images* (alessandrasala79).
# 2. Settings → **GPU T4** and **Internet ON** (to download CLIP).
# 3. *Run All* (~20-30 min) → download `m4_clip_lr.json` from Output → copy it to `app/models/baseline/`.

# %%
import glob
import io
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "| torch", torch.__version__)
random.seed(42)
np.random.seed(42)

CLIP_NAME = "openai/clip-vit-large-patch14"
PER_CLASS = 12_000          # 24k images in total; increase if you have time
BATCH = 64
CLASS_NAMES = ["REAL", "AI"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

# %% [markdown]
# ## 1. Collect images

# %%
rows = []
csvs = glob.glob("/kaggle/input/**/train.csv", recursive=True)
root_a = None
if csvs:
    root_a = os.path.dirname(csvs[0])
    df = pd.read_csv(csvs[0])
    pcol = next(c for c in df.columns if "file" in c.lower() or "path" in c.lower() or "image" in c.lower())
    lcol = next(c for c in df.columns if "label" in c.lower())
    rows += [{"path": os.path.join(root_a, p), "label": int(l)} for p, l in zip(df[pcol], df[lcol])]
    print("CSV dataset:", len(df), "rows")

for path in glob.glob("/kaggle/input/**/*", recursive=True):   # optional extra datasets with real/fake folders
    if not path.lower().endswith(IMG_EXT) or (root_a and path.startswith(root_a)):
        continue
    parts = [p.lower() for p in path.split(os.sep)]
    if "fake" in parts or "ai" in parts:
        rows.append({"path": path, "label": 1})
    elif "real" in parts:
        rows.append({"path": path, "label": 0})

data = pd.DataFrame(rows)
data = data[data["path"].apply(os.path.exists)]
print(data["label"].value_counts().rename({0: "REAL", 1: "AI"}))
assert len(data), "No images found. Did you add the dataset?"

data = (data.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), PER_CLASS), random_state=42))
            .reset_index(drop=True))
train_df, rest = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=42)
val_df, test_df = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=42)
print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

# %% [markdown]
# ## 2. CLIP feature extractor (frozen, not trained)

# %%
processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME, torch_dtype=torch.float16).to(DEVICE).eval()


def degrade(img):
    """Robustness: random JPEG compression and resize, like screenshots / WhatsApp images."""
    if random.random() < 0.5:
        s = random.uniform(0.4, 1.0)
        img = img.resize((max(32, int(img.width * s)), max(32, int(img.height * s))), Image.BICUBIC)
    if random.random() < 0.6:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(50, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


@torch.no_grad()
def embed(paths, augment=False):
    feats = []
    for i in range(0, len(paths), BATCH):
        imgs = []
        for p in paths[i:i + BATCH]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            if augment and random.random() < 0.5:
                img = degrade(img)
            imgs.append(img)
        x = processor(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
        e = clip(pixel_values=x).image_embeds.float()
        feats.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if (i // BATCH) % 50 == 0:
            print(f"  {i + len(imgs)}/{len(paths)}")
    return np.vstack(feats)


X_train = embed(list(train_df["path"]), augment=True)
X_val = embed(list(val_df["path"]))
X_test = embed(list(test_df["path"]))
y_train, y_val, y_test = train_df["label"].values, val_df["label"].values, test_df["label"].values
print("Feature shape:", X_train.shape)

# %% [markdown]
# ## 3. Logistic Regression on CLIP features (pick C on the validation set)

# %%
best = None
for C in [0.1, 0.3, 1, 3, 10]:
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X_train, y_train)
    auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    print(f"C={C:<4} val ROC-AUC={auc:.4f}")
    if best is None or auc > best[0]:
        best = (auc, C, lr)
_, C, clf = best
print("Best C:", C)

# %% [markdown]
# ## 4. Test set evaluation (numbers for the report)

# %%
prob = clf.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))
metrics = {"accuracy": round(float((pred == y_test).mean()), 3),
           "f1": round(float(f1_score(y_test, pred, average="macro")), 3),
           "auc": round(float(roc_auc_score(y_test, prob)), 3)}
print("For metrics.json →", metrics)
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASS_NAMES);

# Robustness check: the same test images after JPEG + resize
X_deg = embed(list(test_df["path"]), augment=True)
p_deg = clf.predict_proba(X_deg)[:, 1]
print("Test accuracy on degraded (screenshot-like) images:",
      round(float(((p_deg >= 0.5).astype(int) == y_test).mean()), 3))

# %%
plt.figure(figsize=(14, 4))
for k, i in enumerate(np.random.choice(len(test_df), 6, replace=False)):
    plt.subplot(1, 6, k + 1)
    plt.imshow(Image.open(test_df["path"].values[i]).convert("RGB"))
    plt.title(f"true {CLASS_NAMES[y_test[i]]}\nP(AI)={prob[i]:.2f}"); plt.axis("off")

# %%
with open("m4_clip_lr.json", "w") as f:
    json.dump({"clip_model": CLIP_NAME, "coef": clf.coef_[0].tolist(),
               "intercept": float(clf.intercept_[0]), "C": C, "test_metrics": metrics}, f)
print("Saved m4_clip_lr.json → copy it to app/models/baseline/")
