# %% [markdown]
# # M1-C · One-sided Bias (Stance Detection)
# **Model:** TF-IDF + Logistic Regression
# **Dataset:** IBM Debater® Claim Stance: 2,394 claims, 55 topics, each claim labelled **PRO / CON**
#
# **Idea:** the model labels each sentence PRO or CON relative to the topic. The app then computes
# `one-sided score = |PRO − CON| / (PRO + CON)` → 0 = balanced, 1 = completely one-sided.
#
# **Data:** just turn **Internet ON**. The notebook downloads IBM's official HuggingFace dataset
# (`ibm-research/claim_stance`) automatically. Nothing to upload.
#
# Note: stance detection is a hard task, so expect lower accuracy than M1-A/B. Report the number you actually get.

# %%
import glob

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

print("scikit-learn version:", sklearn.__version__)


def find_any(patterns):
    for root in ["/kaggle/input", "/content", "."]:
        for p in patterns:
            hits = glob.glob(f"{root}/**/{p}", recursive=True)
            if hits:
                return hits[0]
    return None


def pick(df, names):
    return next((c for c in names if c in df.columns), None)

# %% [markdown]
# ## 1. Data load

# %%
def load_claim_stance():
    df = None
    path = find_any(["claim_stance_dataset_v1.csv", "*claim_stance*.csv"])
    if path:
        df = pd.read_csv(path)
        print("Loaded", path)
    else:
        try:
            from datasets import load_dataset
            for name in ["ibm-research/claim_stance", "ibm/claim_stance"]:
                try:
                    try:
                        ds = load_dataset(name, "claim_stance")  # the dataset has 2 configs
                    except Exception:
                        ds = load_dataset(name)
                    parts = []
                    for split in ds.keys():
                        p = ds[split].to_pandas()
                        p["split"] = split
                        feat = ds[split].features.get("label") or ds[split].features.get("stance")
                        if hasattr(feat, "names"):  # ClassLabel -> name
                            col = "label" if "label" in p.columns else "stance"
                            p[col] = p[col].map(dict(enumerate(feat.names)))
                        parts.append(p)
                    df = pd.concat(parts, ignore_index=True)
                    print("Loaded from HuggingFace:", name)
                    break
                except Exception:
                    continue
        except ImportError:
            pass
    if df is None:
        raise FileNotFoundError("Claim Stance data not found. Is Internet ON in Settings?")

    topic = pick(df, ["topicText", "topic", "topic_text"])
    claim = pick(df, ["claims.claimCorrectedText", "claimCorrectedText", "claim", "claims.claimOriginalText", "text"])
    stance = pick(df, ["claims.stance", "stance", "label"])
    split = pick(df, ["split"])
    out = pd.DataFrame({
        "topic": df[topic].astype(str),
        "claim": df[claim].astype(str),
        "label": df[stance].astype(str).str.upper().str.strip().map({"PRO": 1, "CON": 0, "1": 1, "0": 0}),
        "split": df[split].astype(str).str.lower() if split else "",
    }).dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out


df = load_claim_stance()
print("Rows:", len(df), "| topics:", df["topic"].nunique())
print(df["label"].value_counts().rename({0: "CON", 1: "PRO"}))

# %% [markdown]
# ## 2. Split **by topic**
# Test topics must not appear in training, otherwise the model memorises topics. IBM's own train/test split
# does exactly this; if it is missing we fall back to GroupShuffleSplit.

# %%
df["text"] = df["topic"] + " || " + df["claim"]   # the app sends the same format
if set(df["split"]) >= {"train", "test"}:
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
else:
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(df, groups=df["topic"]))
    train, test = df.iloc[tr], df.iloc[te]
print("train", len(train), "| test", len(test))

# %%
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
])
pipe.fit(train["text"], train["label"])

# %% [markdown]
# ## 3. Evaluation (on unseen topics)

# %%
pred = pipe.predict(test["text"])
print(classification_report(test["label"], pred, target_names=["CON", "PRO"], digits=3))
ConfusionMatrixDisplay.from_predictions(test["label"], pred, display_labels=["CON", "PRO"]);

# %% [markdown]
# ## 4. One-sided score demo

# %%
topic = "We should use nuclear energy"
answer = ["Nuclear energy produces almost no carbon emissions.",
          "It is a reliable source of electricity.",
          "Nuclear plants are among the safest ways to generate power."]
p = pipe.predict_proba([f"{topic} || {s}" for s in answer])[:, 1]
pro, con = int((p >= 0.6).sum()), int((p <= 0.4).sum())
print("P(PRO):", np.round(p, 2), "| PRO", pro, "CON", con,
      "| one-sided =", round(abs(pro - con) / (pro + con), 2) if pro + con >= 2 else "need >=2 opinion sentences")

# %%
joblib.dump(pipe, "m1c_stance.joblib")
print("Saved m1c_stance.joblib → app/models/baseline/")
