# %% [markdown]
# # M1-B · Framing / Subjective Bias Detection
# **Model:** TF-IDF + Logistic Regression
# **Dataset:** BABE (Bias Annotations By Experts): news sentences, labelled by experts as *Biased / Non-biased*
#
# **How to run on Kaggle**
# 1. Turn **Internet: ON** in the notebook settings (the dataset is downloaded from HuggingFace).
# 2. *Run All* → download `m1b_framing_bias.joblib` from Output → `app/models/baseline/`.
#
# If HuggingFace is unavailable, upload the BABE CSV/XLSX as a Kaggle dataset;
# the notebook will find it automatically.

# %%
import glob

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

print("scikit-learn version:", sklearn.__version__)


def find_any(patterns):
    for root in ["/kaggle/input", "/content", "."]:
        for p in patterns:
            hits = glob.glob(f"{root}/**/{p}", recursive=True)
            if hits:
                return hits[0]
    return None

# %% [markdown]
# ## 1. Data load
# Different versions of the dataset use different column names, so the loader handles both.

# %%
def load_babe():
    try:
        from datasets import load_dataset
        ds = load_dataset("mediabiasgroup/BABE")
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        print("Loaded from HuggingFace:", list(df.columns))
    except Exception as e:
        print("HuggingFace load failed:", type(e).__name__, "- looking for a local file")
        path = find_any(["*BABE*.csv", "*babe*.csv", "final_labels_SG2*.csv", "*BABE*.xlsx", "final_labels_SG2*.xlsx"])
        if path is None:
            raise FileNotFoundError("BABE file not found. Turn Internet ON or upload the CSV.")
        df = pd.read_excel(path) if path.endswith("xlsx") else pd.read_csv(path, sep=None, engine="python")
        print("Loaded", path, list(df.columns))

    text_col = next(c for c in ["text", "sentence", "Text"] if c in df.columns)
    if "label_bias" in df.columns:  # strings: "Biased" / "Non-biased" / "No agreement"
        lab = df["label_bias"].astype(str).str.lower().map({"biased": 1, "non-biased": 0})
    else:
        col = next(c for c in ["label", "labels", "bias"] if c in df.columns)
        lab = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame({"text": df[text_col], "label": lab}).dropna()
    out["label"] = out["label"].astype(int)
    return out.drop_duplicates("text")


df = load_babe()
print("Rows:", len(df))
print(df["label"].value_counts().rename({0: "non-biased", 1: "biased"}))
df.sample(5, random_state=1)

# %% [markdown]
# ## 2. Train / test split + model

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
    ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)),
])

# Small dataset, so also report 5-fold cross-validation (a more reliable estimate)
cv = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")
print(f"5-fold CV macro F1: {cv.mean():.3f} ± {cv.std():.3f}")

pipe.fit(X_train, y_train)

# %% [markdown]
# ## 3. Evaluation

# %%
pred = pipe.predict(X_test)
print(classification_report(y_test, pred, target_names=["non-biased", "biased"], digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]), 3))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["non-biased", "biased"]);

# %% [markdown]
# ## 4. Top words (the loaded language the model learned)

# %%
names = pipe.named_steps["tfidf"].get_feature_names_out()
coef = pipe.named_steps["clf"].coef_[0]
order = np.argsort(coef)
print("Towards biased:     ", ", ".join(names[order[-25:]][::-1]))
print("Towards neutral:    ", ", ".join(names[order[:25]]))

# %%
for s in ["The reckless and disastrous policy destroyed the economy.",
          "The policy was approved by parliament on Monday.",
          "Radical activists are once again pushing their extreme agenda."]:
    print(f"{pipe.predict_proba([s])[0, 1]:.2f}  {s}")

# %%
joblib.dump(pipe, "m1b_framing_bias.joblib")
print("Saved m1b_framing_bias.joblib → app/models/baseline/")
