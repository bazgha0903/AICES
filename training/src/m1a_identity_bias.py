# %% [markdown]
# # M1-A · Identity Bias Detection
# **Model:** TF-IDF + Logistic Regression
# **Dataset:** Jigsaw Unintended Bias in Toxicity Classification (Kaggle)
#
# **How to run on Kaggle**
# 1. Kaggle → *New Notebook* → **Add Input** → search *"jigsaw-unintended-bias-in-toxicity-classification"* → add.
#    (The first time, you must accept the competition *Rules*.)
# 2. *Run All*.
# 3. Download `m1a_identity_bias.joblib` from the **Output** panel → put it in `app/models/baseline/`.
#
# **Labelling:** only comments that mention an identity (gender, religion, race, ...) are used.
# `target >= 0.5` → biased (1), otherwise not biased (0).

# %%
import glob

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Install this same scikit-learn version for the local app (otherwise the model will not load)
print("scikit-learn version:", sklearn.__version__)


def find_file(name):
    for root in ["/kaggle/input", "/content", "."]:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found. Did you add the dataset?")

# %% [markdown]
# ## 1. Data load + label

# %%
SAMPLE_SIZE = 100_000      # fits in Kaggle RAM; increase if you like
LABEL_COLUMN = "target"    # alternative: "identity_attack" (only comments attacking an identity)
IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender", "heterosexual", "homosexual_gay_or_lesbian", "bisexual",
    "other_sexual_orientation", "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity", "physical_disability",
    "intellectual_or_learning_disability", "psychiatric_or_mental_illness", "other_disability",
]

path = find_file("train.csv")
print("Reading", path)
df = pd.read_csv(path, usecols=lambda c: c in ["comment_text", "target", "identity_attack"] + IDENTITY_COLS)
ids = [c for c in IDENTITY_COLS if c in df.columns]
df = df[(df[ids].fillna(0) >= 0.5).any(axis=1)].dropna(subset=["comment_text", LABEL_COLUMN])
df["label"] = (df[LABEL_COLUMN] >= 0.5).astype(int)
if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42)
print("Rows:", len(df))
print(df["label"].value_counts().rename({0: "not biased", 1: "biased"}))

# %% [markdown]
# ## 2. Train / test split (80 / 20)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df["comment_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# %% [markdown]
# ## 3. Model: TF-IDF → Logistic Regression
# `class_weight="balanced"` because biased comments are the minority class.

# %%
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000,
                              sublinear_tf=True, strip_accents="unicode")),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
])
pipe.fit(X_train, y_train)

# %% [markdown]
# ## 4. Evaluation (report these numbers)

# %%
pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred, target_names=["not biased", "biased"], digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["not biased", "biased"]);

# %% [markdown]
# ## 5. What did the model learn? (top words, for explainability)

# %%
names = pipe.named_steps["tfidf"].get_feature_names_out()
coef = pipe.named_steps["clf"].coef_[0]
order = np.argsort(coef)
print("Towards biased:     ", ", ".join(names[order[-20:]][::-1]))
print("Towards not biased: ", ", ".join(names[order[:20]]))

# %%
for s in ["Women are too emotional to be good leaders.",
          "The council met on Tuesday to discuss the new school budget.",
          "All Muslims are dangerous.",
          "Many women work in engineering today."]:
    print(f"{pipe.predict_proba([s])[0, 1]:.2f}  {s}")

# %% [markdown]
# ## 6. Save

# %%
joblib.dump(pipe, "m1a_identity_bias.joblib")
print("Saved m1a_identity_bias.joblib → copy it to app/models/baseline/")
