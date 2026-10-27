# %% [markdown]
# # M2 · Hallucination Detection
# **Model:** 3 knowledge-overlap features + Logistic Regression
# **Dataset:** HaluEval (QA): 10,000 questions, each with *knowledge*, a *right answer* and a *hallucinated answer*
#
# **Idea:** an answer that is not supported by the given knowledge (facts) is hallucinated.
#
# | Feature | Meaning |
# |---|---|
# | `word_overlap` | share of the answer's new words found in the knowledge |
# | `unseen_numbers` | numbers in the answer that are not in the knowledge |
# | `unseen_names` | names (capitalised words) in the answer that are not in the knowledge |
#
# **Run:** Internet ON (data comes from GitHub) → *Run All* → `m2_hallucination.joblib` → `app/models/baseline/`.
# No GPU needed; it also runs on a laptop in ~1 minute.

# %%
import glob
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("scikit-learn version:", sklearn.__version__)

# %% [markdown]
# ## 1. Data load

# %%
URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
local = [p for root in ["/kaggle/input", "/content", "."] for p in glob.glob(f"{root}/**/qa_data.json", recursive=True)]
raw = pd.read_json(local[0] if local else URL, lines=True)
print("Questions:", len(raw))

# Each question gives 2 rows: right answer (0) and hallucinated answer (1)
rows = []
for i, r in raw.iterrows():
    rows.append({"qid": i, "knowledge": r.knowledge, "question": r.question, "answer": r.right_answer, "label": 0})
    rows.append({"qid": i, "knowledge": r.knowledge, "question": r.question, "answer": r.hallucinated_answer, "label": 1})
df = pd.DataFrame(rows)
df.head(4)

# %% [markdown]
# ## 2. Features
# **This function is identical to `app/core/features.py`.** If you change one, change the other.

# %%
STOPWORDS = set(
    "the a an of in on at to is was were are and or for by with as from that this it its "
    "be been which who what when where how did does do has have had".split()
)
FEATURE_NAMES = ["word_overlap", "unseen_numbers", "unseen_names"]


# Common words that start with a capital letter are not names ("It", "The", "Anyone"...)
NOT_NAMES = STOPWORDS | set(
    "i it he she they we you this these those there here anyone someone everyone nobody many some most all "
    "each every any no yes not but if so also however although because while after before during since "
    "first second one two three its their his her our your my".split()
)

def _tokens(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def hallucination_features(knowledge, question, answer):
    k_tokens = set(_tokens(knowledge))
    q_tokens = set(_tokens(question))
    a_tokens = [t for t in _tokens(answer) if t not in STOPWORDS]
    new_tokens = [t for t in a_tokens if t not in q_tokens] or a_tokens
    overlap = float(np.mean([t in k_tokens for t in new_tokens])) if new_tokens else 1.0
    numbers = re.findall(r"\d+", answer or "")
    unseen_numbers = sum(n not in (knowledge or "") for n in numbers)
    names = [n for n in re.findall(r"\b[A-Z][a-z]+", answer or "") if n.lower() not in NOT_NAMES]
    unseen_names = sum(n.lower() not in k_tokens for n in names)
    return [overlap, unseen_numbers, unseen_names]

# %% [markdown]
# ## 3. Split **by question**
# Both answers of a question go to the same split (train or test). This keeps the test fair.

# %%
tr_idx, te_idx = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(df, groups=df["qid"]))
train, test = df.iloc[tr_idx], df.iloc[te_idx]


def make_X(d):
    return np.array([hallucination_features(k, q, a)
                     for k, q, a in zip(d["knowledge"], d["question"], d["answer"])])


X_train, X_test = make_X(train), make_X(test)
pd.DataFrame(X_train, columns=FEATURE_NAMES).assign(label=train["label"].values).groupby("label").mean()

# %% [markdown]
# The table above shows that hallucinated answers (label 1) have lower overlap and more unseen names/numbers.

# %%
model = Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
model.fit(X_train, train["label"])

# %% [markdown]
# ## 4. Evaluation

# %%
pred = model.predict(X_test)
print(classification_report(test["label"], pred, target_names=["factual", "hallucinated"], digits=3))
print("ROC-AUC:", round(roc_auc_score(test["label"], model.predict_proba(X_test)[:, 1]), 3))
ConfusionMatrixDisplay.from_predictions(test["label"], pred, display_labels=["factual", "hallucinated"]);

# %%
print("Feature weights (positive = towards hallucination):")
for n, w in zip(FEATURE_NAMES, model.named_steps["clf"].coef_[0]):
    print(f"  {n:18s} {w:+.3f}")

# %% [markdown]
# **Design notes:**
# - An earlier version also used TF-IDF cosine similarity, but it carried almost the same information as
#   `word_overlap` (collinear) and added only ~0.7% accuracy. It was removed to keep the model simple and interpretable.
# - **Limitation:** in HaluEval, right answers are short (2-3 words) and hallucinated answers are longer.
#   Answer length was deliberately **not** used as a feature, otherwise the model would learn length instead of
#   factual support.

# %%
joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, "m2_hallucination.joblib")
print("Saved m2_hallucination.joblib → app/models/baseline/")
