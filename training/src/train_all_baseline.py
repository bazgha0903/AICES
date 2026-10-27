# %% [markdown]
# # AICES · Train ALL baseline models (one click)
# This notebook trains every baseline model and finally creates **`aices_baseline_models.zip`**.
# Download the zip → unzip it → copy all files into `app/models/baseline/`. Done.
#
# ## Before you run (one time, ~5 min)
# 1. **Settings → Internet: ON** (requires a phone-verified Kaggle account)
# 2. **Settings → Accelerator: GPU T4** (for the deepfake CNN)
# 3. **Add Input** (right panel) → add these 2 datasets:
#    - `jigsaw-unintended-bias-in-toxicity-classification` (competition; accept the *Rules* first)
#    - `CIFAKE Real and AI-Generated Synthetic Images`
#    (BABE, IBM Claim Stance, HaluEval and ai4privacy are downloaded automatically from the internet.)
# 4. **Run All**. Total time ~30-40 min.
#
# Every part prints its results (accuracy, F1, confusion matrix); **take screenshots for the report**.
# If a part fails, fix the error and use *Run after* from that part's first cell.

# %% [markdown]
# ---
# # PART 1 · M1-A · Identity Bias Detection
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

# %% [markdown]
# ---
# # PART 2 · M1-B · Framing / Subjective Bias Detection
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

# %% [markdown]
# ---
# # PART 3 · M1-C · One-sided Bias (Stance Detection)
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

# %% [markdown]
# ---
# # PART 4 · M2 · Hallucination Detection
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

# %% [markdown]
# ---
# # PART 5 · M3 · Privacy Leakage Detection (PII token classifier)
# **Model:** word features + Logistic Regression (predicts a label for every word)
# **Dataset:** ai4privacy PII masking (HuggingFace), English sentences only
#
# **Idea:** build features for every word (capitalised? digits? `@`? neighbouring words?) and Logistic Regression
# predicts whether the word is **NAME / EMAIL / PHONE / ADDRESS / ID_NUMBER / FINANCIAL / ONLINE / DATE_OF_BIRTH**
# or ordinary (**O**). In the app, consecutive words with the same label become one span ("Rahul Sharma" = 1 NAME).
#
# **Run:** Internet ON → *Run All* (~5-10 min on CPU) → `m3_privacy.joblib` → `app/models/baseline/`.

# %%
import ast
import json
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

print("scikit-learn version:", sklearn.__version__)

# %% [markdown]
# ## 1. Helper functions
# **These are identical to `app/core/features.py`.** If you change one, change the other.

# %%
_WORD_RE = re.compile(r"\S+")
_STRIP = ".,;:!?()[]{}\"'"


def tokenize_with_offsets(text):
    out = []
    for m in _WORD_RE.finditer(text or ""):
        w, s, e = m.group(), m.start(), m.end()
        while w and w[0] in _STRIP:
            w, s = w[1:], s + 1
        while w and w[-1] in _STRIP:
            w, e = w[:-1], e - 1
        if w:
            out.append((w, s, e))
    return out


def _shape(w):
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"[0-9]", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def word_features(words, i):
    w = words[i]
    f = {
        "lower=" + w.lower(): 1, "shape=" + _shape(w): 1,
        "suffix3=" + w[-3:].lower(): 1, "prefix3=" + w[:3].lower(): 1,
        "is_title": int(w.istitle()), "is_upper": int(w.isupper()), "is_digit": int(w.isdigit()),
        "has_digit": int(any(c.isdigit() for c in w)),
        "digit_count": min(sum(c.isdigit() for c in w), 16) / 16,
        "has_at": int("@" in w), "has_dot": int("." in w), "has_dash": int("-" in w),
        "has_slash": int("/" in w), "length": min(len(w), 20) / 20,
    }
    if i > 0:
        p = words[i - 1]
        f["prev=" + p.lower()] = 1
        f["prev_is_title"] = int(p.istitle())
    else:
        f["BOS"] = 1
    if i < len(words) - 1:
        n = words[i + 1]
        f["next=" + n.lower()] = 1
        f["next_is_title"] = int(n.istitle())
    else:
        f["EOS"] = 1
    if i > 1:
        f["prev2=" + words[i - 2].lower()] = 1
    return f


PII_GROUPS = [
    ("EMAIL", ["EMAIL"]),
    ("PHONE", ["PHONE", "TEL"]),
    ("FINANCIAL", ["CREDITCARD", "IBAN", "BIC", "ACCOUNTNUM", "BITCOIN", "ETHEREUM", "LITECOIN", "MASKEDNUMBER", "PIN"]),
    ("ID_NUMBER", ["SSN", "SOCIALNUM", "PASSPORT", "DRIVERLICENSE", "IDCARD", "TAXNUM", "VEHICLE", "AADHAAR", "PAN"]),
    ("ONLINE", ["IPV", "IP", "MAC", "URL", "PASSWORD", "PASS", "USERAGENT"]),
    ("ADDRESS", ["STREET", "CITY", "STATE", "ZIPCODE", "POSTCODE", "BUILDING", "SECONDARYADDRESS", "SECADDRESS",
                 "COUNTY", "GEOCOORD", "NEARBYGPS", "GPS", "ADDRESS"]),
    ("NAME", ["FIRSTNAME", "LASTNAME", "MIDDLENAME", "USERNAME", "ACCOUNTNAME", "FULLNAME", "GIVENNAME",
              "SURNAME", "NAME"]),
    ("DATE_OF_BIRTH", ["DOB", "BOD"]),
]


def map_pii_label(raw_label):
    raw = (raw_label or "").upper().replace("_", "")
    if raw.startswith(("B-", "I-")):
        raw = raw[2:]
    raw = raw.rstrip("0123456789")
    for group, keys in PII_GROUPS:
        if any(raw.startswith(k) for k in keys):
            return group
    return "O"

# %% [markdown]
# ## 2. Load data + label every word
# The dataset gives PII **character positions** (start, end). A word that falls inside a PII span
# gets that span's label.

# %%
N_SENTENCES = 15_000  # more = better, but slower training

from datasets import load_dataset

ds = None
for name in ["ai4privacy/pii-masking-200k", "ai4privacy/pii-masking-300k"]:
    try:
        ds = load_dataset(name, split="train")
        print("Loaded", name)
        break
    except Exception as e:
        print(name, "could not be loaded:", type(e).__name__)
lang_col = "language" if "language" in ds.column_names else None
if lang_col:
    ds = ds.filter(lambda r: str(r[lang_col]).lower().startswith("en"))
ds = ds.shuffle(seed=42).select(range(min(N_SENTENCES, len(ds))))
print("English sentences:", len(ds))


def parse_mask(m):
    if isinstance(m, str):
        try:
            return json.loads(m)
        except ValueError:
            return ast.literal_eval(m)
    return list(m or [])


def sentence_to_rows(text, mask):
    spans = [(int(s["start"]), int(s["end"]), map_pii_label(s["label"])) for s in parse_mask(mask)]
    toks = tokenize_with_offsets(text)
    words = [w for w, _, _ in toks]
    labels = []
    for _, s, e in toks:
        lab = "O"
        for ss, ee, l in spans:
            if s < ee and ss < e:
                lab = l
                break
        labels.append(lab)
    return [word_features(words, i) for i in range(len(words))], labels


sent_feats, sent_labels = [], []
for r in ds:
    f, l = sentence_to_rows(r["source_text"], r["privacy_mask"])
    if f:
        sent_feats.append(f)
        sent_labels.append(l)
print("Example:", list(zip([w for w, _, _ in tokenize_with_offsets(ds[0]['source_text'])][:15], sent_labels[0][:15])))

# %% [markdown]
# ## 3. Split **by sentence** (all words of a sentence stay in the same split)

# %%
idx = np.arange(len(sent_feats))
tr, te = train_test_split(idx, test_size=0.2, random_state=42)
flat = lambda ids, L: [x for i in ids for x in L[i]]
X_train, y_train = flat(tr, sent_feats), flat(tr, sent_labels)
X_test, y_test = flat(te, sent_feats), flat(te, sent_labels)
print("train words:", len(X_train), "| test words:", len(X_test))
print(pd.Series(y_train).value_counts())

# %%
pipe = Pipeline([
    ("vec", DictVectorizer(sparse=True)),
    ("clf", LogisticRegression(C=1.0, max_iter=300)),
])
pipe.fit(X_train, y_train)

# %% [markdown]
# ## 4. Evaluation (excluding O, since the real task is catching PII)

# %%
pred = pipe.predict(X_test)
labels = sorted(set(y_test) - {"O"})
print(classification_report(y_test, pred, labels=labels, digits=3, zero_division=0))

# %%
def demo(text):
    toks = tokenize_with_offsets(text)
    words = [w for w, _, _ in toks]
    p = pipe.predict([word_features(words, i) for i in range(len(words))])
    return [(w, str(l)) for w, l in zip(words, p) if l != "O"]


print(demo("Hi, I am Priya Verma. Mail me at priya.verma@gmail.com or call +1 415 555 0199. I live at 22 Park Street, Kolkata."))

# %%
joblib.dump(pipe, "m3_privacy.joblib")
print("Saved m3_privacy.joblib → app/models/baseline/")

# %% [markdown]
# ---
# # PART 6 · M4 · Deepfake Image Detection
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

# %% [markdown]
# ---
# # Done: all models in one zip

# %%
import os, zipfile
files = ["m1a_identity_bias.joblib", "m1b_framing_bias.joblib", "m1c_stance.joblib",
         "m2_hallucination.joblib", "m3_privacy.joblib", "m4_deepfake_cnn.keras"]
with zipfile.ZipFile("aices_baseline_models.zip", "w") as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
            print("added  ", f)
        else:
            print("MISSING", f, "(check the error in that part)")
import sklearn
print("\nInstall on your laptop:  pip install scikit-learn==" + sklearn.__version__)
print("Download: right panel → Output → aices_baseline_models.zip")
