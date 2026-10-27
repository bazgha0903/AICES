# %% [markdown]
# # M3 · Privacy Leakage Detection (PII token classifier)
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
