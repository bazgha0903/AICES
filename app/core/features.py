"""
Feature functions that must be IDENTICAL in training and in the app.

The training notebooks (m2_hallucination, m3_privacy) contain an exact copy.
If you change anything here, change the notebook too and retrain the model.
"""
import re

import numpy as np

# ---------------------------------------------------------------------------
# M2 Hallucination features
# ---------------------------------------------------------------------------
STOPWORDS = set(
    "the a an of in on at to is was were are and or for by with as from that this it its "
    "be been which who what when where how did does do has have had".split()
)

HALLUCINATION_FEATURE_NAMES = [
    "word_overlap",       # share of the answer's new words found in the evidence (0-1)
    "unseen_numbers",     # numbers in the answer that are not in the evidence
    "unseen_names",       # names (capitalised words) in the answer that are not in the evidence
]


# Common words that start with a capital letter are not names ("It", "The", "Anyone"...)
NOT_NAMES = STOPWORDS | set(
    "i it he she they we you this these those there here anyone someone everyone nobody many some most all "
    "each every any no yes not but if so also however although because while after before during since "
    "first second one two three its their his her our your my".split()
)

def _tokens(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def hallucination_features(knowledge, question, answer):
    """Builds the 3 features for one (knowledge, question, answer) triple."""
    k_tokens = set(_tokens(knowledge))
    q_tokens = set(_tokens(question))
    a_tokens = [t for t in _tokens(answer) if t not in STOPWORDS]
    # Repeating words from the question is not a "new fact", so drop them
    new_tokens = [t for t in a_tokens if t not in q_tokens] or a_tokens
    overlap = float(np.mean([t in k_tokens for t in new_tokens])) if new_tokens else 1.0

    numbers = re.findall(r"\d+", answer or "")
    unseen_numbers = sum(n not in (knowledge or "") for n in numbers)

    names = [n for n in re.findall(r"\b[A-Z][a-z]+", answer or "") if n.lower() not in NOT_NAMES]
    unseen_names = sum(n.lower() not in k_tokens for n in names)

    return [overlap, unseen_numbers, unseen_names]


# ---------------------------------------------------------------------------
# M3 Privacy word features
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\S+")
_STRIP = ".,;:!?()[]{}\"'"


def tokenize_with_offsets(text):
    """Splits text into words and returns (word, start, end) for each one.
    Leading/trailing punctuation (.,;:) is stripped, but the '.' and '@'
    inside an email address are kept."""
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
    return re.sub(r"(.)\1{2,}", r"\1\1", s)  # 'dddddd' -> 'dd' to keep the shape short


def word_features(words, i):
    """Features for word i: what the word looks like + its neighbouring words."""
    w = words[i]
    f = {
        "lower=" + w.lower(): 1,
        "shape=" + _shape(w): 1,
        "suffix3=" + w[-3:].lower(): 1,
        "prefix3=" + w[:3].lower(): 1,
        "is_title": int(w.istitle()),
        "is_upper": int(w.isupper()),
        "is_digit": int(w.isdigit()),
        "has_digit": int(any(c.isdigit() for c in w)),
        "digit_count": min(sum(c.isdigit() for c in w), 16) / 16,
        "has_at": int("@" in w),
        "has_dot": int("." in w),
        "has_dash": int("-" in w),
        "has_slash": int("/" in w),
        "length": min(len(w), 20) / 20,
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


# Map the ~50 ai4privacy labels to 8 simple groups.
# A label must START with one of the prefixes (e.g. 'PHONENUMBER' -> 'PHONE').
# Order matters: 'BITCOINADDRESS' must match FINANCIAL before ADDRESS.
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
    """'FIRSTNAME' -> 'NAME', 'PHONENUMBER' -> 'PHONE', 'JOBTITLE' -> 'O' (not sensitive)."""
    raw = (raw_label or "").upper().replace("_", "")
    if raw.startswith(("B-", "I-")):
        raw = raw[2:]
    raw = raw.rstrip("0123456789")  # 'GIVENNAME1' -> 'GIVENNAME'
    for group, keys in PII_GROUPS:
        if any(raw.startswith(k) for k in keys):
            return group
    return "O"
