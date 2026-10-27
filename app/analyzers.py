"""
Analysis logic for every module. It does not matter which model is underneath
(Logistic Regression or a transformer): all of them share one interface (see models.py).
"""
import base64
import json
import re

import numpy as np

from .explain import explain_text
from .features import NOT_NAMES
from .registry import MODELS_DIR, get_model
from .text_utils import split_sentences
from . import wiki

FLAG_THRESHOLD = 0.5
NEUTRAL_LOW, NEUTRAL_HIGH = 0.4, 0.6  # stance: probability in this range = neutral


def _unavailable(reason):
    return {"available": False, "reason": reason}


# ---------------------------------------------------------------------------
# M1: Bias (A identity, B framing, C one-sided)
# ---------------------------------------------------------------------------
def confidence_risk(p):
    """Map a probability to risk, counting only the confidence above 50%."""
    return max(0.0, (p - 0.5) / 0.5)


def _sentence_classifier(module, sentences, version):
    model, err = get_model(module, version)
    if model is None:
        return _unavailable(err)
    probs = model.predict_proba(sentences)
    top = int(np.argmax(probs))
    flagged = [{"sentence": s, "probability": round(p, 3)}
               for s, p in zip(sentences, probs) if p >= FLAG_THRESHOLD]
    # Explain the most suspicious sentence
    explanation = explain_text(model, sentences[top])
    # Only confidence above 50% counts as risk: p=0.5 -> 0, p=0.75 -> 0.5, p=1.0 -> 1.
    # A borderline guess on a short, neutral sentence no longer drags the score down,
    # while clearly biased sentences (p close to 1) still give high risk.
    top_p = float(probs[top])
    risk = confidence_risk(top_p)
    return {
        "available": True, "model": model.kind, "risk": round(risk, 3), "max_probability": round(top_p, 3),
        "flagged_sentences": flagged, "most_suspicious": sentences[top], "explanation": explanation,
    }


# Identity bias needs a group to be biased about. Lexicon of identity groups (as in the AICES v3
# report: separate gender / religion / race-ethnicity / other category word lists).
_GROUP_WORDS = {
    "gender": "women woman men man girls girl boys boy female females male males ladies lady wife wives "
              "husband husbands mothers fathers gay gays lesbian lesbians transgender trans",
    "religion": "muslim muslims islam hindu hindus christian christians jew jews jewish sikh sikhs "
                "buddhist buddhists religion religions religious atheist atheists",
    "race / ethnicity": "black blacks white whites asian asians african africans indian indians chinese "
                        "pakistani pakistanis bengali biharis bihari immigrant immigrants refugee refugees "
                        "race races caste castes dalit dalits tribal tribals brahmin brahmins foreigners",
    "age / disability": "elderly boomers teenagers disabled handicapped blind deaf",
}
_GROUP_RE = {cat: re.compile(r"\b(" + "|".join(words.split()) + r")\b", re.I) for cat, words in _GROUP_WORDS.items()}
# A sweeping claim about a group: "<group> are / always / never / naturally / tend to / can't ..."
_GENERALISATION = re.compile(r"\b(" + "|".join(" ".join(_GROUP_WORDS.values()).split()) + r")\b"
                             r"(\s+\w+){0,2}?\s+(are|is|always|never|naturally|tend to|can't|cannot|"
                             r"should not|shouldn't|must not|all)\b", re.I)


def mentioned_groups(sentence):
    return [cat for cat, rx in _GROUP_RE.items() if rx.search(sentence)]


def _identity(sentences, version):
    """Hybrid, as in the report: ML model (TF-IDF + LR on Jigsaw) + group lexicon + stereotype pattern.
    A sentence counts only if it mentions an identity group; a sweeping generalisation about a group
    counts as at least 80% risk. Insults with no group ('ignorant') are not identity bias."""
    model, err = get_model("identity_bias", version)
    if model is None:
        return _unavailable(err)
    probs = model.predict_proba(sentences)
    rows = []
    for s, p in zip(sentences, probs):
        groups = mentioned_groups(s)
        if not groups:
            continue
        general = bool(_GENERALISATION.search(s))
        risk = max(confidence_risk(p), 0.8 if general else 0.0)
        rows.append({"sentence": s, "probability": round(max(p, 0.9 if general else 0), 3), "risk": risk,
                     "groups": groups, "generalisation": general})
    flagged = [r for r in rows if r["risk"] > 0]
    if not flagged:
        return {"available": True, "model": model.kind + " + group lexicon", "risk": 0.0, "max_probability": 0.0,
                "flagged_sentences": [], "most_suspicious": None, "explanation": {"method": "", "words": []}}
    top = max(flagged, key=lambda r: r["risk"])
    return {"available": True, "model": model.kind + " + group lexicon", "risk": round(top["risk"], 3),
            "max_probability": top["probability"],
            "flagged_sentences": [{"sentence": r["sentence"], "probability": r["probability"],
                                   "groups": r["groups"], "generalisation": r["generalisation"]} for r in flagged],
            "most_suspicious": top["sentence"], "explanation": explain_text(model, top["sentence"])}


_DEBATE_START = re.compile(r"^\s*(should|is|are|was|were|do|does|did|can|could|will|would|must|has|have|"
                           r"shall|ought)\b", re.I)
_DEBATE_WORDS = re.compile(r"\b(pros and cons|good or bad|for or against|better or worse|agree|disagree|"
                           r"debate|opinion|argue|benefits|advantages|disadvantages)\b", re.I)


def is_debate_question(question):
    """One-sidedness only makes sense for a question with two sides ('Should we ...?', 'Is X good?').
    A factual question ('When was ...?', 'Who is ...?') has one correct answer, not a PRO and a CON side."""
    return bool(_DEBATE_START.search(question) or _DEBATE_WORDS.search(question))


def _stance(question, sentences, version):
    if not question.strip():
        return _unavailable("a question / topic is required for the one-sidedness check")
    if not is_debate_question(question):
        return {"available": True, "model": "not applied", "risk": 0.0, "pro": 0, "con": 0,
                "neutral": len(sentences), "leaning": "not a debate question", "sentences": [],
                "formula": "factual question: one-sidedness not checked"}
    model, err = get_model("stance", version)
    if model is None:
        return _unavailable(err)
    probs = model.predict_proba([(question, s) for s in sentences])
    rows, pro, con = [], 0, 0
    for s, p in zip(sentences, probs):
        if p >= NEUTRAL_HIGH:
            stance, pro = "PRO", pro + 1
        elif p <= NEUTRAL_LOW:
            stance, con = "CON", con + 1
        else:
            stance = "NEUTRAL"
        rows.append({"sentence": s, "stance": stance, "p_pro": round(p, 3)})
    opinion = pro + con
    # Balance of the opinion sentences, scaled by how much evidence we have:
    # 1 opinion sentence counts half, 2 or more count fully, 0 means no evidence.
    balance = abs(pro - con) / opinion if opinion else 0.0
    coverage = min(1.0, opinion / 2)
    one_sided = balance * coverage
    side = "PRO" if pro > con else "CON" if con > pro else "balanced"
    return {
        "available": True, "model": model.kind, "risk": round(one_sided, 3),
        "pro": pro, "con": con, "neutral": len(sentences) - opinion, "leaning": side,
        "formula": "|PRO - CON| / (PRO + CON) × min(1, (PRO + CON) / 2)", "sentences": rows,
    }


SEVERITY = {"identity": 1.0, "framing": 0.8, "one_sided": 0.6}


def analyze_bias(question, answer, version):
    sentences = split_sentences(answer) or [answer]
    parts = {
        "identity": _identity(sentences, version),
        "framing": _sentence_classifier("framing_bias", sentences, version),
        "one_sided": _stance(question, sentences, version),
    }
    risks = [p["risk"] for p in parts.values() if p["available"]]
    if not risks:
        return {"available": False, "reason": "no bias model available", "parts": parts}
    # One clear bias is enough to make content biased, so the strongest bias type decides
    # (an average would let a strong gender stereotype be hidden by two clean bias types).
    # Not every bias is equally harmful: identity bias counts fully, loaded framing a bit less,
    # and a one-sided (incomplete but not false) argument least. The strongest weighted type decides.
    risk = float(max(SEVERITY[k] * p["risk"] for k, p in parts.items() if p["available"]))

    # One combined view: every sentence with all the bias types it shows
    ident = parts["identity"]; frame = parts["framing"]; side = parts["one_sided"]
    TAG = 0.7  # framing: a sentence is tagged only when the model is clearly confident
    ifl = {f["sentence"]: f for f in ident.get("flagged_sentences", [])} if ident["available"] else {}
    fp = {f["sentence"]: f["probability"] for f in frame.get("flagged_sentences", []) if f["probability"] >= TAG} if frame["available"] else {}
    stance = {r["sentence"]: r["stance"] for r in side.get("sentences", [])} if side["available"] else {}
    one_sided_on = side["available"] and side["risk"] >= 0.5
    rows = []
    for s in sentences:
        tags = []
        if s in ifl:
            f = ifl[s]
            what = "sweeping claim about" if f["generalisation"] else "negative / toxic about"
            tags.append({"type": "Identity", "detail": f"{what} a group: {', '.join(f['groups'])} ({round(100 * f['probability'])}%)"})
        if s in fp:
            tags.append({"type": "Framing", "detail": f"loaded / opinionated words ({round(100 * fp[s])}%)"})
        if one_sided_on and stance.get(s) in ("PRO", "CON") and stance[s] == side["leaning"]:
            tags.append({"type": "One-sided", "detail": f"argues only the {side['leaning']} side"})
        rows.append({"sentence": s, "stance": stance.get(s), "tags": tags})

    def level(r):
        return "High" if r >= 0.6 else "Medium" if r >= 0.4 else "Low"
    types = [{"type": name, "what": what, "available": p["available"],
              "risk": p.get("risk"), "level": level(p["risk"]) if p["available"] else None,
              "reason": p.get("reason")}
             for name, what, p in [("Identity", "stereotypes about gender, religion, race", ident),
                                   ("Framing", "loaded / opinionated language", frame),
                                   ("One-sided", "arguments from only one side", side)]]
    found = [t["type"] for t in types if t["available"] and t["level"] != "Low"]
    return {"available": True, "risk": round(risk, 3), "score": round(100 * (1 - risk), 1), "parts": parts,
            "types": types, "sentences": rows, "found": found}


# ---------------------------------------------------------------------------
# M2: Hallucination
# ---------------------------------------------------------------------------
def has_factual_claim(sentence):
    """A sentence is fact-checked if it contains a number or a name
    (a capitalised word that is not the first word and not a common word).
    Contact details (an email or a phone number) are personal data, not facts: the privacy module
    handles them, so those sentences are not fact-checked."""
    if re.search(r"\S+@\S+\.\w+", sentence) or re.search(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b", sentence):
        return False
    if re.search(r"\d", sentence):
        return True
    words = re.findall(r"[A-Za-z][\w'-]*", sentence)
    return any(w[0].isupper() and w.lower() not in NOT_NAMES for w in words[1:])


REASON_TEXT = {
    "word_overlap": ("many answer words are missing from the evidence", "answer words found in the evidence"),
    "unseen_numbers": ("answer has numbers not found in the evidence", "numbers match the evidence"),
    "unseen_names": ("answer has names not found in the evidence", "names found in the evidence"),
}


def _readable_reasons(contribs):
    out = []
    for c in contribs:
        bad, good = REASON_TEXT.get(c["feature"], (c["feature"], c["feature"]))
        out.append({**c, "reason": bad if c["weight"] > 0 else good})
    return out


_NLI = {}


def _nli():
    """Pretrained NLI cross-encoder (same model as the one-sided check), loaded once.
    Returns None if transformers / the model is not available, so the app still works."""
    if "m" not in _NLI:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            cfg = json.loads((MODELS_DIR / "baseline" / "m1c_stance_nli.json").read_text())
            tok = AutoTokenizer.from_pretrained(cfg["model"])
            mdl = AutoModelForSequenceClassification.from_pretrained(cfg["model"]).eval()
            labels = {i: l.lower() for i, l in mdl.config.id2label.items()}
            ci = next(i for i, l in labels.items() if "contra" in l)
            ei = next(i for i, l in labels.items() if "entail" in l)
            _NLI["m"] = (torch, tok, mdl, ci, ei)
        except Exception:
            _NLI["m"] = None
    return _NLI["m"]


def contradiction_check(evidence, sentence, top=8):
    """Does the evidence CONTRADICT the sentence? Checks the evidence sentences that share the most
    words with the claim. Returns (p_contradiction, evidence sentence) or (0.0, None)."""
    nli = _nli()
    if nli is None:
        return 0.0, None
    torch, tok, mdl, ci, ei = nli
    words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    ev = sorted(split_sentences(evidence), key=lambda e: -len(words & set(re.findall(r"[a-z0-9]+", e.lower()))))[:top]
    if not ev:
        return 0.0, None
    with torch.no_grad():
        x = tok(ev, [sentence] * len(ev), truncation=True, max_length=256, padding=True, return_tensors="pt")
        p = torch.softmax(mdl(**x).logits, dim=-1)
    # If any evidence sentence SUPPORTS the claim, it is not a contradiction (small NLI models
    # sometimes call a correct paraphrase a contradiction, e.g. "completed in 1889" vs "built 1887-1889").
    if float(p[:, ei].max()) >= 0.5:
        return 0.0, None
    j = int(p[:, ci].argmax())
    return float(p[j, ci]), ev[j]


def analyze_hallucination(question, answer, reference, version):
    model, err = get_model("hallucination", version)
    if model is None:
        return _unavailable(err)
    knowledge, source, titles = (reference or "").strip(), "user reference", []
    if not knowledge:
        knowledge, titles = wiki.fetch_evidence(question or answer[:200])
        source = "Wikipedia: " + ", ".join(titles) if titles else ""
    if not knowledge:
        return _unavailable("no reference text given and no Wikipedia evidence found (check internet connection)")

    all_sentences = [s for s in split_sentences(answer) if len(re.findall(r"\w+", s)) >= 2] or [answer]
    # Only sentences with a checkable fact (a number or a name) are fact-checked.
    # Opinions ("X is the best", "anyone who disagrees is ignorant") are bias, not hallucination.
    sentences = [s for s in all_sentences if has_factual_claim(s)]
    rows = [{"sentence": s, "skipped": True, "reason": "opinion, contact details or no checkable fact (number or name)"}
            for s in all_sentences if s not in sentences]
    if not sentences:
        return {"available": True, "model": model.kind, "risk": 0.0, "score": 100.0,
                "evidence_source": source, "evidence_preview": knowledge[:400], "sentences": rows,
                "note": "no factual claims to check"}
    probs = model.predict_proba([(knowledge, question, s) for s in sentences])
    for i, (s, p) in enumerate(zip(sentences, probs)):
        reasons = _readable_reasons(model.explain((knowledge, question, s))) if hasattr(model, "explain") else []
        # Second check: the overlap model cannot see a contradiction when the wrong word also appears in the
        # evidence (e.g. "Sydney" in the Australia article). A pretrained NLI model can.
        pc, ev = contradiction_check(knowledge, s)
        if pc >= 0.8 and pc > p:
            p = pc
            reasons = [{"feature": "contradiction", "weight": round(pc, 3),
                        "reason": f"contradicts the evidence: \"{ev[:160]}\""}] + reasons
        probs[i] = p
        row = {"sentence": s, "probability": round(p, 3)}
        if p >= FLAG_THRESHOLD:
            row["reasons"] = reasons
        rows.append(row)
    order = {s: i for i, s in enumerate(all_sentences)}
    rows.sort(key=lambda r: order.get(r["sentence"], 0))
    risk = float(np.mean(probs))
    return {
        "available": True, "model": model.kind + (" + NLI contradiction check" if _nli() else ""),
        "risk": round(risk, 3), "score": round(100 * (1 - risk), 1),
        "evidence_source": source, "evidence_preview": knowledge[:400], "sentences": rows,
    }


# ---------------------------------------------------------------------------
# M3: Privacy
# ---------------------------------------------------------------------------
PII_PENALTY = {"ID_NUMBER": 35, "FINANCIAL": 35, "EMAIL": 20, "PHONE": 20, "ONLINE": 15,
               "DATE_OF_BIRTH": 15, "ADDRESS": 15, "NAME": 10}

# Backup only for Indian ID formats that ai4privacy does not cover
REGEX_BACKUP = {
    "ID_NUMBER": [r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",  # Aadhaar
                  r"\b[A-Z]{5}\d{4}[A-Z]\b"],          # PAN
    "PHONE": [r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"],       # Indian mobile
}


def _regex_spans(text):
    spans = []
    for label, patterns in REGEX_BACKUP.items():
        for pat in patterns:
            for m in re.finditer(pat, text):
                spans.append({"start": m.start(), "end": m.end(), "label": label, "confidence": 1.0,
                              "source": "regex backup", "text": m.group()})
    return spans


def _merge(model_spans, regex_spans):
    """Regex patterns (Aadhaar, PAN, Indian mobile) are exact, so they win over an overlapping
    model span (e.g. a phone number the model called FINANCIAL)."""
    out = [s for s in model_spans
           if not any(r["start"] < s["end"] and s["start"] < r["end"] for r in regex_spans)]
    out += regex_spans
    return sorted(out, key=lambda s: s["start"])


def _plausible(span):
    """Drop obvious false positives: a plain year (e.g. 1925) or a short number is not personal data."""
    t = span["text"].strip(" .,;:()")
    if re.fullmatch(r"(1[5-9]|20)\d{2}", t):        # a year
        return False
    if span["label"] in ("ADDRESS", "FINANCIAL", "ID_NUMBER", "PHONE") and re.fullmatch(r"\d{1,4}", t):
        return False
    return True


def analyze_privacy(text, version):
    model, err = get_model("privacy", version)
    if model is None:
        # No score without the ML model; regex is only a backup
        return {**_unavailable(err), "spans": _regex_spans(text), "counts": {}}
    spans = [s for s in _merge(model.tag(text), _regex_spans(text)) if _plausible(s)]
    penalty = sum(PII_PENALTY.get(s["label"], 10) for s in spans)
    score = max(0.0, 100.0 - penalty)
    counts = {}
    for s in spans:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    return {
        "available": True,
        "model": model.kind,
        "risk": round(1 - score / 100, 3), "score": round(score, 1),
        "spans": spans, "counts": counts,
    }


# ---------------------------------------------------------------------------
# M4: Deepfake image
# ---------------------------------------------------------------------------
def suspicion_level(p):
    """P(AI-generated) -> suspicion level shown to the user."""
    if p >= 0.7:
        return "High"
    if p >= 0.3:
        return "Medium"
    return "Low"


def analyze_image(img, version):
    """Any image -> how suspicious it is of being AI-generated (CLIP features + Logistic Regression)."""
    model, err = get_model("deepfake", version)
    if model is None:
        return _unavailable(err)
    p = model.predict(img)  # P(AI-generated)
    level = suspicion_level(p)
    out = {"available": True, "model": model.kind, "prob_fake": round(p, 4),
           "suspicion": round(100 * p, 1), "level": level,
           "label": {"High": "LIKELY AI-GENERATED", "Medium": "SUSPICIOUS", "Low": "LIKELY REAL"}[level]}
    try:
        png = model.gradcam(img, p)
        out["gradcam_png"] = base64.b64encode(png).decode() if png else None
    except Exception as e:
        out["gradcam_png"], out["gradcam_error"] = None, str(e)
    return out
