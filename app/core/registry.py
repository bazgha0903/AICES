"""
Where each model is loaded from, for both baseline and advanced versions.

Baseline version: models/baseline/
Upgrade:          models/advanced/
If a model file is missing, that module shows "not trained yet" and the rest
of the app keeps working.
"""
import os
from pathlib import Path

from . import models as M

MODELS_DIR = Path(os.environ.get("AICES_MODELS_DIR") or Path(__file__).resolve().parent.parent / "models")
VERSIONS = ("baseline", "advanced")


def _halluc_pair(item):
    knowledge, question, answer = item
    return knowledge, f"{question} {answer}".strip()


def _stance_pair(item):
    topic, claim = item
    return topic, claim


def _stance_loader(path):
    """Prefer the pretrained zero-shot NLI model (works on any topic); if torch / transformers
    are missing or the download fails, fall back to the trained TF-IDF + LR stance model."""
    try:
        return M.HFZeroShotStance(path)
    except Exception:
        return M.SklearnStanceClassifier(str(Path(path).parent / "m1c_stance.joblib"))


# module -> version -> (loader, file/folder name)
SPEC = {
    "identity_bias": {
        "baseline": (M.SklearnTextClassifier, "m1a_identity_bias.joblib"),
        "advanced": (lambda p: M.HFSequenceClassifier(p), "m1a_identity_bias"),
    },
    "framing_bias": {
        "baseline": (M.SklearnTextClassifier, "m1b_framing_bias.joblib"),
        "advanced": (lambda p: M.HFSequenceClassifier(p), "m1b_framing_bias"),
    },
    "stance": {
        "baseline": (_stance_loader, "m1c_stance_nli.json"),
        "advanced": (lambda p: M.HFSequenceClassifier(p, pair_fn=_stance_pair), "m1c_stance"),
    },
    "hallucination": {
        "baseline": (M.SklearnHallucinationClassifier, "m2_hallucination.joblib"),
        "advanced": (lambda p: M.HFSequenceClassifier(p, pair_fn=_halluc_pair, max_length=384), "m2_hallucination"),
    },
    "privacy": {
        "baseline": (M.SklearnPIITagger, "m3_privacy.joblib"),
        "advanced": (M.HFPIITagger, "m3_privacy"),
    },
    "deepfake": {
        "baseline": (M.HFImageDetector, "m4_hf_detector.json"),  # any image: pretrained AI-image detector
        "advanced": (M.KerasImageClassifier, "m4_deepfake_efficientnet.keras"),
    },
}

_cache = {}


def get_model(module, version):
    """Return (model, None) or (None, reason). Each model is loaded once and cached."""
    key = (module, version)
    if key in _cache:
        return _cache[key]
    loader, name = SPEC[module][version]
    path = MODELS_DIR / version / name
    if not path.exists():
        result = (None, f"{version} model not trained yet (expected models/{version}/{name})")
    else:
        try:
            result = (loader(str(path)), None)
        except ImportError as e:
            result = (None, f"library missing: {e.name}. See requirements.txt.")
        except Exception as e:  # wrong file, version mismatch, etc.
            result = (None, f"could not load model: {e}")
    _cache[key] = result
    return result


def status():
    out = {}
    for version in VERSIONS:
        out[version] = {}
        for module, spec in SPEC.items():
            out[version][module] = (MODELS_DIR / version / spec[version][1]).exists()
    return out
