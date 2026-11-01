"""
Model wrappers. Every wrapper exposes the SAME interface, whether it wraps Logistic
Regression or DeBERTa. That is why baseline and advanced models can run in one app.

  Text classifier : predict_proba(list[str])             -> list[float]  (P(label=1))
  Pair classifier : predict_proba(list[tuple])           -> list[float]
  Tagger (PII)    : tag(text)                            -> list[dict]   (spans)
  Image classifier: predict(PIL.Image) / gradcam(PIL)    -> float / PNG bytes
"""
import io

import joblib
import numpy as np

from .features import hallucination_features, tokenize_with_offsets, word_features, HALLUCINATION_FEATURE_NAMES


# ---------------------------------------------------------------------------
# BASELINE (scikit-learn) wrappers
# ---------------------------------------------------------------------------
class SklearnTextClassifier:
    """TF-IDF + Logistic Regression pipeline (M1-A, M1-B)."""

    kind = "TF-IDF + Logistic Regression"

    def __init__(self, path):
        self.pipe = joblib.load(path)
        self._pos = list(self.pipe.classes_).index(1)

    def _texts(self, items):
        return list(items)

    def predict_proba(self, items):
        if not items:
            return []
        return self.pipe.predict_proba(self._texts(items))[:, self._pos].tolist()

    def explain(self, item, top_k=6):
        """For a linear model each word's contribution = TF-IDF value x LR weight.
        This is exact (not a LIME approximation) and instant."""
        text = self._texts([item])[0]
        vec, clf = self.pipe.steps[0][1], self.pipe.steps[-1][1]
        x = vec.transform([text])
        coef = clf.coef_[0] if self._pos == 1 else -clf.coef_[0]
        names = vec.get_feature_names_out()
        contrib = [(names[j], float(v * coef[j])) for j, v in zip(x.indices, x.data)]
        contrib.sort(key=lambda t: -abs(t[1]))
        return [{"feature": f, "weight": round(w, 4)} for f, w in contrib[:top_k]]

    # LIME needs a text-in, probability-out function
    def lime_predict_fn(self):
        return lambda texts: self.pipe.predict_proba(list(texts))


class SklearnStanceClassifier(SklearnTextClassifier):
    """M1-C: (topic, sentence) -> P(PRO). Training used the same 'topic || claim' format."""

    def _texts(self, items):
        return [f"{t} || {c}" for t, c in items]

    def lime_predict_fn(self):
        return None  # LIME on a pair input is not meaningful, so skip it


class HFZeroShotStance:
    """M1-C (one-sided bias): pretrained NLI model used zero-shot, so it works on ANY topic.
    For each sentence the model checks three hypotheses: the sentence says YES to the question,
    says NO, or takes no side. P(PRO) = 0.5 + 0.5 × (P(yes) − P(no)); close to 0.5 means neutral.
    Runs locally (downloaded once from HuggingFace, no API key)."""

    LABELS = ("says yes to it", "says no to it", "takes no side on it")

    def __init__(self, path):
        import json

        from transformers import pipeline

        cfg = json.loads(open(path, encoding="utf-8").read())
        self.clf = pipeline("zero-shot-classification", model=cfg["model"])
        self.kind = f"Pretrained NLI model, zero-shot ({cfg['model']})"

    def predict_proba(self, items):
        out = []
        for topic, sentence in items:
            r = self.clf(sentence, candidate_labels=list(self.LABELS),
                         hypothesis_template=f"About the question '{topic}', this text {{}}.")
            p = dict(zip(r["labels"], r["scores"]))
            out.append(0.5 + 0.5 * (p[self.LABELS[0]] - p[self.LABELS[1]]))
        return out

    def explain(self, item, top_k=6):
        return []

    def lime_predict_fn(self):
        return None


class SklearnHallucinationClassifier:
    """M2: (knowledge, question, answer) -> P(hallucinated). 3 features + Logistic Regression."""

    kind = "Knowledge-overlap features + Logistic Regression"

    def __init__(self, path):
        bundle = joblib.load(path)
        self.model = bundle["model"]  # StandardScaler + LogisticRegression
        self._pos = list(self.model.classes_).index(1)

    def _X(self, items):
        return np.array([hallucination_features(k, q, a) for k, q, a in items])

    def predict_proba(self, items):
        if not items:
            return []
        return self.model.predict_proba(self._X(items))[:, self._pos].tolist()

    def explain(self, item, top_k=4):
        """Each feature's contribution = scaled value x LR weight (positive = towards hallucination)."""
        raw = self._X([item])[0]
        scaler, clf = self.model.steps[0][1], self.model.steps[-1][1]
        scaled = scaler.transform([raw])[0]
        coef = clf.coef_[0] if self._pos == 1 else -clf.coef_[0]
        out = [
            {"feature": n, "value": round(float(r), 3), "weight": round(float(s * c), 3)}
            for n, r, s, c in zip(HALLUCINATION_FEATURE_NAMES, raw, scaled, coef)
        ]
        out.sort(key=lambda d: -abs(d["weight"]))
        return out[:top_k]


class SklearnPIITagger:
    """M3: labels every word (NAME, EMAIL, PHONE... or O). DictVectorizer + Logistic Regression."""

    kind = "Word features + Logistic Regression (token classifier)"

    def __init__(self, path):
        self.pipe = joblib.load(path)

    def tag(self, text, min_conf=0.5):
        toks = tokenize_with_offsets(text)
        if not toks:
            return []
        words = [w for w, _, _ in toks]
        feats = [word_features(words, i) for i in range(len(words))]
        probs = self.pipe.predict_proba(feats)
        classes = list(self.pipe.classes_)
        spans = []
        for (w, s, e), p in zip(toks, probs):
            j = int(np.argmax(p))
            label, conf = classes[j], float(p[j])
            if label == "O" or conf < min_conf:
                continue
            # Merge consecutive words with the same label into one span ("Rahul Sharma" = 1 NAME)
            if spans and spans[-1]["label"] == label and s - spans[-1]["end"] <= 1:
                spans[-1]["end"] = e
                spans[-1]["confidence"] = min(spans[-1]["confidence"], conf)
            else:
                spans.append({"start": s, "end": e, "label": label, "confidence": conf, "source": "model"})
        for sp in spans:
            sp["text"] = text[sp["start"]:sp["end"]]
            sp["confidence"] = round(sp["confidence"], 3)
        return spans


# ---------------------------------------------------------------------------
# ADVANCED (HuggingFace transformers) wrappers
# These work once a fine-tuned model is saved in models/advanced/<name>/
# (model.save_pretrained + tokenizer.save_pretrained). Label 1 = positive class.
# ---------------------------------------------------------------------------
class HFSequenceClassifier:
    kind = "Fine-tuned transformer"

    def __init__(self, path, pair_fn=None, max_length=256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.pair_fn = pair_fn  # tuple -> (text_a, text_b)
        self.max_length = max_length
        self.kind = f"Fine-tuned {self.model.config.model_type}"

    def _encode(self, items):
        if self.pair_fn:
            a, b = zip(*[self.pair_fn(it) for it in items])
            return self.tok(list(a), list(b), truncation=True, max_length=self.max_length,
                            padding=True, return_tensors="pt")
        return self.tok(list(items), truncation=True, max_length=self.max_length,
                        padding=True, return_tensors="pt")

    def _proba_matrix(self, items):
        out = []
        for i in range(0, len(items), 16):
            with self.torch.no_grad():
                logits = self.model(**self._encode(items[i:i + 16])).logits
            out.append(self.torch.softmax(logits, dim=-1).numpy())
        return np.vstack(out)

    def predict_proba(self, items):
        if not items:
            return []
        return self._proba_matrix(list(items))[:, 1].tolist()

    def explain(self, item, top_k=6):
        return []  # advanced models are explained with LIME (explain.py)

    def lime_predict_fn(self):
        if self.pair_fn:
            return None
        return lambda texts: self._proba_matrix(list(texts))


class HFPIITagger:
    kind = "Fine-tuned transformer (NER)"

    def __init__(self, path):
        from transformers import pipeline

        self.ner = pipeline("token-classification", model=path, tokenizer=path, aggregation_strategy="simple")

    def tag(self, text, min_conf=0.5):
        from .features import map_pii_label

        spans = []
        for ent in self.ner(text):
            label = map_pii_label(ent["entity_group"])
            if label == "O" or ent["score"] < min_conf:
                continue
            spans.append({"start": int(ent["start"]), "end": int(ent["end"]), "label": label,
                          "confidence": round(float(ent["score"]), 3), "source": "model",
                          "text": text[int(ent["start"]):int(ent["end"])]})
        return spans


# ---------------------------------------------------------------------------
# Image (Keras) wrapper: used for both the baseline CNN and advanced EfficientNet
# ---------------------------------------------------------------------------
class KerasImageClassifier:
    kind = "CNN (Keras)"

    def __init__(self, path):
        import tensorflow as tf
        from tensorflow import keras

        self.tf, self.keras = tf, keras
        self.model = keras.models.load_model(path)
        shape = self.model.input_shape
        self.size = (int(shape[1]), int(shape[2]))
        self.last_conv = self._find_last_conv()
        name = "EfficientNet-B0 (transfer learning)" if "efficientnet" in self.model.name.lower() or any(
            l.name == "top_activation" for l in self.model.layers) else "CNN (Keras)"
        self.kind = f"{name}, {self.size[0]}×{self.size[1]} input"

    def _find_last_conv(self):
        try:
            return self.model.get_layer("last_conv").name
        except ValueError:
            for layer in reversed(self.model.layers):
                try:
                    if len(layer.output.shape) == 4:
                        return layer.name
                except Exception:
                    continue
        return None

    def _array(self, img):
        # Resize EXACTLY like the training notebooks: tf.image.resize (bilinear), then uint8.
        # AI-image detectors rely on fine pixel patterns, so a different resize method
        # (e.g. PIL's) changes those patterns and can flip the prediction.
        arr = np.asarray(img.convert("RGB"), dtype="uint8")
        arr = self.tf.image.resize(arr, self.size).numpy()
        arr = np.clip(arr, 0, 255).astype("uint8")  # same as tf.cast(..., tf.uint8) in training
        # Normalisation (Rescaling / EfficientNet) is inside the model, so raw 0-255 pixels are passed
        return np.expand_dims(arr.astype("float32"), 0)

    def predict(self, img):
        return float(self.model.predict(self._array(img), verbose=0)[0][0])  # P(FAKE)

    def gradcam(self, img, prob_fake):
        """Grad-CAM: heatmap from the last conv layer's gradients. Returns PNG bytes (overlay)."""
        if not self.last_conv:
            return None
        tf = self.tf
        grad_model = self.keras.Model(self.model.inputs, [self.model.get_layer(self.last_conv).output,
                                                          self.model.output])
        x = tf.convert_to_tensor(self._array(img))
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            # Explain the class that was predicted
            score = preds[:, 0] if prob_fake >= 0.5 else 1.0 - preds[:, 0]
        grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return overlay_heatmap(img, heatmap)


class _OneDetector:
    """One pretrained HuggingFace AI-image classifier. Finds which output label means 'AI'."""

    AI_LABELS = ("artificial", "ai", "ai_gen", "ai image", "fake", "generated", "deepfake", "synthetic")

    def __init__(self, name, torch):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.name, self.torch = name, torch
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).eval()
        labels = {int(i): str(l).lower().strip() for i, l in self.model.config.id2label.items()}
        ai = [i for i, l in labels.items()
              if l in self.AI_LABELS or l.startswith("ai") or any(w in l for w in ("artificial", "fake", "generated"))]
        if not ai:
            raise ValueError(f"{name}: cannot tell which label means AI ({labels})")
        self.ai_index = ai[0]

    def probs(self, imgs):
        with self.torch.no_grad():
            x = self.processor(images=[i.convert("RGB") for i in imgs], return_tensors="pt")
            p = self.torch.softmax(self.model(**x).logits, dim=-1)[:, self.ai_index]
        return p.numpy()


class HFImageDetector:
    """M4: ensemble of pretrained AI-generated image detectors from HuggingFace.

    One detector alone is often over-confident on ordinary real photos (dark webcam pictures,
    phone photos with heavy processing). Several detectors trained on different data make
    different mistakes, so the final P(AI) is the MEDIAN of their scores: a single detector
    cannot call a real photo 'AI' on its own. Each detector also looks at the full image and
    up to 5 crops (median), so one odd region cannot flip the result.
    Runs locally (no API); every model is downloaded once on first use and then cached.
    A model that cannot be loaded is skipped, so the app keeps working with the others."""

    def __init__(self, path):
        import json

        import torch

        cfg = json.loads(open(path, encoding="utf-8").read())
        names = cfg.get("models") or [cfg["model"]]
        self.detectors, self.skipped = [], {}
        for name in names:
            try:
                self.detectors.append(_OneDetector(name, torch))
            except Exception as e:  # no internet, unsupported model, unknown labels ...
                self.skipped[name] = str(e)[:200]
        if not self.detectors:
            raise RuntimeError(f"no AI-image detector could be loaded: {self.skipped}")
        n = len(self.detectors)
        self.kind = (f"Ensemble of {n} pretrained detectors (median)" if n > 1
                     else f"Pretrained detector ({self.detectors[0].name})")
        self.last_scores = {}

    @staticmethod
    def _views(img, max_side=1024, crop=512):
        """Full image + up to 5 crops (centre and 4 corners).

        Phone photos are huge (e.g. 4000x3000). Squeezing the whole photo into the
        model's 224x224 input makes it look very smooth, and smooth = "AI" for these
        detectors. Crops keep the real camera texture (noise, grain)."""
        from PIL import Image

        img = img.convert("RGB")
        if max(img.size) > max_side:
            img = img.copy()
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        views = [img]
        w, h = img.size
        if min(w, h) >= int(crop * 1.25):  # only crop when the image is big enough
            for x, y in [((w - crop) // 2, (h - crop) // 2), (0, 0), (w - crop, 0), (0, h - crop), (w - crop, h - crop)]:
                views.append(img.crop((x, y, x + crop, y + crop)))
        return views

    def _probs(self, imgs):
        """P(AI) for each image = median over the detectors."""
        return np.median(np.stack([d.probs(imgs) for d in self.detectors]), axis=0)

    def predict(self, img):
        """P(AI-generated) = median over detectors of (median over full image + crops).
        The score of every detector is kept in last_scores to show in the app."""
        views = self._views(img)
        self.last_scores = {d.name: float(np.median(d.probs(views))) for d in self.detectors}
        return float(np.median(list(self.last_scores.values())))

    def gradcam(self, img, prob_fake, grid=3):
        """Occlusion map: grey out each of 3×3 regions and measure how much P(AI) changes.
        Red = regions that drive the decision."""
        from PIL import ImageDraw

        img = img.convert("RGB")
        w, h = img.size
        masked = []
        for r in range(grid):
            for c in range(grid):
                m = img.copy()
                ImageDraw.Draw(m).rectangle([c * w // grid, r * h // grid, (c + 1) * w // grid, (r + 1) * h // grid],
                                            fill=(128, 128, 128))
                masked.append(m)
        heat = np.abs(prob_fake - self._probs(masked)).reshape(grid, grid)
        heat = heat / heat.max() if heat.max() > 0 else heat
        return overlay_heatmap(img, heat)


def overlay_heatmap(img, heatmap, size=256, alpha=0.45):
    from matplotlib import colormaps
    from PIL import Image

    base = img.convert("RGB").resize((size, size))
    hm = Image.fromarray(np.uint8(heatmap * 255)).resize((size, size), Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(hm) / 255.0)[:, :, :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    out = Image.blend(base, colored, alpha)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
