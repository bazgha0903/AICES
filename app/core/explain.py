"""
Explainability (XAI) layer.

Text: LIME (if `lime` is installed), otherwise Logistic Regression word weights.
Image: Grad-CAM (KerasImageClassifier.gradcam in models.py).
"""
try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

LIME_SAMPLES = 300  # more = more accurate but slower


def explain_text(model, text, top_k=6):
    """Return {'method': ..., 'words': [{'feature','weight'}]} — positive weight pushes towards 'flagged'."""
    fn = model.lime_predict_fn() if hasattr(model, "lime_predict_fn") else None
    if HAS_LIME and fn is not None:
        pos = getattr(model, "_pos", 1)
        explainer = LimeTextExplainer(class_names=["clean", "flagged"], random_state=42)
        exp = explainer.explain_instance(text, fn, labels=(pos,), num_features=top_k,
                                         num_samples=LIME_SAMPLES)
        words = [{"feature": w, "weight": round(float(v), 4)} for w, v in exp.as_list(label=pos)]
        return {"method": "LIME", "words": words}
    words = model.explain(text, top_k=top_k) if hasattr(model, "explain") else []
    return {"method": "Logistic Regression word weights", "words": words}
