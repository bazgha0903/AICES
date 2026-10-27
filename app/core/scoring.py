"""
Final Integrity Score: WEIGHTED HARMONIC MEAN.

    Integrity Score = Σ wᵢ / Σ (wᵢ / Sᵢ)

Every module returns a 0-100 score (100 = clean).
The harmonic mean penalises the worst module the most: if even one dimension is
very bad (e.g. an Aadhaar leak), the overall score drops, even when the other
modules look fine. That is the behaviour an integrity check needs.

Example: Bias 90, Hallucination 90, Privacy 10
  weighted average  -> 66 (Needs review)   <- the leak is hidden
  weighted harmonic -> 26 (High risk)      <- the leak is caught
"""

WEIGHTS = {"bias": 0.35, "hallucination": 0.35, "privacy": 0.30}
MIN_SCORE = 1.0  # floor at 1 to avoid division by zero when a score is 0


def verdict(score):
    if score >= 80:
        return "Trustworthy"
    if score >= 60:
        return "Needs review"
    return "High risk"


def weighted_harmonic_mean(scores, weights):
    total_w = sum(weights[k] for k in scores)
    return total_w / sum(weights[k] / max(scores[k], MIN_SCORE) for k in scores)


def overall_score(modules):
    """Modules that are unavailable (model missing / no evidence found) are left out,
    and the remaining weights are renormalised."""
    used = {k: w for k, w in WEIGHTS.items() if modules.get(k, {}).get("available")}
    if not used:
        return {"score": None, "verdict": "No models available", "weights_used": {}, "method": "weighted harmonic mean"}
    total = sum(used.values())
    score = weighted_harmonic_mean({k: modules[k]["score"] for k in used}, used)
    return {
        "score": round(score, 1),
        "verdict": verdict(score),
        "weights_used": {k: round(w / total, 3) for k, w in used.items()},
        "method": "weighted harmonic mean",
    }
