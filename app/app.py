"""
AICES - AI Content Integrity Evaluation System
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import io
import json

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from core import analyzers, registry
from core.explain import HAS_LIME
from core.scoring import WEIGHTS, overall_score

# Baseline only for now. Set to True once advanced (transformer) models are trained.
SHOW_ADVANCED = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit


def _versions(requested):
    if not SHOW_ADVANCED:
        return ["baseline"]
    if requested == "compare":
        return list(registry.VERSIONS)
    return [requested if requested in registry.VERSIONS else "baseline"]


def evaluate_text(question, answer, reference, version):
    modules = {
        "bias": analyzers.analyze_bias(question, answer, version),
        "hallucination": analyzers.analyze_hallucination(question, answer, reference, version),
        "privacy": analyzers.analyze_privacy(answer, version),
    }
    return {"version": version, "overall": overall_score(modules), "modules": modules}


@app.get("/")
def index():
    return render_template("index.html", show_advanced=SHOW_ADVANCED)


@app.get("/api/status")
def api_status():
    return jsonify({"models": registry.status(), "lime_installed": HAS_LIME, "show_advanced": SHOW_ADVANCED,
                    "code_version": analyzers.CODE_VERSION})


@app.get("/api/metrics")
def api_metrics():
    """Training metrics from models/baseline/metrics.json (Benchmarks tab)."""
    path = registry.MODELS_DIR / "baseline" / "metrics.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.pop("_note", None)
    return jsonify({"metrics": data, "weights": WEIGHTS})


@app.post("/api/analyze-text")
def api_analyze_text():
    data = request.get_json(silent=True) or {}
    answer = (data.get("answer") or "").strip()
    if not answer:
        return jsonify({"error": "please paste the AI-generated text"}), 400
    question = (data.get("question") or "").strip()
    reference = (data.get("reference") or "").strip()
    results = [evaluate_text(question, answer, reference, v) for v in _versions(data.get("version", "baseline"))]
    return jsonify({"results": results})


@app.post("/api/analyze-image")
def api_analyze_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "please upload an image"}), 400
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "this is not a valid image file"}), 400
    versions = _versions(request.form.get("version", "baseline"))
    return jsonify({"results": [{"version": v, "deepfake": analyzers.analyze_image(img, v)} for v in versions]})


if __name__ == "__main__":
    app.run(debug=True)
