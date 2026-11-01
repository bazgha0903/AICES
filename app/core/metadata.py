"""
M4 helper: read hidden image metadata (EXIF / PNG text / XMP).

Why: the pretrained AI-image detector sometimes calls real phone photos "AI"
(phones apply HDR, smoothing and sharpening, which look like AI patterns).
Metadata gives an extra, independent signal:

  * camera_info  -> the photo has camera EXIF (brand, model, date taken)
                    = evidence that it came from a real camera
  * ai_tool      -> the file carries traces of an AI generator
                    (Stable Diffusion / ComfyUI prompt, C2PA "AI-generated" tag, etc.)
                    = strong evidence that it is AI-generated

Metadata can be removed or faked, so it is a supporting signal, not proof.
"""

# EXIF tag ids
_MAKE, _MODEL, _SOFTWARE, _EXIF_IFD, _DATE_TAKEN = 271, 272, 305, 0x8769, 36867

# Words that AI generators write into the file (PNG text chunks, XMP, Software tag)
_AI_WORDS = (
    "stable diffusion", "midjourney", "dall-e", "dall·e", "dalle", "comfyui", "automatic1111",
    "novelai", "adobe firefly", "leonardo.ai", "trainedalgorithmicmedia",
)
# PNG text keys written by Stable Diffusion WebUI / ComfyUI
_AI_PNG_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "dream")


def read_metadata(img):
    """PIL image (opened from the original upload) -> dict with camera / AI-tool info."""
    out = {"camera": None, "date_taken": None, "ai_tool": None}

    try:
        exif = img.getexif()
    except Exception:
        exif = {}

    make = str(exif.get(_MAKE, "") or "").strip("\x00 ").strip()
    model = str(exif.get(_MODEL, "") or "").strip("\x00 ").strip()
    if make or model:
        # avoid "Apple Apple iPhone 13" when the model already contains the brand
        out["camera"] = model if make and model.lower().startswith(make.lower()) else f"{make} {model}".strip()

    try:
        date = exif.get_ifd(_EXIF_IFD).get(_DATE_TAKEN)
        out["date_taken"] = str(date) if date else None
    except Exception:
        pass

    # Collect every text place where a generator could leave a trace
    texts = [str(exif.get(_SOFTWARE, "") or "")]
    info = getattr(img, "info", {}) or {}
    for key, value in info.items():
        if isinstance(key, str) and key.lower() in _AI_PNG_KEYS and value:
            out["ai_tool"] = f"AI generator settings found in the file ('{key}')"
        if isinstance(value, (str, bytes)):
            texts.append(value.decode("utf-8", "ignore") if isinstance(value, bytes) else value)

    if out["ai_tool"] is None:
        blob = " ".join(texts).lower()
        hit = next((w for w in _AI_WORDS if w in blob), None)
        if hit:
            out["ai_tool"] = f"AI tool name found in the file metadata ('{hit}')"

    return out
