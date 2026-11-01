# AICES: AI Content Integrity Evaluation System

AICES checks AI-generated content before you trust it. Paste an AI answer (or upload an image) and it reports:

| Module | What it detects | Model | Training data |
|---|---|---|---|
| **M1 Bias** | **Identity** bias (stereotypes about gender, religion, race, caste) | TF-IDF + Logistic Regression **+** identity-group lexicon | Jigsaw Unintended Bias |
| | **Framing** bias (loaded / opinionated language) | TF-IDF + Logistic Regression | BABE |
| | **One-sided** arguments (only one side of a debate) | Pretrained NLI model, zero-shot (fallback: TF-IDF + LR) | IBM Claim Stance (fallback) |
| **M2 Hallucination** | facts not supported by, or contradicting, the evidence | 3 knowledge-overlap features + Logistic Regression **+** NLI contradiction check | HaluEval (QA) |
| **M3 Privacy** | names, emails, phones, addresses, Aadhaar, PAN, ... | Word-level Logistic Regression tagger **+** regex for Indian IDs | ai4privacy |
| **M4 AI-generated image** | how suspicious an image is of being AI-generated | Ensemble of 3 pretrained detectors (Swin, SwinV2, SigLIP) **+** face CNN check + camera-metadata check | pretrained · 140k Real and Fake Faces (face CNN) |
| **XAI** | why each decision was made | LIME / LR word weights, feature contributions, highlighted spans, occlusion map | — |

The three text modules are combined into one **Integrity Score** (weighted harmonic mean) with a verdict:
**Trustworthy** (≥ 80), **Needs review** (60–79) or **High risk** (< 60).

Everything is free and runs locally: no paid API, no API key. Evidence for fact checking comes from the
user's reference text or from the free Wikipedia API.

---

## Quick start

```bash
cd aices/app
python -m venv venv
venv\Scripts\activate            # Windows  (Mac / Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**.

- The pretrained models (NLI and image detectors) are downloaded once from HuggingFace on first use.
- The face model `m4_deepfake_cnn.keras` is trained with `training/m4_deepfake_faces.ipynb` (Kaggle) and
  goes into `app/models/baseline/`.
- **Load sample** fills in a demo text. The ☀ / ☾ button switches dark / light mode.
- The **Pipeline & Models**, **Benchmarks & Formula** and **History** tabs explain the system.

### Automated tests

With the app running, open a second terminal:

```bash
cd aices/app
venv\Scripts\activate
python run_tests.py
```

`run_tests.py` sends 10 test cases to the API (correct facts, wrong facts, Wikipedia check, gender and
religion stereotypes, one-sided and balanced arguments, Indian personal data, an insult with no group, and a
mixed sample) and prints a PASS / FAIL table. **Current result: 10 / 10 pass.**

---

## How it works

```
text → split into sentences → M1 Bias · M2 Hallucination · M3 Privacy → explanations → Integrity Score
image → M4 detectors (+ face check) → suspicion % + occlusion map (shown separately)
```

### M1 Bias

- **Identity** (hybrid): the ML model scores each sentence, and a lexicon of identity groups (gender,
  religion, race / ethnicity / caste, age / disability) decides whether a group is mentioned at all.
  A sentence counts only if it mentions a group; a sweeping claim about a group ("women are ...",
  "muslims always ...") counts as at least 80% risk. An insult with no group ("people who disagree are
  stupid") is framing, not identity bias.
- **Framing**: a sentence is tagged only when the model is at least 70% confident.
- **One-sided**: only for debate questions ("Should ...?", "Is X good?"). A factual question ("When was
  ...?") has one correct answer, not two sides, so it is skipped. Each sentence is classified as PRO, CON or
  neutral by a pretrained NLI model (`cross-encoder/nli-distilroberta-base`), which works on any topic.
  One-sidedness = `|PRO − CON| / (PRO + CON) × min(1, (PRO + CON) / 2)`.
- Confidence calibration for identity and framing: `risk = max(0, (p − 0.5) / 0.5)`, so borderline
  guesses do not lower the score.
- **Bias score** = `100 × (1 − max(1.0 × identity, 0.8 × framing, 0.6 × one-sided))`.
  The strongest bias decides (an average would hide one clear stereotype), and bias types are weighted by
  severity: identity bias harms a group of people, a one-sided argument is incomplete but not false.

### M2 Hallucination

- Only sentences with a checkable fact (a number or a name) are checked. Opinions are left to the bias
  module, and contact details to the privacy module.
- **Trained model**: 3 features (share of answer words found in the evidence, unseen numbers, unseen names)
  + Logistic Regression, trained on HaluEval. It catches wrong numbers and names (e.g. "built in 1925").
- **Contradiction check**: the overlap model cannot see a contradiction when the wrong word also appears in
  the evidence ("Sydney" in the Australia article). A pretrained NLI model checks the most similar evidence
  sentences. A contradiction counts only if (1) no evidence sentence supports the claim, (2) a name or number
  of the claim is missing from the contradicting sentence, and (3) that sentence is about the same topic
  (shares a content word such as "capital"). This avoids false alarms on paraphrases.
- Score = `100 × (1 − average P(hallucinated))`.

### M3 Privacy

- A word-level classifier labels each word (NAME, EMAIL, PHONE, ADDRESS, ID_NUMBER, ...).
- Regex patterns for Aadhaar, PAN and Indian mobile numbers (missing from ai4privacy) override an
  overlapping model label. Plain years and short numbers are not counted as personal data.
- Score = `100 − penalties` (ID / financial 35, email / phone 20, address / date of birth / online 15, name 10).

### Integrity Score

**Weighted harmonic mean** = `Σwᵢ / Σ(wᵢ / Sᵢ)` with weights Bias 0.35, Hallucination 0.35, Privacy 0.30.
The harmonic mean is pulled down by the weakest module: with Bias 90, Hallucination 90 and Privacy 10 a
simple average gives 66, the harmonic mean gives 26. One leaked phone number makes content untrustworthy.

### M4 AI-generated image

Three pretrained detectors trained on different data look at the full image and up to 5 crops:
`Organika/sdxl-detector` (Swin), `haywoodsloan/ai-image-detector-deploy` (SwinV2) and
`Ateeqq/ai-vs-human-image-detector` (SigLIP), set in `app/models/baseline/m4_hf_detector.json`.

- **Final P(AI) = median of the three**, so one over-confident detector cannot decide alone
  (one detector alone called several real phone and webcam photos 94–100% AI; the median fixed most of them).
- Levels: Low < 30%, Uncertain 30–90%, High ≥ 90%. **High also needs every detector to be at least 80% sure**,
  because calling a real photo AI-generated is the worse mistake.
- Camera EXIF metadata (make, model, date) is read; a photo with camera metadata is never marked High,
  and traces of an AI tool in the file mark it High.
- **Face check** (only for an Uncertain result): OpenCV finds the faces, and a CNN trained by us on
  *140k Real and Fake Faces* (Keras, 128 × 128) scores each face. With several faces the median is used,
  so one dark or half-cut face cannot decide alone. If the faces get P(AI) < 20%, the result becomes
  **Likely real** (the badge then says "Low (face check)"). The face check can never turn a result into
  "AI-generated": the face CNN was trained on older StyleGAN faces, so it is used only to rescue real
  photos, not to accuse.
- The occlusion map greys out 3 × 3 regions one at a time; red regions change the prediction most.
- Licences differ per model (sdxl-detector is CC-BY-NC, non-commercial).

---

## Results

| Model | Test data | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| M2 Hallucination (overlap features + LR) | HaluEval QA, 20% held-out split, split by question | **85.2%** | 0.851 | 0.91 |

Feature weights of M2: `word_overlap −1.54`, `unseen_numbers +1.61`, `unseen_names +2.58`.
The other models print their test metrics in their training notebooks; the numbers shown in the app's
Benchmarks tab are read from `app/models/baseline/metrics.json`.

---

## Project structure

```
aices/
├── README.md
├── training/                    Kaggle notebooks, one per model (+ train_all_baseline)
│   └── src/                     the same notebooks as .py files, and build_notebooks.py
└── app/
    ├── app.py                   Flask server and REST API
    ├── run_tests.py             automated test suite (10 cases)
    ├── requirements.txt
    ├── core/
    │   ├── analyzers.py         logic of every module (bias, hallucination, privacy, image)
    │   ├── models.py            model wrappers (scikit-learn, HuggingFace, Keras)
    │   ├── metadata.py          camera / AI-tool EXIF metadata of images
    │   ├── registry.py          which model file each module loads
    │   ├── features.py          M2 and M3 features (identical copy in the training notebooks)
    │   ├── explain.py           LIME / word weights
    │   ├── scoring.py           Integrity Score (weighted harmonic mean)
    │   ├── wiki.py              Wikipedia evidence
    │   └── text_utils.py
    ├── models/baseline/         trained model files + metrics.json
    ├── models/advanced/         slot for future transformer models
    ├── templates/index.html
    └── static/ (app.js, style.css)
```

### REST API

| Method | Route | Body |
|---|---|---|
| GET | `/api/status` | which models are ready |
| GET | `/api/metrics` | benchmark numbers and score weights |
| POST | `/api/analyze-text` | JSON `{question, answer, reference, version}` |
| POST | `/api/analyze-image` | form-data `image`, `version` |

---

## Training the models (Kaggle, free)

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → choose a notebook from `training/`.
2. Add the dataset with **Add Input** and, where noted, turn on **Internet** / **GPU T4**.
3. **Run All**, then download the model file from **Output** into `app/models/baseline/`.

| Notebook | Data | Output |
|---|---|---|
| m1a_identity_bias | Kaggle: jigsaw-unintended-bias-in-toxicity-classification | `m1a_identity_bias.joblib` |
| m1b_framing_bias | HuggingFace: mediabiasgroup/BABE (Internet ON) | `m1b_framing_bias.joblib` |
| m1c_stance | HuggingFace: ibm-research/claim_stance (Internet ON) | `m1c_stance.joblib` (fallback model) |
| m2_hallucination | GitHub: HaluEval (Internet ON) | `m2_hallucination.joblib` |
| m3_privacy | HuggingFace: ai4privacy (Internet ON) | `m3_privacy.joblib` |

`train_all_baseline.ipynb` trains all of them in one run. Install the same scikit-learn version as on Kaggle
(`scikit-learn==1.6.1`), otherwise the `.joblib` files will not load.

`m4_deepfake_faces.ipynb` (Kaggle: 140k Real and Fake Faces, GPU T4) trains the face CNN of the face check
(`m4_deepfake_cnn.keras`). `m4_deepfake_cnn` (CNN on CIFAKE) and `m4_clip_ai_image` (CLIP + Logistic
Regression) are earlier experiments for the image module, kept for reference.

---

## What testing taught us

Each fix below came from a test case that failed:

- The first image model (CNN on CIFAKE) called a real photo 76% fake; Grad-CAM showed it was looking at the
  background (domain shift). A face-only CNN then failed on non-face images, so the app now uses a general
  pretrained detector.
- Smooth, evenly lit webcam selfies were called 81–95% AI by the general detectors. A rule that
  "High" needs every detector to be ≥ 80% sure turned them into Uncertain, and a face CNN trained on
  real and AI faces then marks them Likely real. In a group photo one dark face scored 28%, so the
  median over all faces is used instead of only the largest face.
- The identity model (trained on Jigsaw, a toxicity dataset) flagged insults with no group as identity bias,
  so an identity-group lexicon was added.
- The trained stance model called every sentence neutral on new topics, so one-sidedness now uses a
  pretrained NLI model, and it is skipped for factual questions.
- Averaging the three bias types hid a clear gender stereotype, so the strongest bias type now decides,
  weighted by severity.
- The overlap model could not see "Sydney is the capital of Australia" as wrong, so an NLI contradiction
  check was added, with topic and entity checks to avoid false alarms.

## Limitations

- English only.
- TF-IDF models look at words, not meaning: sarcasm, negation and paraphrases can be misjudged.
- Wikipedia search does not always return the right page for vague questions.
- No AI-image detector catches every generator; new generators and heavily compressed images can be missed.
  Smooth, evenly lit webcam selfies can look AI-generated to the detectors; the strong-agreement rule and
  the face check handle most of them. The face CNN only knows StyleGAN faces, and photos without a face
  rely on the detectors alone. The result is a suspicion level, not a final answer.

## Future scope

Fine-tuned transformers for every module (DistilBERT / RoBERTa for bias, DeBERTa NLI for hallucination,
DistilBERT NER for privacy), video deepfake detection and Hindi support. The `models/advanced/` slot and the
`SHOW_ADVANCED` switch in `app/app.py` are ready for this: every model exposes the same interface, so no app
code needs to change.

## Author

**Bazgha Arshi** · B.Tech CSE, Ramgarh Engineering College (2022–2026)
