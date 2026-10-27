# Advanced (upgrade) models

The baseline version lives in `models/baseline/` and is never removed. Upgraded models go here.
Once `SHOW_ADVANCED = True` is set in `app/app.py`, the **Advanced** and **Compare** modes use them automatically.

| Module | What to put here | How to build it |
|---|---|---|
| M1-A Identity bias | folder `m1a_identity_bias/` | Fine-tune DistilBERT → `model.save_pretrained()` + `tokenizer.save_pretrained()` |
| M1-B Framing bias | folder `m1b_framing_bias/` | Fine-tune RoBERTa-base (same way) |
| M1-C Stance | folder `m1c_stance/` | DeBERTa-v3, input pair = (topic, sentence) |
| M2 Hallucination | folder `m2_hallucination/` | DeBERTa-v3 NLI, input pair = (knowledge, question + answer) |
| M3 Privacy | folder `m3_privacy/` | DistilBERT token classification (NER) |
| M4 Deepfake | file `m4_deepfake_efficientnet.keras` | EfficientNet-B0 transfer learning (Keras) |

**Rules (so the app works without code changes):**
- Classifiers: **label 1 = positive** (biased / PRO / hallucinated), label 0 = negative.
- NER labels should follow ai4privacy (FIRSTNAME, EMAIL, ...); the app maps them to groups automatically.
- The Keras image model must include its `Rescaling` layer, with output `Dense(1, sigmoid)` = P(FAKE).
- Install: `pip install torch transformers`.
