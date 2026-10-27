# Baseline models

These files are produced by the training notebooks in `training/`. Download them and place them here:

| File | Notebook | Status |
|---|---|---|
| `m1a_identity_bias.joblib` | m1a_identity_bias.ipynb | trained on Kaggle |
| `m1b_framing_bias.joblib` | m1b_framing_bias.ipynb | trained on Kaggle |
| `m1c_stance.joblib` | m1c_stance.ipynb | trained on Kaggle (fallback for one-sided bias) |
| `m1c_stance_nli.json` | — | included: points to the pretrained NLI model (one-sided bias + contradiction check) |
| `m2_hallucination.joblib` | m2_hallucination.ipynb | **already trained** (HaluEval, test accuracy 85.2%) |
| `m3_privacy.joblib` | m3_privacy.ipynb | trained on Kaggle |
| `m4_hf_detector.json` | — | included: points to the pretrained HuggingFace detector |

`metrics.json` holds the test-set numbers shown in the app's Benchmarks tab. Update it after training.

If `m2_hallucination.joblib` fails to load (different scikit-learn version), run `m2_hallucination.ipynb`
on your laptop; it takes about a minute.
