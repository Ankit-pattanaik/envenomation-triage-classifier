# 🦂 Silent Sting — Envenomation Triage Intelligence

End-to-end ML project on a 1,000,000-patient envenomation triage dataset:
proper preprocessing, EDA, three model families (XGBoost, LightGBM, TabNet/PyTorch),
and a polished Streamlit app with a 12-chart dashboard, a live prediction section,
and Groq-powered AI insights.

## What the data says (the analysis that drives everything)

Target: `Bite_Source_Target` ∈ {Harmless_Insect, Scorpion, Viper_Snake, Black_Widow_Spider}.

| Culprit | Signature in the data |
|---|---|
| 🕷️ Black Widow | `Local_Swelling` **always missing** + muscle paralysis (~95%) + high BP (~150) |
| 🐍 Viper Snake | coagulation failure (~84%) + low BP (~89) + severe/medium swelling |
| 🦂 Scorpion | high heart rate (~145) + elevated BP + mild swelling |
| 🐜 Harmless Insect | everything mild/normal, low HR (~79) |

`Age` and `Time_Since_Bite_Min` are noise (identical means across classes).

**Critical preprocessing choice:** the 150,157 missing `Local_Swelling` values are *not*
dropped — the missingness is exactly the Black-Widow rows, so we encode it as the
`Swelling_Missing` feature. That single decision is worth ~15% accuracy.

## Model results (20% held-out test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TabNet (PyTorch) | ~99.7% | ~0.997 |
| XGBoost | ~99.7% | ~0.997 |
| LightGBM | ~97.2% | ~0.969 |

## ⚠️ About the `models/` folder

This copy was delivered through a **text-only** file bridge, so the trained
**binary** artifacts are NOT here yet:

    models/xgb_model.joblib
    models/lgbm_model.joblib
    models/tabnet_model.zip
    models/eda_stats.json      (large — dashboard needs this)

Get them one of two ways:
1. **Copy** them from the downloaded `snakebite_triage.zip` → its `models/` folder
   into `C:\mcp\snakebite_triage\models\`, **or**
2. **Regenerate** them (see below). The small JSONs (`metrics.json`,
   `feature_importance.json`, `tabnet_scaler.json`) are already here.

## Setup

```bash
pip install -r requirements.txt
```

## Train the models (regenerates every artifact)

Place `silent_sting_triage_data.csv` in this folder, then:

```bash
python train_models.py --csv silent_sting_triage_data.csv   # XGB + LGBM (full data) + eda_stats
python train_tabnet.py                                      # deployed TabNet + scaler
```

TabNet on 1M rows is slow on CPU, so it trains on a stratified sample by default
(`--tabnet-sample 120000`); pass `--tabnet-sample 0` to use the full data.

## Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Groq AI insights (optional but recommended)

1. Get a free key at https://console.groq.com/keys
2. Paste it into the sidebar (or `set GROQ_API_KEY=...` before launching).
3. Pick a model in the sidebar. Groq rotates model IDs often — if one errors, try
   another (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `llama-3.1-8b-instant`).

## Files

| File | Purpose |
|---|---|
| `data_prep.py` | Preprocessing + feature engineering (shared by training & app) |
| `train_models.py` | Trains XGBoost, LightGBM, TabNet; saves metrics & EDA stats |
| `train_tabnet.py` | Standalone lean TabNet trainer (writes the scaler the app needs) |
| `app.py` | Streamlit UI: dashboard + prediction + Groq insights |
| `models/` | Saved models, metrics, feature importances, EDA aggregates |

## ⚕️ Disclaimer

Educational ML demo on synthetic data. Not a medical device and must not be used for
real clinical decisions.
