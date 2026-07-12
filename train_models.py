"""
train_models.py
---------------
Trains three model families on the triage data and saves artifacts the
Streamlit app loads at runtime:

    models/xgb_model.joblib
    models/lgbm_model.joblib
    models/tabnet_model.zip        (TabNet's own save format)
    models/metrics.json            (accuracy / F1 / per-class report / confusion mtx)
    models/feature_importance.json (for dashboard charts)
    models/eda_stats.json          (precomputed aggregates -> fast dashboard)

Run:  python train_models.py --csv silent_sting_triage_data.csv

TabNet on 1M rows is slow on CPU, so by default we train it on a stratified
sample (--tabnet-sample). Trees train on the full data. Set --tabnet-sample 0
to train TabNet on everything.

NOTE: the Streamlit app's TabNet path also needs models/tabnet_scaler.json,
which the standalone train_tabnet.py writes. Run train_tabnet.py to produce
the deployed TabNet + its scaler.
"""

import argparse, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_prep import load_and_prepare, CLASS_ORDER, FEATURE_ORDER

warnings.filterwarnings("ignore")


def eval_model(name, model, X_te, y_te, tabnet=False):
    Xin = X_te.values if tabnet else X_te
    y_pred = model.predict(Xin)
    if tabnet:
        y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    rep = classification_report(
        y_te, y_pred, target_names=CLASS_ORDER, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred).tolist()
    print(f"  {name:10s}  acc={acc:.4f}  macroF1={f1m:.4f}")
    return {"accuracy": acc, "macro_f1": f1m, "report": rep, "confusion_matrix": cm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="silent_sting_triage_data.csv")
    ap.add_argument("--tabnet-sample", type=int, default=120_000,
                    help="rows to train TabNet on (0 = full data)")
    ap.add_argument("--tabnet-epochs", type=int, default=30)
    args = ap.parse_args()

    print("Loading + preprocessing ...")
    X, y, raw = load_and_prepare(args.csv)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  train={X_tr.shape}  test={X_te.shape}")

    metrics, importances = {}, {}

    # ---------------- XGBoost ----------------
    print("\nTraining XGBoost ...")
    t = time.time()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.15,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        objective="multi:softprob", num_class=len(CLASS_ORDER),
        n_jobs=-1, random_state=42, eval_metric="mlogloss",
    )
    xgb.fit(X_tr, y_tr)
    print(f"  trained in {time.time()-t:.1f}s")
    metrics["XGBoost"] = eval_model("XGBoost", xgb, X_te, y_te)
    importances["XGBoost"] = dict(zip(FEATURE_ORDER, xgb.feature_importances_.tolist()))
    joblib.dump(xgb, "models/xgb_model.joblib")

    # ---------------- LightGBM ----------------
    # NOTE: subsample needs subsample_freq>=1 in LightGBM or it is a no-op.
    print("\nTraining LightGBM ...")
    t = time.time()
    lgbm = LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.1,
        min_child_samples=50, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, objective="multiclass",
        n_jobs=-1, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    print(f"  trained in {time.time()-t:.1f}s")
    metrics["LightGBM"] = eval_model("LightGBM", lgbm, X_te, y_te)
    importances["LightGBM"] = dict(zip(FEATURE_ORDER, lgbm.feature_importances_.tolist()))
    joblib.dump(lgbm, "models/lgbm_model.joblib")

    # ---------------- TabNet (deep learning) ----------------
    print("\nTraining TabNet (PyTorch) ...")
    from pytorch_tabnet.tab_model import TabNetClassifier
    if args.tabnet_sample and args.tabnet_sample < len(X_tr):
        Xt, _, yt, _ = train_test_split(
            X_tr, y_tr, train_size=args.tabnet_sample,
            random_state=42, stratify=y_tr)
        print(f"  (sampled {len(Xt):,} rows for CPU speed)")
    else:
        Xt, yt = X_tr, y_tr
    t = time.time()
    tabnet = TabNetClassifier(seed=42, verbose=0)
    tabnet.fit(
        Xt.values, yt.values,
        eval_set=[(X_te.values, y_te.values)], eval_metric=["accuracy"],
        max_epochs=args.tabnet_epochs, patience=8, batch_size=8192,
        virtual_batch_size=1024,
    )
    print(f"  trained in {time.time()-t:.1f}s")
    metrics["TabNet"] = eval_model("TabNet", tabnet, X_te, y_te, tabnet=True)
    importances["TabNet"] = dict(zip(FEATURE_ORDER, np.asarray(tabnet.feature_importances_).tolist()))
    tabnet.save_model("models/tabnet_model")  # -> models/tabnet_model.zip

    # ---------------- persist artifacts ----------------
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("models/feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    # Precompute EDA aggregates so the dashboard never touches the 54MB CSV.
    eda = compute_eda_stats(raw)
    with open("models/eda_stats.json", "w") as f:
        json.dump(eda, f, indent=2)

    print("\nBest model:",
          max(metrics, key=lambda m: metrics[m]["accuracy"]),
          "-> saved all artifacts to models/")


def compute_eda_stats(raw: pd.DataFrame) -> dict:
    """Aggregates the dashboard renders as charts (kept small, JSON-safe)."""
    df = raw.copy()
    df["Swelling_State"] = df["Local_Swelling"].fillna("Not Recorded")
    num_cols = ["Age", "Time_Since_Bite_Min", "Heart_Rate_BPM", "Blood_Pressure_Systolic"]

    def hist(series, bins=30):
        counts, edges = np.histogram(series.dropna(), bins=bins)
        return {"counts": counts.tolist(), "edges": edges.tolist()}

    stats = {
        "n_rows": int(len(df)),
        "class_counts": df["Bite_Source_Target"].value_counts().to_dict(),
        "gender_counts": df["Gender"].value_counts().to_dict(),
        "swelling_counts": df["Swelling_State"].value_counts().to_dict(),
        "paralysis_by_class": pd.crosstab(
            df["Bite_Source_Target"], df["Muscle_Paralysis_Present"]).to_dict(),
        "coag_by_class": pd.crosstab(
            df["Bite_Source_Target"], df["Blood_Coagulation_Failure"]).to_dict(),
        "swelling_by_class": pd.crosstab(
            df["Bite_Source_Target"], df["Swelling_State"]).to_dict(),
        "mean_vitals_by_class": df.groupby("Bite_Source_Target")[num_cols].mean().round(2).to_dict(),
        "hist_by_class": {
            col: {cls: hist(df.loc[df["Bite_Source_Target"] == cls, col])
                  for cls in CLASS_ORDER}
            for col in ["Heart_Rate_BPM", "Blood_Pressure_Systolic"]
        },
        "overall_hist": {col: hist(df[col]) for col in num_cols},
        "corr": df[num_cols + ["Muscle_Paralysis_Present", "Blood_Coagulation_Failure"]]
                 .corr().round(3).to_dict(),
        "vitals_scatter_sample": df.sample(min(4000, len(df)), random_state=1)[
            ["Heart_Rate_BPM", "Blood_Pressure_Systolic", "Bite_Source_Target"]
        ].to_dict(orient="list"),
    }
    return stats


if __name__ == "__main__":
    main()
