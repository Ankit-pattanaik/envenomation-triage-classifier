"""
train_tabnet.py
---------------
Standalone, lean TabNet trainer. This is the one that produces the DEPLOYED
TabNet used by the app, because it also standardizes the numeric features and
writes models/tabnet_scaler.json (which app.py needs at inference time).

Run:  python train_tabnet.py     (expects models/metrics.json + feature_importance.json
                                   to already exist from train_models.py)
"""

import json, numpy as np
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from data_prep import load_and_prepare, FEATURE_ORDER
from train_models import eval_model

X, y, raw = load_and_prepare("silent_sting_triage_data.csv")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerics for the neural net (index 0..3 are the continuous cols)
mu = X_tr.iloc[:, :4].mean(); sd = X_tr.iloc[:, :4].std().replace(0, 1)
Xtr = X_tr.copy(); Xte = X_te.copy()
Xtr.iloc[:, :4] = (Xtr.iloc[:, :4] - mu) / sd
Xte.iloc[:, :4] = (Xte.iloc[:, :4] - mu) / sd

# Stratified sample for CPU speed
Xs, _, ys, _ = train_test_split(Xtr, y_tr, train_size=60000, random_state=42, stratify=y_tr)

clf = TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, seed=42, verbose=1)
clf.fit(Xs.values, ys.values,
        eval_set=[(Xte.values, y_te.values)], eval_metric=["accuracy"],
        max_epochs=20, patience=6, batch_size=4096, virtual_batch_size=512)

res = eval_model("TabNet", clf, Xte, y_te, tabnet=True)
clf.save_model("models/tabnet_model")
json.dump({"mu": mu.tolist(), "sd": sd.tolist()}, open("models/tabnet_scaler.json", "w"))

m = json.load(open("models/metrics.json")); m["TabNet"] = res
json.dump(m, open("models/metrics.json", "w"), indent=2)
imp = json.load(open("models/feature_importance.json"))
imp["TabNet"] = dict(zip(FEATURE_ORDER, np.asarray(clf.feature_importances_).tolist()))
json.dump(imp, open("models/feature_importance.json", "w"), indent=2)
print("TabNet done + saved")
