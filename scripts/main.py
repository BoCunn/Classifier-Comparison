"""
Adult Income Classification
===========================
Predicts whether income is >50K or <=50K using:
  - k-Nearest Neighbors  (KNN, k=11)
  - Support Vector Machine (LinearSVC + Platt calibration)
  - Random Forest         (200 trees)

All three models use IDENTICAL 75/25 train/test splits (random_state=42).

Usage:
    python income_classifier_final.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)

# ── 1. Column names ────────────────────────────────────────────────────────────

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# ── 2. Load ────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  ADULT INCOME CLASSIFICATION")
print("  Models: KNN  |  SVM  |  Random Forest")
print("=" * 65)

train_df = pd.read_csv(
    "data/adult.data",
    header=None, names=COLUMNS,
    sep=r",\s*", engine="python",
    na_values=["?"],
)
test_df = pd.read_csv(
    "data/adult.test",
    header=None, names=COLUMNS,
    sep=r",\s*", engine="python",
    na_values=["?"],
    skiprows=1,          # first row is a comment in the .test file
)

df = pd.concat([train_df, test_df], ignore_index=True)
print(f"\n  Rows loaded   : {len(df):,}")
print(f"  Missing cells : {df.isnull().sum().sum():,}")

# ── 3. Pre-process ─────────────────────────────────────────────────────────────

df.dropna(inplace=True)
print(f"  Rows after dropna: {len(df):,}")

df["income"] = df["income"].str.rstrip(".")          # remove trailing dots
df["income_binary"] = (df["income"] == ">50K").astype(int)

X_raw = df.drop(columns=["income", "income_binary"])
y     = df["income_binary"]

le = LabelEncoder()
for col in X_raw.select_dtypes(include="object").columns:
    X_raw[col] = le.fit_transform(X_raw[col])

X = X_raw.values
y = y.values

# ── 4. Identical split for all models ─────────────────────────────────────────

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=SEED,
    stratify=y,
)

print(f"\n  Train set : {len(X_train):,} rows")
print(f"  Test  set : {len(X_test):,} rows")
print(f"  Income >50K in test: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# Scale once — KNN and SVM need it; RF doesn't but it doesn't hurt
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Models ──────────────────────────────────────────────────────────────────

models = {
    "k-Nearest Neighbors (k=11)": {
        "clf":    KNeighborsClassifier(n_neighbors=11, metric="euclidean", n_jobs=-1),
        "scaled": True,
    },
    "Spprt Vec Machine (LinearSVC)": {
        # LinearSVC is O(n) in training — ideal for 30k+ rows.
        # CalibratedClassifierCV adds Platt scaling so we get probabilities.
        "clf":    CalibratedClassifierCV(
                      LinearSVC(C=1.0, max_iter=2000, random_state=SEED),
                      cv=3,
                  ),
        "scaled": True,
    },
    "Random Forest (200 trees)": {
        "clf":    RandomForestClassifier(
                      n_estimators=200,
                      random_state=SEED,
                      n_jobs=-1,
                  ),
        "scaled": False,   # tree ensembles don't need scaling
    },
}

results = {}

for name, cfg in models.items():
    print(f"\n{'─'*55}")
    print(f"  ▶ {name}")
    print(f"{'─'*55}")

    X_tr = X_train_sc if cfg["scaled"] else X_train
    X_te = X_test_sc  if cfg["scaled"] else X_test

    t0 = time.time()
    cfg["clf"].fit(X_tr, y_train)
    train_sec = time.time() - t0

    t0 = time.time()
    y_pred = cfg["clf"].predict(X_te)
    y_prob = cfg["clf"].predict_proba(X_te)[:, 1]
    pred_sec = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = dict(
        Accuracy=acc, Precision=prec, Recall=rec,
        F1=f1, ROC_AUC=auc,
        Train_s=train_sec, Pred_s=pred_sec,
    )

    print(f"  Accuracy  {acc:.4f}   F1  {f1:.4f}   ROC-AUC  {auc:.4f}")
    print(f"  Train {train_sec:.1f}s   Predict {pred_sec:.2f}s")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"            Pred <=50K   Pred >50K")
    print(f"  Act <=50K   {cm[0,0]:6d}      {cm[0,1]:6d}")
    print(f"  Act  >50K   {cm[1,0]:6d}      {cm[1,1]:6d}")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred,
                                target_names=["<=50K", ">50K"],
                                digits=4))

# ── 6. Comparison table ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  PERFORMANCE COMPARISON TABLE  (identical 75/25 split, seed=42)")
print("=" * 65)

COL = 26
header = f"{'Metric':<18}" + "".join(f"{n[:COL]:<{COL}}" for n in results)
print(header)
print("-" * (18 + COL * len(results)))

rows = [
    ("Accuracy",     "Accuracy"),
    ("Precision",    "Precision"),
    ("Recall",       "Recall"),
    ("F1 Score",     "F1"),
    ("ROC-AUC",      "ROC_AUC"),
    ("Train (sec)",  "Train_s"),
    ("Predict (sec)","Pred_s"),
]
for label, key in rows:
    vals = [results[n][key] for n in results]
    best_idx = int(np.argmax(vals)) if key not in ("Train_s", "Pred_s") else int(np.argmin(vals))
    row = f"{label:<18}"
    for i, (n, v) in enumerate(zip(results, vals)):
        cell = f"{v:.4f}" if key not in ("Train_s", "Pred_s") else f"{v:.2f}s"
        marker = " ◀" if i == best_idx else "  "
        row += f"{cell+marker:<{COL}}"
    print(row)

print("\n  ◀ = best value in that row\n")

# ── 7. Feature importance ──────────────────────────────────────────────────────

rf_clf = models["Random Forest (200 trees)"]["clf"]
fi = (pd.DataFrame({"Feature": X_raw.columns,
                     "Importance": rf_clf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True))

print("=" * 65)
print("  TOP FEATURE IMPORTANCES  (Random Forest)")
print("=" * 65)
for _, row in fi.iterrows():
    bar = "█" * int(row["Importance"] * 180)
    print(f"  {row['Feature']:<20} {row['Importance']:.4f}  {bar}")

print("\n✅  Done.\n")