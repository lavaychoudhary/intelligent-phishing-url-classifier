"""
Intelligent Phishing & Malicious URL Classifier - Training Pipeline
Trains a 300-estimator Random Forest classifier on URL lexical features,
integrating Tranco allowlist domain augmentation and saving compressed weights.
"""
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from features import FEATURE_NAMES, extract_features, normalize

DATA_PATH = BASE_DIR / "urldata.csv"
TRANCO_PATH = DASHBOARD_DIR / "top-1m.csv"
MODEL_SAVE_PATH = DASHBOARD_DIR / "phishing_model.pkl"
METRICS_SAVE_PATH = DASHBOARD_DIR / "metrics.json"

AUG_FROM_DATASET = 20000
AUG_FROM_TRANCO = 20000
MIN_INDIAN = 3000
SEED = 42
SAMPLE = int(os.environ.get("SAMPLE", 0))

PROBE = [
    "manavrachna.edu.in", "mru.edu.in", "iitd.ac.in", "iitb.ac.in", "ugc.gov.in",
    "isro.gov.in", "irctc.co.in", "sbi.co.in", "amazon.in", "flipkart.com",
    "google.com", "wikipedia.org", "github.com", "mit.edu", "bbc.co.uk"
]


def is_probe(host: str) -> bool:
    return any(host == p or host.endswith("." + p) for p in PROBE)


def host_of(url: str) -> str:
    return normalize(url).split("/")[0].split("?")[0].lower()


def main():
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, usecols=["url", "result"]).dropna().drop_duplicates(subset="url")
    if SAMPLE:
        df = df.sample(SAMPLE, random_state=SEED)
        print(f"Sampled {SAMPLE} rows for fast training run.")

    print(f"Total Rows: {len(df):,} | Malicious Ratio: {df['result'].mean():.2%}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["result"])

    rng = np.random.default_rng(SEED)
    pat = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$")

    ben_hosts = pd.Series([host_of(u) for u in train_df.loc[train_df["result"] == 0, "url"]]).drop_duplicates()
    ben_hosts = ben_hosts[ben_hosts.map(lambda h: bool(pat.match(h)) and not is_probe(h))]
    aug = list(ben_hosts.sample(min(AUG_FROM_DATASET, len(ben_hosts)), random_state=SEED))
    print(f"Added bare domains from benign training URLs: {len(aug):,}")

    tranco_hold = []
    if TRANCO_PATH.exists():
        tr = pd.read_csv(TRANCO_PATH, header=None, names=["rank", "domain"]).head(300000)
        tr["domain"] = tr["domain"].astype(str).str.lower()
        tr = tr[tr["domain"].map(lambda h: bool(pat.match(h)))]
        n_probe = int(tr["domain"].map(is_probe).sum())
        tr = tr[~tr["domain"].map(is_probe)]
        print(f"Probe domains found in Tranco and excluded from training: {n_probe}")
        indian = tr[tr["domain"].str.endswith(".in")]
        rest = tr[~tr["domain"].str.endswith(".in")]
        pick = pd.concat([
            indian.sample(min(MIN_INDIAN, len(indian)), random_state=SEED),
            rest.sample(min(AUG_FROM_TRANCO - min(MIN_INDIAN, len(indian)), len(rest)), random_state=SEED),
        ])["domain"].tolist()
        rng.shuffle(pick)
        n_hold = max(1, int(0.10 * len(pick)))
        tranco_hold, pick = pick[:n_hold], pick[n_hold:]
        aug += pick
        print(f"Added Tranco domains: {len(pick):,} train, {len(tranco_hold):,} held-out test")
    else:
        print("Note: top-1m.csv not found; training without external Tranco augmentation.")

    forms = ["{d}", "https://{d}", "https://www.{d}/", "http://{d}/"]
    aug_urls = [forms[i].format(d=d) for d, i in zip(aug, rng.integers(0, len(forms), len(aug)))]
    X_urls = pd.concat([train_df["url"], pd.Series(aug_urls)], ignore_index=True)
    y_train = np.concatenate([train_df["result"].values, np.zeros(len(aug_urls), dtype=int)])

    print("Extracting features from training set...")
    X_train = pd.DataFrame([extract_features(u) for u in X_urls], columns=FEATURE_NAMES)
    print("Extracting features from test set...")
    X_test = pd.DataFrame([extract_features(u) for u in test_df["url"]], columns=FEATURE_NAMES)
    y_test = test_df["result"].values

    print("\nTraining RandomForestClassifier (300 estimators, min_samples_leaf=2)...")
    model = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced",
        min_samples_leaf=2, random_state=SEED
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS ON UNTOUCHED TEST SET")
    print("=" * 50)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=["benign", "malicious"]))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    if tranco_hold:
        Xh = pd.DataFrame([extract_features("https://" + d) for d in tranco_hold], columns=FEATURE_NAMES)
        ph = model.predict_proba(Xh)[:, 1]
        is_in = np.array([d.endswith(".in") for d in tranco_hold])
        fa = float((ph >= 0.5).mean())
        print(f"\nFalse-alarm rate on {len(ph):,} UNSEEN legitimate Tranco domains: {fa:.1%}")
        if is_in.any():
            print(f"  Of which .in domains ({int(is_in.sum())}): {float((ph[is_in] >= 0.5).mean()):.1%}")
        metrics["unseen_legit_false_alarm_rate"] = fa

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaving model to {MODEL_SAVE_PATH} (compress=3)...")
    joblib.dump(model, MODEL_SAVE_PATH, compress=3)
    print("Done! Model and metrics saved successfully.")


if __name__ == "__main__":
    main()
