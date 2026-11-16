"""Phases 4-5 – Model Development and Evaluation.

Trains four classifiers, tunes hyper-parameters with GridSearchCV (3-fold, F1),
evaluates on a held-out 20 % test set, runs a feature-ablation study, and
saves all outputs (plots, metrics CSV, best model) to `outputs/`.

Models compared
---------------
- Logistic Regression  — fast, interpretable linear baseline
- Random Forest        — ensemble of decision trees, handles non-linearity
- Multinomial Naive Bayes — classic text-classification algorithm, works well
                           with TF-IDF counts
- MLP Neural Network   — simple feed-forward net with early stopping
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe inside any runner
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .preprocess import clean_text, build_combined
from .features import make_preprocessor, metadata_frame, META_COLS

warnings.filterwarnings("ignore")
SEED = 42

# ------------------------------------------------------------------ model zoo
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"clf__C": [0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20]},
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {"clf__alpha": [0.1, 0.5, 1.0]},
    ),
    "Neural Network (MLP)": (
        MLPClassifier(max_iter=300, early_stopping=True, random_state=SEED),
        {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
    ),
}


# ------------------------------------------------------------------ helpers
def load_and_prepare(path, text_col="body", label_col="label",
                     subject_col="subject", sender_col="sender_domain"):
    """Load raw CSV, clean text with NLTK pipeline, add metadata columns."""
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "body", label_col: "label"})
    if subject_col not in df.columns:
        df["subject"] = ""
    else:
        df = df.rename(columns={subject_col: "subject"}) if subject_col != "subject" else df
    if sender_col not in df.columns:
        df["sender_domain"] = ""
    else:
        df = df.rename(columns={sender_col: "sender_domain"}) if sender_col != "sender_domain" else df

    df = (df
          .dropna(subset=["body", "label"])
          .drop_duplicates(subset=["subject", "body"])
          .reset_index(drop=True))
    df["label"] = df["label"].astype(int)
    df["clean_text"] = build_combined(df).map(clean_text)   # NLTK pipeline
    df = pd.concat([df, metadata_frame(df)], axis=1)
    return df


def evaluate(y_true, y_pred, y_prob) -> dict:
    """Return a dict of all five classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# ------------------------------------------------------------------ main run
def run(data_path="data/raw_emails.csv", out_dir="outputs",
        cleaned_path="data/cleaned_emails.csv", **cols):
    """Full end-to-end pipeline. Returns (results_df, ablation_dict, best_model_name)."""

    # ---- Phase 2: load & clean ----
    df = load_and_prepare(data_path, **cols)
    df.to_csv(cleaned_path, index=False)
    print(f"✔ Cleaned dataset → {cleaned_path}  shape={df.shape}")

    # ---- Phase 3: feature matrix ----
    X = df[["clean_text"] + META_COLS]
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}  "
          f"Phishing rate (test): {y_te.mean():.1%}")

    # ---- Phase 4: train + tune ----
    results, fitted, preds, probs = {}, {}, {}, {}
    for name, (est, grid) in MODELS.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", est)])
        gs   = GridSearchCV(pipe, grid, scoring="f1", cv=3, n_jobs=-1).fit(X_tr, y_tr)
        best = gs.best_estimator_
        y_pred = best.predict(X_te)
        y_prob = best.predict_proba(X_te)[:, 1]
        results[name] = {
            **evaluate(y_te, y_pred, y_prob),
            "cv_f1": gs.best_score_,
            "best_params": str(gs.best_params_),
        }
        fitted[name], preds[name], probs[name] = best, y_pred, y_prob
        print(f"  {name:24s} acc={results[name]['accuracy']:.4f} "
              f"f1={results[name]['f1']:.4f}  {gs.best_params_}")

    res = pd.DataFrame(results).T
    res.to_csv(f"{out_dir}/model_comparison.csv")

    # ---- Feature ablation (Logistic Regression) ----
    abl = {}
    for label, (t, m) in {
        "Text only (TF-IDF)": (True,  False),
        "Metadata only":      (False, True),
        "Text + Metadata":    (True,  True),
    }.items():
        p = Pipeline([
            ("prep", make_preprocessor(t, m)),
            ("clf",  LogisticRegression(max_iter=1000, C=10)),
        ]).fit(X_tr, y_tr)
        abl[label] = evaluate(y_te, p.predict(X_te), p.predict_proba(X_te)[:, 1])
    pd.DataFrame(abl).T.to_csv(f"{out_dir}/ablation_features.csv")

    # ---- Phase 5: plots ----
    # Confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, yp) in zip(axes, preds.items()):
        sns.heatmap(confusion_matrix(y_te, yp), annot=True, fmt="d",
                    cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Legit", "Phish"],
                    yticklabels=["Legit", "Phish"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=150)
    plt.close()

    # Model comparison bar chart
    res[["accuracy", "precision", "recall", "f1"]].astype(float).plot(
        kind="bar", figsize=(9, 4.5), ylim=(0.5, 1.02),
        color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
    )
    plt.title("Model Comparison — Held-out Test Set")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(7, 5))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    for (name, yp), color in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_te, yp)
        auc = roc_auc_score(y_te, yp)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curves.png", dpi=150)
    plt.close()

    # Feature importance — Random Forest (top-20)
    rf    = fitted["Random Forest"]
    names = [n.split("__", 1)[1] for n in rf.named_steps["prep"].get_feature_names_out()]
    imp   = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(20)[::-1]
    imp.plot(kind="barh", figsize=(7, 6), color="#2a6fdb")
    plt.title("Random Forest — Top 20 Features"); plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_rf.png", dpi=150)
    plt.close()

    # Feature importance — Logistic Regression (top-10 each direction)
    lr    = fitted["Logistic Regression"]
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=names)
    top   = pd.concat([coefs.nsmallest(10), coefs.nlargest(10)]).sort_values()
    top.plot(kind="barh", figsize=(7, 6),
             color=["#2e9d5b" if v < 0 else "#d9433b" for v in top])
    plt.title("Logistic Regression — Top Features\n(red=phishing, green=legitimate)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_lr.png", dpi=150)
    plt.close()

    # ---- Save best model ----
    best_name = res["f1"].astype(float).idxmax()
    joblib.dump(fitted[best_name], f"{out_dir}/best_model.joblib")
    report = classification_report(
        y_te, preds[best_name], target_names=["legit", "phish"], output_dict=True
    )
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump({
            "best_model":  best_name,
            "n_samples":   int(len(df)),
            "n_test":      int(len(y_te)),
            "report":      report,
        }, f, indent=2)

    print(f"\n✔ Best model: {best_name}")
    print(classification_report(y_te, preds[best_name], target_names=["legit", "phish"]))
    return res, abl, best_name
