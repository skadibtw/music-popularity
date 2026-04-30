import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from music_success_predictor import DEFAULT_PREVIEW_SECONDS, add_key_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict


REPORT_JSON_PATH = "reports/model_observability.json"
REPORT_MD_PATH = "reports/model_observability.md"
QC_JSON_PATH = "reports/dataset_qc.json"
SOURCE_FEATURE_AUC_THRESHOLD = 0.65


def make_xgboost(scale_pos_weight):
    return xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )


def load_qc_report():
    if not os.path.exists(QC_JSON_PATH):
        return {}
    with open(QC_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.quantile(values, quantile))
    return float(np.interp(quantile * cumulative[-1], cumulative, values))


def weighted_reference_stats(X, weights):
    medians = {}
    iqrs = {}
    for col in X.columns:
        values = X[col].astype(float).values
        q25 = weighted_quantile(values, weights, 0.25)
        q50 = weighted_quantile(values, weights, 0.50)
        q75 = weighted_quantile(values, weights, 0.75)
        medians[col] = q50
        iqrs[col] = max(q75 - q25, 1.0)
    return pd.Series(medians), pd.Series(iqrs)


def source_feature_auc_report(df, feature_cols):
    if "label_source" not in df.columns or df["label_source"].nunique() != 2:
        return []

    y = df["label_source"].astype("category").cat.codes.values
    rows = []
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        clean = pd.DataFrame({"feature": values, "source": y}).dropna()
        if clean["feature"].nunique() <= 1:
            continue
        auc = float(roc_auc_score(clean["source"], clean["feature"]))
        rows.append(
            {
                "feature": col,
                "source_auc": auc,
                "source_abs_auc": max(auc, 1 - auc),
            }
        )
    return sorted(rows, key=lambda item: item["source_abs_auc"], reverse=True)


def source_separability_auc(df, feature_cols):
    if "label_source" not in df.columns or df["label_source"].nunique() != 2:
        return None

    clean = df.dropna(subset=feature_cols + ["label_source"])
    y = clean["label_source"].astype("category").cat.codes.values
    if len(np.unique(y)) != 2:
        return None

    n_splits = min(5, int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return None

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_predict(model, clean[feature_cols], y, cv=splitter, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores))


def write_observability_reports(report):
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    top_features = "\n".join(
        f"- `{item['feature']}`: {item['importance']:.4f}" for item in report["top_features"]
    )
    label_sources = "\n".join(
        f"- `{source}`: {count}" for source, count in report["dataset"]["label_source_counts"].items()
    )
    tiers = "\n".join(
        f"- `{tier}`: {count}" for tier, count in report["dataset"]["popularity_tier_counts"].items()
    )
    excluded_source_features = "\n".join(
        f"- `{item['feature']}`: source abs AUC {item['source_abs_auc']:.4f}"
        for item in report["dataset"].get("excluded_source_confounded_features", [])
    ) or "- None"
    top_source_features = "\n".join(
        f"- `{item['feature']}`: source abs AUC {item['source_abs_auc']:.4f}"
        for item in report.get("source_feature_diagnostics", [])[:10]
    ) or "- Not available"
    qc_checks = json.dumps(report.get("qc", {}).get("checks", {}), ensure_ascii=False, indent=2)
    source_bias = report.get("qc", {}).get("source_bias", {})
    extraction = report["dataset"].get("feature_extraction", {})
    extraction_text = f"{extraction.get('mode')} ({extraction.get('preview_seconds')}s)"
    model_source_auc = report["validation"].get("model_feature_source_separability_auc")
    model_source_auc_text = "N/A" if model_source_auc is None else f"{model_source_auc:.4f}"

    markdown = f"""# Model Observability Report

Generated by `src/train_extended_model.py`.

## Dataset

- Rows used: {report['dataset']['rows_used']}
- Feature count: {report['dataset']['feature_count']}
- Feature extraction: `{extraction_text}`
- Source-confounded feature threshold: `{report['dataset'].get('source_feature_auc_threshold')}`
- Positive label: `{report['labels']['positive']}`
- Negative label: `{report['labels']['negative']}`
- Popular base rate: {report['dataset']['base_rate']:.2%}

## Label Sources

{label_sources}

## Popularity Tiers

{tiers}

## Validation

- Split: {report['validation']['split']}
- Distance-score ROC-AUC: {report['validation']['distance_score_roc_auc']:.4f}
- XGBoost holdout accuracy: {report['validation']['xgboost_holdout_accuracy']:.4f}
- XGBoost holdout ROC-AUC: {report['validation']['xgboost_holdout_roc_auc']:.4f}
- XGBoost out-of-fold ROC-AUC: {report['validation']['xgboost_oof_roc_auc']:.4f}
- Model-feature source separability ROC-AUC: {model_source_auc_text}
- OOD distance p95: {report['scoring']['ood_distance_p95']:.4f}

## QC And Source Bias

- Median duration seconds: {report.get('qc', {}).get('audio_qc', {}).get('median_duration_seconds')}
- Source separability ROC-AUC, audio features only: {source_bias.get('source_separability_auc_audio_features')}
- Source separability ROC-AUC, all numeric fields: {source_bias.get('source_separability_auc_all_numeric')}
- Source-holdout possible: {source_bias.get('source_holdout_possible')}

```json
{qc_checks}
```

## Source-Confounded Features

Excluded from model and charted-reference scoring:

{excluded_source_features}

Top source-separating audio features before exclusion:

{top_source_features}

## Score Semantics

{report['scoring']['description']}

The headline score is not a calibrated probability or a guarantee of commercial success. It is a reference-set percentile derived from distance to the weighted charted audio-feature profile.

## Top XGBoost Diagnostic Features

{top_features}
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)


def train_xgboost():
    print("Loading extended features...")
    df = pd.read_csv("data/processed/extended_features.csv")
    if "popular" not in df.columns:
        raise RuntimeError("extended_features.csv must contain the new `popular` target column.")

    df = add_key_features(df)

    exclude_cols = [
        "file_path",
        "artist",
        "title",
        "label_source",
        "stream_count",
        "key",
        "popular",
        "success",
        "peak_rank",
        "weeks-on-board",
        "popularity_tier",
        "popularity_weight",
        "duration_seconds",
        "analyzed_duration_seconds",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df = df.dropna(subset=feature_cols + ["popular"])

    source_feature_report = source_feature_auc_report(df, feature_cols)
    excluded_source_features = [
        item for item in source_feature_report if item["source_abs_auc"] >= SOURCE_FEATURE_AUC_THRESHOLD
    ]
    excluded_source_feature_names = {item["feature"] for item in excluded_source_features}
    if excluded_source_feature_names:
        print(
            "Excluding source-confounded features: "
            f"{sorted(excluded_source_feature_names)}"
        )
        feature_cols = [c for c in feature_cols if c not in excluded_source_feature_names]
    if not feature_cols:
        raise RuntimeError("No usable model features remain after source-confounded feature exclusion.")

    X = df[feature_cols]
    y = df["popular"].astype(int).values
    groups = df["artist"].fillna(df["file_path"]).values
    model_feature_source_auc = source_separability_auc(df, feature_cols)

    min_class_count = pd.Series(y).value_counts().min()
    n_splits = min(5, int(min_class_count))
    if n_splits < 2:
        raise RuntimeError("Need at least two rows per class for grouped validation split.")

    chart_weights = df.loc[df["popular"].astype(int) == 1, "popularity_weight"].fillna(1.0).clip(lower=0.0)
    charted_X = X[df["popular"].astype(int) == 1]
    charted_medians, charted_iqrs = weighted_reference_stats(charted_X, chart_weights.values)
    charted_reference_distances = ((X - charted_medians).abs() / charted_iqrs).mean(axis=1)
    distance_auc = roc_auc_score(y, -charted_reference_distances)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_iter = list(splitter.split(X, y, groups))
    train_idx, test_idx = split_iter[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\nSplit: StratifiedGroupKFold by artist.")
    print(f"Train target distribution: {pd.Series(y_train).value_counts(normalize=True).round(3).to_dict()}")
    print(f"Test target distribution: {pd.Series(y_test).value_counts(normalize=True).round(3).to_dict()}")
    class_counts = pd.Series(y_train).value_counts()
    scale_pos_weight = class_counts.get(0, 1) / max(class_counts.get(1, 1), 1)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    print("\nTraining XGBoost diagnostic model...")
    model = make_xgboost(scale_pos_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    holdout_accuracy = accuracy_score(y_test, y_pred)
    try:
        holdout_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        holdout_auc = 0.0

    print("\nRESULTS:")
    print(f"Distance-score ROC-AUC: {distance_auc:.4f}")
    print(f"XGBoost accuracy: {holdout_accuracy:.4f}")
    print(f"XGBoost ROC-AUC: {holdout_auc:.4f}")
    if model_feature_source_auc is not None:
        print(f"Model-feature source separability ROC-AUC: {model_feature_source_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    oof_scores = np.zeros(len(df), dtype=float)
    for fold_train_idx, fold_test_idx in split_iter:
        fold_y = y[fold_train_idx]
        fold_counts = pd.Series(fold_y).value_counts()
        fold_weight = fold_counts.get(0, 1) / max(fold_counts.get(1, 1), 1)
        fold_model = make_xgboost(fold_weight)
        fold_model.fit(X.iloc[fold_train_idx], fold_y)
        oof_scores[fold_test_idx] = fold_model.predict_proba(X.iloc[fold_test_idx])[:, 1]

    try:
        oof_auc = roc_auc_score(y, oof_scores)
        print(f"XGBoost out-of-fold ROC-AUC: {oof_auc:.4f}")
    except ValueError:
        oof_auc = 0.0

    final_counts = pd.Series(y).value_counts()
    final_weight = final_counts.get(0, 1) / max(final_counts.get(1, 1), 1)
    model = make_xgboost(final_weight)
    model.fit(X, y)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_imp)
    plt.title("Top-15 audio features (XGBoost diagnostic)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance_xgb.png")
    plt.close()

    medians = X.median()
    iqrs = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0)
    distances = ((X - medians).abs() / iqrs).mean(axis=1)

    metadata = {
        "score_description": "Hit potential score: percentile closeness to the weighted charted audio-feature reference profile.",
        "training_population": "Matched chart MP3s plus Internet Archive netlabels low-download MP3s.",
        "feature_extraction": {
            "mode": "preview",
            "preview_seconds": DEFAULT_PREVIEW_SECONDS,
        },
        "source_feature_auc_threshold": SOURCE_FEATURE_AUC_THRESHOLD,
        "excluded_source_confounded_features": excluded_source_features,
        "source_feature_diagnostics": source_feature_report[:20],
        "model_feature_source_separability_auc": model_feature_source_auc,
        "positive_label": "charted track",
        "negative_label": "archive_low_download track",
        "label_source_counts": df["label_source"].value_counts().to_dict(),
        "popularity_tier_counts": df["popularity_tier"].value_counts().to_dict(),
        "base_rate": float(y.mean()),
        "distance_score_auc": float(distance_auc),
        "oof_auc": float(oof_auc),
        "oof_scores": oof_scores.tolist(),
        "oof_labels": y.tolist(),
        "feature_median": medians.to_dict(),
        "feature_iqr": iqrs.to_dict(),
        "ood_distance_p95": float(np.percentile(distances, 95)),
        "charted_feature_median": charted_medians.to_dict(),
        "charted_feature_iqr": charted_iqrs.to_dict(),
        "charted_reference_distances": charted_reference_distances.tolist(),
        "n_tracks": int(len(df)),
    }
    qc_report = load_qc_report()

    observability_report = {
        "dataset": {
            "rows_used": int(len(df)),
            "feature_count": int(len(feature_cols)),
            "feature_extraction": metadata["feature_extraction"],
            "source_feature_auc_threshold": SOURCE_FEATURE_AUC_THRESHOLD,
            "excluded_source_confounded_features": excluded_source_features,
            "target_counts": pd.Series(y).value_counts().sort_index().astype(int).to_dict(),
            "label_source_counts": df["label_source"].value_counts().to_dict(),
            "popularity_tier_counts": df["popularity_tier"].value_counts().to_dict(),
            "base_rate": float(y.mean()),
        },
        "labels": {
            "positive": metadata["positive_label"],
            "negative": metadata["negative_label"],
        },
        "validation": {
            "split": f"StratifiedGroupKFold(n_splits={n_splits}, group=artist)",
            "distance_score_roc_auc": float(distance_auc),
            "xgboost_holdout_accuracy": float(holdout_accuracy),
            "xgboost_holdout_roc_auc": float(holdout_auc),
            "xgboost_oof_roc_auc": float(oof_auc),
            "model_feature_source_separability_auc": model_feature_source_auc,
            "train_target_distribution": pd.Series(y_train).value_counts(normalize=True).round(4).to_dict(),
            "test_target_distribution": pd.Series(y_test).value_counts(normalize=True).round(4).to_dict(),
        },
        "scoring": {
            "description": metadata["score_description"],
            "ood_distance_p95": metadata["ood_distance_p95"],
        },
        "top_features": feat_imp.to_dict(orient="records"),
        "source_feature_diagnostics": source_feature_report[:20],
        "qc": {
            "checks": qc_report.get("checks", {}),
            "audio_qc": qc_report.get("audio_qc", {}),
            "source_bias": qc_report.get("source_bias", {}),
        },
    }

    joblib.dump(model, "models/xgboost_music_model.pkl")
    joblib.dump(feature_cols, "models/xgboost_features.pkl")
    joblib.dump(metadata, "models/xgboost_score_metadata.pkl")
    write_observability_reports(observability_report)
    print("\nModel, feature list, and score metadata saved")
    print(f"Observability report saved to {REPORT_MD_PATH}")
    print("Feature importance plot saved to plots/feature_importance_xgb.png")


if __name__ == "__main__":
    train_xgboost()
