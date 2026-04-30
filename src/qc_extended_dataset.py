import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


REPORT_JSON_PATH = "reports/dataset_qc.json"
REPORT_MD_PATH = "reports/dataset_qc.md"


def rate(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator / denominator)


def load_optional_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def source_separability_auc(df, extra_exclude_cols=None):
    if "label_source" not in df.columns or df["label_source"].nunique() != 2:
        return None

    extra_exclude_cols = extra_exclude_cols or set()
    exclude_cols = {
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
    } | set(extra_exclude_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) < 2:
        return None

    clean = df.dropna(subset=feature_cols + ["label_source"])
    y = clean["label_source"].astype("category").cat.codes.values
    if len(np.unique(y)) != 2 or min(pd.Series(y).value_counts()) < 5:
        return None

    n_splits = min(5, int(min(pd.Series(y).value_counts())))
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_predict(model, clean[feature_cols], y, cv=splitter, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores))


def grouped_numeric_summary(df, group_col, numeric_cols):
    if group_col not in df.columns:
        return {}
    summary = {}
    for source, group in df.groupby(group_col):
        summary[str(source)] = {}
        for col in numeric_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if not values.empty:
                    summary[str(source)][col] = {
                        "median": float(values.median()),
                        "p10": float(values.quantile(0.10)),
                        "p90": float(values.quantile(0.90)),
                    }
    return summary


def build_qc_report(features_path, sample_manifest_path, failures_path, download_failures_path, output_json, output_md):
    df = pd.read_csv(features_path)
    sample_manifest = load_optional_csv(sample_manifest_path)
    failures = load_optional_csv(failures_path)
    download_failures = load_optional_csv(download_failures_path)

    expected_low_stream = len(sample_manifest) if not sample_manifest.empty else 0
    extracted_low_stream = int((df.get("label_source") == "archive_low_download").sum())
    extraction_failures = len(failures)
    extraction_attempts = len(df) + extraction_failures

    archive_counts = sample_manifest.get("archive_item", pd.Series(dtype=str)).value_counts()
    artist_counts = sample_manifest.get("artist", pd.Series(dtype=str)).fillna("Unknown").value_counts()

    duration_values = pd.to_numeric(df.get("duration_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    analyzed_duration_values = pd.to_numeric(
        df.get("analyzed_duration_seconds", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    duration_median = None if duration_values.empty else float(duration_values.median())
    analyzed_duration_median = None if analyzed_duration_values.empty else float(analyzed_duration_values.median())
    duration_under_30 = int((duration_values < 30).sum()) if not duration_values.empty else 0

    source_auc_all_numeric = source_separability_auc(df)
    source_auc_audio_features = source_separability_auc(df, {"duration_seconds", "analyzed_duration_seconds"})
    source_auc = source_auc_audio_features
    download_success_rate = rate(expected_low_stream - len(download_failures), expected_low_stream)
    source_target_table = (
        pd.crosstab(df["label_source"], df["popular"]).to_dict()
        if {"label_source", "popular"} <= set(df.columns)
        else {}
    )
    source_holdout_possible = bool(
        {"label_source", "popular"} <= set(df.columns)
        and df.groupby("label_source")["popular"].nunique().gt(1).any()
    )
    checks = {
        "download_success_rate_ge_90pct": bool(download_success_rate >= 0.90),
        "feature_extraction_success_rate_ge_85pct": bool(rate(len(df), extraction_attempts) >= 0.85),
        "duration_metadata_present": bool(duration_median is not None),
        "median_analyzed_duration_25_to_31s": bool(
            analyzed_duration_median is not None and 25 <= analyzed_duration_median <= 31
        ),
        "max_archive_item_le_10": bool((int(archive_counts.max()) if not archive_counts.empty else 0) <= 10),
        "max_artist_le_20_except_unknown": bool(artist_counts.drop(labels=["Unknown"], errors="ignore").max() <= 20) if not artist_counts.empty else True,
        "unknown_artist_le_50": bool(int((sample_manifest.get("artist", pd.Series(dtype=str)).fillna("Unknown") == "Unknown").sum()) <= 50) if not sample_manifest.empty else True,
        "source_separability_auc_lt_80pct": bool(source_auc_audio_features is not None and source_auc_audio_features < 0.80),
    }
    report = {
        "dataset": {
            "rows_extracted": int(len(df)),
            "target_counts": df.get("popular", pd.Series(dtype=int)).value_counts().sort_index().astype(int).to_dict(),
            "label_source_counts": df.get("label_source", pd.Series(dtype=str)).value_counts().to_dict(),
            "popularity_tier_counts": df.get("popularity_tier", pd.Series(dtype=str)).value_counts().to_dict(),
        },
        "sample_manifest": {
            "rows": int(expected_low_stream),
            "unique_source_urls": int(sample_manifest.get("source_url", pd.Series(dtype=str)).nunique()) if not sample_manifest.empty else 0,
            "unique_archive_items": int(sample_manifest.get("archive_item", pd.Series(dtype=str)).nunique()) if not sample_manifest.empty else 0,
            "unique_artists": int(sample_manifest.get("artist", pd.Series(dtype=str)).fillna("Unknown").nunique()) if not sample_manifest.empty else 0,
            "max_per_archive_item": int(archive_counts.max()) if not archive_counts.empty else 0,
            "max_per_artist": int(artist_counts.max()) if not artist_counts.empty else 0,
            "unknown_artist_rows": int((sample_manifest.get("artist", pd.Series(dtype=str)).fillna("Unknown") == "Unknown").sum()) if not sample_manifest.empty else 0,
        },
        "extraction": {
            "attempts": int(extraction_attempts),
            "failures": int(extraction_failures),
            "success_rate": rate(len(df), extraction_attempts),
            "download_failures": int(len(download_failures)),
            "download_success_rate": download_success_rate,
            "expected_low_stream_rows": int(expected_low_stream),
            "extracted_low_stream_rows": int(extracted_low_stream),
            "low_stream_extraction_rate": rate(extracted_low_stream, expected_low_stream),
        },
        "audio_qc": {
            "median_duration_seconds": duration_median,
            "median_analyzed_duration_seconds": analyzed_duration_median,
            "duration_under_30_seconds_rows": duration_under_30,
        },
        "source_bias": {
            "source_separability_auc": source_auc,
            "source_separability_auc_audio_features": source_auc_audio_features,
            "source_separability_auc_all_numeric": source_auc_all_numeric,
            "audio_feature_auc_excludes": ["duration_seconds", "analyzed_duration_seconds"],
            "source_target_table": source_target_table,
            "source_holdout_possible": source_holdout_possible,
            "interpretation": "High AUC means source/domain artifacts may dominate popularity separation." if source_auc is not None else "Not enough data for source separability check.",
        },
        "checks": checks,
        "by_source_summary": grouped_numeric_summary(
            df,
            "label_source",
            ["duration_seconds", "tempo", "rms_mean", "spectral_centroid_mean", "spectral_bandwidth_mean"],
        ),
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    markdown = f"""# Dataset QC Report

Generated by `src/qc_extended_dataset.py`.

## Dataset

- Rows extracted: {report['dataset']['rows_extracted']}
- Target counts: `{report['dataset']['target_counts']}`
- Label source counts: `{report['dataset']['label_source_counts']}`
- Popularity tier counts: `{report['dataset']['popularity_tier_counts']}`

## Sample Manifest

- Rows: {report['sample_manifest']['rows']}
- Unique source URLs: {report['sample_manifest']['unique_source_urls']}
- Unique archive items: {report['sample_manifest']['unique_archive_items']}
- Unique artists: {report['sample_manifest']['unique_artists']}
- Max per archive item: {report['sample_manifest']['max_per_archive_item']}
- Max per artist: {report['sample_manifest']['max_per_artist']}
- Unknown artist rows: {report['sample_manifest']['unknown_artist_rows']}

## Extraction

- Attempts: {report['extraction']['attempts']}
- Failures: {report['extraction']['failures']}
- Success rate: {report['extraction']['success_rate']:.2%}
- Download failures: {report['extraction']['download_failures']}
- Download success rate: {report['extraction']['download_success_rate']:.2%}
- Expected low-stream rows: {report['extraction']['expected_low_stream_rows']}
- Extracted low-stream rows: {report['extraction']['extracted_low_stream_rows']}
- Low-stream extraction rate: {report['extraction']['low_stream_extraction_rate']:.2%}

## Audio QC

- Median duration seconds: {report['audio_qc']['median_duration_seconds']}
- Median analyzed duration seconds: {report['audio_qc']['median_analyzed_duration_seconds']}
- Rows under 30 seconds: {report['audio_qc']['duration_under_30_seconds_rows']}

## Source Bias

- Source separability ROC-AUC, audio features only: {report['source_bias']['source_separability_auc_audio_features']}
- Source separability ROC-AUC, all numeric fields: {report['source_bias']['source_separability_auc_all_numeric']}
- Audio-feature AUC excludes: `{report['source_bias']['audio_feature_auc_excludes']}`
- Source-holdout possible: {report['source_bias']['source_holdout_possible']}
- Interpretation: {report['source_bias']['interpretation']}

## QC Checks

```json
{json.dumps(report['checks'], ensure_ascii=False, indent=2)}
```
"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"QC report saved to {output_md}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QC report for the extended feature dataset.")
    parser.add_argument("--features", default="data/processed/extended_features.csv", help="Feature dataset CSV.")
    parser.add_argument("--sample-manifest", default="data/raw/low_stream_tracks_sample_2000.csv", help="Staged low-stream sample manifest.")
    parser.add_argument("--failures", default="reports/feature_extraction_failures.csv", help="Feature extraction failures CSV.")
    parser.add_argument("--download-failures", default="reports/low_stream_download_failures.csv", help="Download failures CSV.")
    parser.add_argument("--output-json", default=REPORT_JSON_PATH, help="Output JSON report.")
    parser.add_argument("--output-md", default=REPORT_MD_PATH, help="Output markdown report.")
    args = parser.parse_args()

    build_qc_report(
        features_path=args.features,
        sample_manifest_path=args.sample_manifest,
        failures_path=args.failures,
        download_failures_path=args.download_failures,
        output_json=args.output_json,
        output_md=args.output_md,
    )
