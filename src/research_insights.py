"""Research summaries for dataset-level music popularity insights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_RESEARCH_FEATURES = [
    "tempo",
    "rms_mean",
    "rms_std",
    "chroma_mean",
    "chroma_std",
    "tonnetz_mean",
    "tonnetz_std",
    "zcr_std",
]

FEATURE_LABELS = {
    "tempo": "Tempo",
    "rms_mean": "RMS mean",
    "rms_std": "RMS std",
    "chroma_mean": "Chroma mean",
    "chroma_std": "Chroma std",
    "tonnetz_mean": "Tonnetz mean",
    "tonnetz_std": "Tonnetz std",
    "zcr_std": "ZCR std",
}

PROFILE_TIERS = ["top20", "chart_21_50", "chart_51_100", "low_stream"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data/processed/extended_features.csv"
DEFAULT_QC_REPORT_PATH = PROJECT_ROOT / "reports/dataset_qc.json"
DEFAULT_MODEL_REPORT_PATH = PROJECT_ROOT / "reports/model_observability.json"


def load_json_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report from disk."""
    with Path(path).open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


def load_features(path: str | Path) -> pd.DataFrame:
    """Load extracted feature rows from a CSV, JSON, JSONL, or parquet file."""
    feature_path = Path(path)
    suffix = feature_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(feature_path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(feature_path, lines=True)
    if suffix == ".json":
        return pd.read_json(feature_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(feature_path)
    raise ValueError(f"Unsupported feature file extension: {feature_path.suffix}")


def top_model_mfcc_features(model_report: dict[str, Any], limit: int = 3) -> list[str]:
    """Return top MFCC feature names from a model report, preserving report order."""
    features: list[str] = []
    for item in model_report.get("top_features", []):
        feature = item.get("feature") if isinstance(item, dict) else None
        if isinstance(feature, str) and feature.startswith("mfcc_") and feature not in features:
            features.append(feature)
        if len(features) >= limit:
            break
    return features


def select_research_features(
    df: pd.DataFrame,
    model_report: dict[str, Any],
    mfcc_limit: int = 3,
) -> list[str]:
    """Choose stable research features plus top available MFCC model features."""
    selected: list[str] = []

    for feature in BASE_RESEARCH_FEATURES:
        if feature in df.columns and feature not in selected:
            selected.append(feature)

    for feature in top_model_mfcc_features(model_report, limit=mfcc_limit):
        if feature in df.columns and feature not in selected:
            selected.append(feature)

    return selected


def build_dataset_snapshot(
    qc_report: dict[str, Any],
    model_report: dict[str, Any],
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build compact dataset and validation metrics for research reporting."""
    dataset = qc_report.get("dataset", {})
    label_counts = dataset.get("label_source_counts", {})
    extraction = qc_report.get("extraction", {})
    source_bias = qc_report.get("source_bias", {})
    quality = qc_report.get("audio_source_quality", {})
    validation = model_report.get("validation", {})
    qc_source_auc = source_bias.get("source_separability_auc_audio_features")

    feature_rows = dataset.get("rows_extracted")
    charted_rows = label_counts.get("charted")
    low_download_rows = label_counts.get("archive_low_download")

    if df is not None:
        if feature_rows is None:
            feature_rows = int(len(df))
        if charted_rows is None and "label_source" in df.columns:
            charted_rows = int((df["label_source"] == "charted").sum())
        if low_download_rows is None and "label_source" in df.columns:
            low_download_rows = int((df["label_source"] == "archive_low_download").sum())

    return {
        "feature_rows": feature_rows,
        "charted_rows": charted_rows,
        "low_download_rows": low_download_rows,
        "feature_extraction_success_rate": extraction.get("success_rate"),
        "source_auc": qc_source_auc
        if qc_source_auc is not None
        else validation.get("model_feature_source_separability_auc"),
        "quality_source_auc": quality.get("source_separability_auc_format_only"),
        "distance_score_auc": validation.get("distance_score_roc_auc"),
        "xgboost_oof_auc": validation.get("xgboost_oof_roc_auc"),
        "model_feature_source_auc": validation.get("model_feature_source_separability_auc"),
    }


def summarize_feature_profiles(
    df: pd.DataFrame,
    model_report: dict[str, Any],
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize charted reference distributions and popularity-tier medians."""
    selected = features or select_research_features(df, model_report)
    charted = _charted_rows(df)
    profiles: list[dict[str, Any]] = []
    charted_medians: dict[str, float | None] = {}
    charted_iqrs: dict[str, float | None] = {}

    for feature in selected:
        charted_values = pd.to_numeric(charted[feature], errors="coerce").dropna()
        q25 = _series_quantile(charted_values, 0.25)
        q75 = _series_quantile(charted_values, 0.75)
        median = _series_median(charted_values)

        row: dict[str, Any] = {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
            "charted_median": median,
            "charted_q25": q25,
            "charted_q75": q75,
        }

        for tier in PROFILE_TIERS:
            row[f"{tier}_median"] = _tier_median(df, tier, feature)

        profiles.append(row)
        charted_medians[feature] = median
        charted_iqrs[feature] = None if q25 is None or q75 is None else q75 - q25

    return {
        "features": selected,
        "feature_profiles": profiles,
        "charted_feature_median": charted_medians,
        "charted_feature_iqr": charted_iqrs,
    }


def summarize_key_distribution(df: pd.DataFrame, top_n: int = 8) -> list[dict[str, Any]]:
    """Return top musical keys with total and charted counts."""
    if "key" not in df.columns:
        return []

    counts = df["key"].dropna().value_counts().head(top_n)
    charted = _charted_rows(df)
    charted_counts = (
        charted["key"].dropna().value_counts() if "key" in charted.columns else pd.Series(dtype=int)
    )

    return [
        {
            "key": key,
            "all_count": int(all_count),
            "charted_count": int(charted_counts.get(key, 0)),
        }
        for key, all_count in counts.items()
    ]


def compare_track_to_reference(
    row: pd.DataFrame | pd.Series | dict[str, Any],
    metadata: dict[str, Any],
    features: list[str],
) -> list[dict[str, Any]]:
    """Classify a track's feature values against charted median +/- IQR/2."""
    values = _first_row_values(row)
    medians = metadata.get("charted_feature_median", {})
    iqrs = metadata.get("charted_feature_iqr", {})
    comparison: list[dict[str, Any]] = []

    for feature in features:
        if feature not in values or feature not in medians:
            continue

        value = _float_or_none(values[feature])
        median = _float_or_none(medians.get(feature))
        iqr = _float_or_none(iqrs.get(feature))
        if value is None or median is None:
            continue

        half_iqr = (iqr or 0.0) / 2.0
        lower = median - half_iqr
        upper = median + half_iqr
        if value < lower:
            status = "below"
        elif value > upper:
            status = "above"
        else:
            status = "inside"

        comparison.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "value": value,
                "reference_median": median,
                "reference_low": lower,
                "reference_high": upper,
                "reference_iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "status": status,
            }
        )

    return comparison


def build_research_summary(
    features_df: pd.DataFrame | None = None,
    qc_report: dict[str, Any] | None = None,
    model_report: dict[str, Any] | None = None,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    qc_report_path: str | Path = DEFAULT_QC_REPORT_PATH,
    model_report_path: str | Path = DEFAULT_MODEL_REPORT_PATH,
    key_top_n: int = 8,
) -> dict[str, Any]:
    """Build the full research summary consumed by markdown/reporting tasks."""
    if features_df is None:
        features_df = load_features(features_path)
    if qc_report is None:
        qc_report = load_json_report(qc_report_path)
    if model_report is None:
        model_report = load_json_report(model_report_path)

    profile_summary = summarize_feature_profiles(features_df, model_report)
    return {
        "snapshot": build_dataset_snapshot(qc_report, model_report, features_df),
        "features": profile_summary["features"],
        "feature_profiles": profile_summary["feature_profiles"],
        "charted_feature_median": profile_summary["charted_feature_median"],
        "charted_feature_iqr": profile_summary["charted_feature_iqr"],
        "key_distribution": summarize_key_distribution(features_df, top_n=key_top_n),
    }


def format_research_markdown(summary: dict[str, Any]) -> str:
    """Format a bias-aware markdown report for research findings."""
    snapshot = summary.get("snapshot", {})
    profiles = summary.get("feature_profiles", [])
    keys = summary.get("key_distribution", [])

    lines = [
        "# Features of Popular-Reference Tracks",
        "",
        "This report describes patterns in this dataset, not universal rules for music popularity.",
        "Because charted tracks and Internet Archive low-download tracks come from different collection paths, source/domain effects must be treated as a core caveat.",
        "",
        "## Dataset Snapshot",
        "",
        f"- Feature rows: {_format_value(snapshot.get('feature_rows'))}",
        f"- Charted rows: {_format_value(snapshot.get('charted_rows'))}",
        f"- Low-download rows: {_format_value(snapshot.get('low_download_rows'))}",
        f"- Feature extraction success rate: {_format_metric(snapshot.get('feature_extraction_success_rate'))}",
        f"- Audio feature source/domain AUC: {_format_metric(snapshot.get('source_auc'))}",
        f"- Format-only source/domain AUC: {_format_metric(snapshot.get('quality_source_auc'))}",
        f"- Distance score ROC AUC: {_format_metric(snapshot.get('distance_score_auc'))}",
        f"- XGBoost OOF ROC AUC: {_format_metric(snapshot.get('xgboost_oof_auc'))}",
        "",
        "## Feature Profiles",
        "",
        "| Feature | Charted median | Charted q25 | Charted q75 | Top 20 median | 21-50 median | 51-100 median | Low-stream median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in profiles:
        lines.append(
            "| {label} | {charted_median} | {charted_q25} | {charted_q75} | {top20} | {chart_21_50} | {chart_51_100} | {low_stream} |".format(
                label=row.get("label") or row.get("feature"),
                charted_median=_format_value(row.get("charted_median")),
                charted_q25=_format_value(row.get("charted_q25")),
                charted_q75=_format_value(row.get("charted_q75")),
                top20=_format_value(row.get("top20_median")),
                chart_21_50=_format_value(row.get("chart_21_50_median")),
                chart_51_100=_format_value(row.get("chart_51_100_median")),
                low_stream=_format_value(row.get("low_stream_median")),
            )
        )

    lines.extend(["", "## Key Distribution", ""])
    if keys:
        lines.extend(["| Key | All tracks | Charted tracks |", "| --- | ---: | ---: |"])
        for row in keys:
            lines.append(
                f"| {row.get('key')} | {_format_value(row.get('all_count'))} | {_format_value(row.get('charted_count'))} |"
            )
    else:
        lines.append("No key metadata is available in this dataset.")

    lines.extend(
        [
            "",
            "## Interpretation Caveats",
            "",
            "High validation AUC can reflect real audio-pattern signal, source/domain separation, or both. Treat these summaries as descriptive reference statistics in this dataset until the collection design is balanced.",
            "",
            "## Data Roadmap",
            "",
            "- Expand Internet Archive sampling to cover more genres, years, formats, and download bands.",
            "- Pair archive tracks with additional chart and non-chart references acquired through yt-dlp only where rights, terms, and project policy allow it.",
            "- Track source/domain diagnostics beside model metrics for every dataset refresh.",
        ]
    )

    return "\n".join(lines) + "\n"


def _charted_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "label_source" in df.columns:
        return df[df["label_source"] == "charted"]
    if "popular" in df.columns:
        return df[df["popular"] == 1]
    return df.iloc[0:0]


def _tier_median(df: pd.DataFrame, tier: str, feature: str) -> float | None:
    if "popularity_tier" not in df.columns:
        return None
    values = pd.to_numeric(
        df.loc[df["popularity_tier"] == tier, feature],
        errors="coerce",
    ).dropna()
    return _series_median(values)


def _series_median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.median())


def _series_quantile(series: pd.Series, quantile: float) -> float | None:
    if series.empty:
        return None
    return float(series.quantile(quantile))


def _first_row_values(row: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _format_metric(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4f}"


def _format_value(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")
