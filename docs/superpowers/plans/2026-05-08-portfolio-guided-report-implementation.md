# Portfolio Guided Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Streamlit demo into a guided portfolio report and add a small, bias-aware research summary about popular-reference song features using the current dataset.

**Architecture:** Keep scoring behavior in `src/music_success_predictor.py`, move research-summary calculations into a new pure helper module, and keep `app.py` focused on Streamlit rendering. Generate `reports/popular_feature_research.md` from the same helper module so the UI and written report cannot drift.

**Tech Stack:** Python, Streamlit, pandas, Plotly, SHAP, XGBoost/joblib artifacts, stdlib `unittest`.

---

## Scope Notes

The repository is currently dirty, including `app.py`, `README.md`, report files, model artifacts, and data files. Before implementing each task, run `git status --short` and `git diff -- <file>` for files you will edit. Do not revert unrelated changes. If a file already contains user edits, work with the current file and commit only when the staged diff is known to contain this implementation's changes.

This plan does not download new tracks and does not run `yt-dlp`.

## File Structure

- Create `src/research_insights.py`: pure data/report helpers for dataset snapshot, feature summaries, key distributions, uploaded-track comparison bands, and markdown formatting.
- Create `tests/test_research_insights.py`: stdlib unit tests for the pure helper module.
- Create `src/write_popular_feature_research.py`: CLI script that writes `reports/popular_feature_research.md` from current CSV/JSON artifacts.
- Create `reports/popular_feature_research.md`: generated mini research report.
- Modify `app.py`: render pre-upload dataset/method context, research insights, and guided post-upload report using the helper module.
- Modify `README.md`: add a short research-summary pointer and mention the new report.

## Task 1: Add Research Insight Helpers With Unit Tests

**Files:**
- Create: `tests/test_research_insights.py`
- Create: `src/research_insights.py`

- [ ] **Step 1: Create the failing tests**

Create `tests/test_research_insights.py` with this content:

```python
import unittest

import pandas as pd

from src.research_insights import (
    build_dataset_snapshot,
    compare_track_to_reference,
    format_research_markdown,
    select_research_features,
    summarize_feature_profiles,
    summarize_key_distribution,
)


class ResearchInsightsTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "popular": 1,
                    "label_source": "charted",
                    "popularity_tier": "top20",
                    "tempo": 120,
                    "rms_mean": 0.20,
                    "rms_std": 0.03,
                    "chroma_mean": 0.45,
                    "chroma_std": 0.10,
                    "tonnetz_mean": 0.02,
                    "tonnetz_std": 0.11,
                    "zcr_std": 0.04,
                    "mfcc_11_std": 4.0,
                    "key": "C major",
                },
                {
                    "popular": 1,
                    "label_source": "charted",
                    "popularity_tier": "chart_21_50",
                    "tempo": 132,
                    "rms_mean": 0.25,
                    "rms_std": 0.04,
                    "chroma_mean": 0.50,
                    "chroma_std": 0.12,
                    "tonnetz_mean": 0.04,
                    "tonnetz_std": 0.13,
                    "zcr_std": 0.05,
                    "mfcc_11_std": 5.0,
                    "key": "A minor",
                },
                {
                    "popular": 0,
                    "label_source": "archive_low_download",
                    "popularity_tier": "low_stream",
                    "tempo": 90,
                    "rms_mean": 0.10,
                    "rms_std": 0.02,
                    "chroma_mean": 0.30,
                    "chroma_std": 0.08,
                    "tonnetz_mean": -0.01,
                    "tonnetz_std": 0.08,
                    "zcr_std": 0.02,
                    "mfcc_11_std": 2.0,
                    "key": "C major",
                },
            ]
        )
        self.model_report = {
            "top_features": [
                {"feature": "zcr_std", "importance": 0.09},
                {"feature": "mfcc_11_std", "importance": 0.07},
                {"feature": "mfcc_1_std", "importance": 0.06},
            ],
            "validation": {
                "distance_score_roc_auc": 0.8613,
                "xgboost_oof_roc_auc": 0.9535,
                "model_feature_source_separability_auc": 0.9352,
            },
        }

    def test_select_research_features_adds_available_top_mfcc_without_duplicates(self):
        features = select_research_features(self.df, self.model_report)

        self.assertIn("tempo", features)
        self.assertIn("zcr_std", features)
        self.assertIn("mfcc_11_std", features)
        self.assertNotIn("mfcc_1_std", features)
        self.assertEqual(len(features), len(set(features)))

    def test_dataset_snapshot_prefers_qc_counts_and_validation_metrics(self):
        qc_report = {
            "dataset": {
                "rows_extracted": 3975,
                "label_source_counts": {"charted": 1996, "archive_low_download": 1979},
            },
            "extraction": {"success_rate": 0.9985},
            "source_bias": {"source_separability_auc_audio_features": 0.9488},
            "audio_source_quality": {"source_separability_auc_format_only": 1.0},
        }

        snapshot = build_dataset_snapshot(qc_report, self.model_report)

        self.assertEqual(snapshot["feature_rows"], 3975)
        self.assertEqual(snapshot["charted_rows"], 1996)
        self.assertEqual(snapshot["low_download_rows"], 1979)
        self.assertAlmostEqual(snapshot["feature_extraction_success_rate"], 0.9985)
        self.assertAlmostEqual(snapshot["source_auc"], 0.9488)
        self.assertAlmostEqual(snapshot["quality_source_auc"], 1.0)
        self.assertAlmostEqual(snapshot["distance_score_auc"], 0.8613)

    def test_summarize_feature_profiles_returns_charted_and_tier_stats(self):
        summary = summarize_feature_profiles(self.df, self.model_report)
        tempo_row = next(row for row in summary["feature_profiles"] if row["feature"] == "tempo")

        self.assertEqual(tempo_row["charted_median"], 126.0)
        self.assertEqual(tempo_row["low_stream_median"], 90.0)
        self.assertEqual(tempo_row["top20_median"], 120.0)
        self.assertEqual(tempo_row["chart_21_50_median"], 132.0)
        self.assertIsNone(tempo_row["chart_51_100_median"])

    def test_summarize_key_distribution_counts_charted_keys(self):
        summary = summarize_key_distribution(self.df, top_n=2)

        self.assertEqual(summary[0]["key"], "C major")
        self.assertEqual(summary[0]["all_count"], 2)
        self.assertEqual(summary[0]["charted_count"], 1)

    def test_compare_track_to_reference_classifies_relative_band(self):
        row = pd.DataFrame([{"tempo": 140, "rms_mean": 0.20, "zcr_std": 0.01}])
        metadata = {
            "charted_feature_median": {"tempo": 120.0, "rms_mean": 0.20, "zcr_std": 0.05},
            "charted_feature_iqr": {"tempo": 20.0, "rms_mean": 0.10, "zcr_std": 0.02},
        }

        comparison = compare_track_to_reference(row, metadata, ["tempo", "rms_mean", "zcr_std"])

        by_feature = {item["feature"]: item for item in comparison}
        self.assertEqual(by_feature["tempo"]["status"], "above")
        self.assertEqual(by_feature["rms_mean"]["status"], "inside")
        self.assertEqual(by_feature["zcr_std"]["status"], "below")

    def test_markdown_report_contains_bias_aware_language(self):
        summary = {
            "snapshot": {
                "feature_rows": 3,
                "charted_rows": 2,
                "low_download_rows": 1,
                "source_auc": 0.9488,
                "quality_source_auc": 1.0,
                "distance_score_auc": 0.8613,
                "xgboost_oof_auc": 0.9535,
            },
            "feature_profiles": [
                {
                    "feature": "tempo",
                    "label": "Tempo",
                    "charted_median": 126.0,
                    "charted_q25": 123.0,
                    "charted_q75": 129.0,
                    "low_stream_median": 90.0,
                    "top20_median": 120.0,
                    "chart_21_50_median": 132.0,
                    "chart_51_100_median": None,
                }
            ],
            "key_distribution": [{"key": "C major", "all_count": 2, "charted_count": 1}],
        }

        markdown = format_research_markdown(summary)

        self.assertIn("Features of Popular-Reference Tracks", markdown)
        self.assertIn("in this dataset", markdown)
        self.assertIn("source/domain", markdown)
        self.assertIn("| Tempo |", markdown)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_research_insights -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.research_insights'`.

- [ ] **Step 3: Add the research helper implementation**

Create `src/research_insights.py` with this content:

```python
"""Research-summary helpers for the hit-likeness Streamlit app and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RESEARCH_FEATURES = [
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
    "rms_mean": "RMS energy mean",
    "rms_std": "RMS energy variation",
    "chroma_mean": "Chroma mean",
    "chroma_std": "Chroma variation",
    "tonnetz_mean": "Tonnetz mean",
    "tonnetz_std": "Tonnetz variation",
    "zcr_std": "Zero-crossing variation",
}

TIER_COLUMNS = {
    "top20": "top20_median",
    "chart_21_50": "chart_21_50_median",
    "chart_51_100": "chart_51_100_median",
    "low_stream": "low_stream_median",
}


def load_json_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report if present; return an empty dict if it is unavailable."""
    report_path = Path(path)
    if not report_path.exists():
        return {}
    with report_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_features(path: str | Path) -> pd.DataFrame:
    """Load extracted feature rows if present; return an empty frame if unavailable."""
    feature_path = Path(path)
    if not feature_path.exists():
        return pd.DataFrame()
    return pd.read_csv(feature_path)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rounded(value: Any, digits: int = 4) -> float | None:
    safe_value = _safe_float(value)
    if safe_value is None:
        return None
    return round(safe_value, digits)


def _is_charted(df: pd.DataFrame) -> pd.Series:
    if "popular" in df.columns:
        numeric = pd.to_numeric(df["popular"], errors="coerce").fillna(0)
        return numeric.astype(int).eq(1)
    if "label_source" in df.columns:
        return df["label_source"].astype(str).eq("charted")
    return pd.Series(False, index=df.index)


def top_model_mfcc_features(model_report: dict[str, Any], limit: int = 3) -> list[str]:
    """Return top MFCC features from the observability report."""
    features: list[str] = []
    for item in model_report.get("top_features", []):
        feature = item.get("feature")
        if isinstance(feature, str) and feature.startswith("mfcc_") and feature not in features:
            features.append(feature)
        if len(features) >= limit:
            break
    return features


def select_research_features(
    df: pd.DataFrame,
    model_report: dict[str, Any] | None = None,
) -> list[str]:
    """Choose available features for the compact research section."""
    model_report = model_report or {}
    requested = DEFAULT_RESEARCH_FEATURES + top_model_mfcc_features(model_report)
    selected: list[str] = []
    for feature in requested:
        if feature in df.columns and feature not in selected:
            selected.append(feature)
    return selected


def build_dataset_snapshot(
    qc_report: dict[str, Any] | None,
    model_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract the metrics needed for the app's dataset snapshot."""
    qc_report = qc_report or {}
    model_report = model_report or {}
    dataset = qc_report.get("dataset", {})
    source_counts = dataset.get("label_source_counts", {})
    extraction = qc_report.get("extraction", {})
    source_bias = qc_report.get("source_bias", {})
    quality = qc_report.get("audio_source_quality", {})
    validation = model_report.get("validation", {})

    return {
        "feature_rows": dataset.get("rows_extracted") or model_report.get("dataset", {}).get("rows_used"),
        "charted_rows": source_counts.get("charted"),
        "low_download_rows": source_counts.get("archive_low_download"),
        "feature_extraction_success_rate": extraction.get("success_rate"),
        "source_auc": source_bias.get("source_separability_auc_audio_features"),
        "quality_source_auc": quality.get("source_separability_auc_format_only"),
        "distance_score_auc": validation.get("distance_score_roc_auc"),
        "xgboost_oof_auc": validation.get("xgboost_oof_roc_auc"),
        "model_feature_source_auc": validation.get("model_feature_source_separability_auc"),
    }


def _median_for_mask(df: pd.DataFrame, mask: pd.Series, feature: str) -> float | None:
    if feature not in df.columns:
        return None
    values = pd.to_numeric(df.loc[mask, feature], errors="coerce").dropna()
    if values.empty:
        return None
    return _rounded(values.median())


def summarize_feature_profiles(
    df: pd.DataFrame,
    model_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize selected feature distributions by charted reference and tier."""
    if df.empty:
        return {"feature_profiles": []}

    selected_features = select_research_features(df, model_report)
    charted_mask = _is_charted(df)
    rows: list[dict[str, Any]] = []

    for feature in selected_features:
        values = pd.to_numeric(df.loc[charted_mask, feature], errors="coerce").dropna()
        if values.empty:
            continue

        row: dict[str, Any] = {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "charted_median": _rounded(values.median()),
            "charted_q25": _rounded(values.quantile(0.25)),
            "charted_q75": _rounded(values.quantile(0.75)),
            "low_stream_median": None,
            "top20_median": None,
            "chart_21_50_median": None,
            "chart_51_100_median": None,
        }

        if "popularity_tier" in df.columns:
            for tier, output_key in TIER_COLUMNS.items():
                tier_mask = df["popularity_tier"].astype(str).eq(tier)
                row[output_key] = _median_for_mask(df, tier_mask, feature)

        rows.append(row)

    return {"feature_profiles": rows}


def summarize_key_distribution(df: pd.DataFrame, top_n: int = 8) -> list[dict[str, Any]]:
    """Return the most common keys with all-row and charted-only counts."""
    if df.empty or "key" not in df.columns:
        return []

    charted_mask = _is_charted(df)
    all_counts = df["key"].fillna("Unknown").astype(str).value_counts().head(top_n)
    charted_counts = df.loc[charted_mask, "key"].fillna("Unknown").astype(str).value_counts()

    return [
        {
            "key": key,
            "all_count": int(all_count),
            "charted_count": int(charted_counts.get(key, 0)),
        }
        for key, all_count in all_counts.items()
    ]


def compare_track_to_reference(
    row: pd.DataFrame,
    metadata: dict[str, Any] | None,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Compare an uploaded track to a robust charted-reference band."""
    metadata = metadata or {}
    medians = metadata.get("charted_feature_median", {})
    iqrs = metadata.get("charted_feature_iqr", {})
    comparisons: list[dict[str, Any]] = []

    if row.empty:
        return comparisons

    for feature in feature_names:
        if feature not in row.columns or feature not in medians:
            continue

        value = _safe_float(row[feature].iloc[0])
        median = _safe_float(medians.get(feature))
        iqr = _safe_float(iqrs.get(feature)) or 1.0
        if value is None or median is None:
            continue

        lower = median - iqr / 2
        upper = median + iqr / 2
        if value < lower:
            status = "below"
        elif value > upper:
            status = "above"
        else:
            status = "inside"

        comparisons.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": _rounded(value),
                "reference_median": _rounded(median),
                "reference_low": _rounded(lower),
                "reference_high": _rounded(upper),
                "status": status,
            }
        )

    return comparisons


def build_research_summary(
    features_path: str | Path = "data/processed/extended_features.csv",
    qc_report_path: str | Path = "reports/dataset_qc.json",
    model_report_path: str | Path = "reports/model_observability.json",
) -> dict[str, Any]:
    """Build the full research summary from current project artifacts."""
    df = load_features(features_path)
    qc_report = load_json_report(qc_report_path)
    model_report = load_json_report(model_report_path)
    feature_summary = summarize_feature_profiles(df, model_report)
    return {
        "snapshot": build_dataset_snapshot(qc_report, model_report),
        "feature_profiles": feature_summary["feature_profiles"],
        "key_distribution": summarize_key_distribution(df),
    }


def _format_number(value: Any) -> str:
    safe_value = _safe_float(value)
    if safe_value is None:
        return "N/A"
    if abs(safe_value) < 1:
        return f"{safe_value:.4f}"
    return f"{safe_value:.2f}"


def _format_percent(value: Any) -> str:
    safe_value = _safe_float(value)
    if safe_value is None:
        return "N/A"
    return f"{safe_value:.2%}"


def format_research_markdown(summary: dict[str, Any]) -> str:
    """Render the research summary as a markdown report."""
    snapshot = summary.get("snapshot", {})
    feature_profiles = summary.get("feature_profiles", [])
    key_distribution = summary.get("key_distribution", [])

    feature_rows = "\n".join(
        "| {label} | {charted_median} | {charted_iqr} | {top20} | {chart_21_50} | {chart_51_100} | {low_stream} |".format(
            label=row["label"],
            charted_median=_format_number(row.get("charted_median")),
            charted_iqr=f"{_format_number(row.get('charted_q25'))} to {_format_number(row.get('charted_q75'))}",
            top20=_format_number(row.get("top20_median")),
            chart_21_50=_format_number(row.get("chart_21_50_median")),
            chart_51_100=_format_number(row.get("chart_51_100_median")),
            low_stream=_format_number(row.get("low_stream_median")),
        )
        for row in feature_profiles
    )
    if not feature_rows:
        feature_rows = "| N/A | N/A | N/A | N/A | N/A | N/A | N/A |"

    key_rows = "\n".join(
        f"| {row['key']} | {row['all_count']} | {row['charted_count']} |"
        for row in key_distribution
    )
    if not key_rows:
        key_rows = "| N/A | N/A | N/A |"

    return f"""# Features of Popular-Reference Tracks

Generated from the current local dataset. Treat every finding as exploratory and specific to tracks in this dataset.

## Dataset Snapshot

- Feature rows: `{snapshot.get('feature_rows', 'N/A')}`
- Charted reference rows: `{snapshot.get('charted_rows', 'N/A')}`
- Low-download reference rows: `{snapshot.get('low_download_rows', 'N/A')}`
- Feature extraction success rate: `{_format_percent(snapshot.get('feature_extraction_success_rate'))}`
- Distance-score ROC-AUC on the current proxy task: `{_format_number(snapshot.get('distance_score_auc'))}`
- XGBoost out-of-fold ROC-AUC on the current proxy task: `{_format_number(snapshot.get('xgboost_oof_auc'))}`
- Audio-feature source separability ROC-AUC: `{_format_number(snapshot.get('source_auc'))}`
- Technical-quality source separability ROC-AUC: `{_format_number(snapshot.get('quality_source_auc'))}`

## Feature Profile

The table compares charted-reference tracks with chart tiers and low-download reference tracks. Values are medians unless a range is shown.

| Feature | Charted median | Charted IQR | Top 20 | Chart 21-50 | Chart 51-100 | Low-download |
|---|---:|---:|---:|---:|---:|---:|
{feature_rows}

## Key Distribution

| Key | All tracks | Charted tracks |
|---|---:|---:|
{key_rows}

## Interpretation

These summaries describe audio-feature patterns in this dataset, not universal rules for popular music. The current dataset still has strong source/domain bias: charted tracks are mostly uniform 30-second MP3 previews, while low-download Internet Archive tracks have different technical characteristics. Use the feature profile as a research view into the current reference set, not as causal production advice.

## Data Roadmap

The next scientific step is to reduce source/domain bias before calibrating any score as a probability. A later dataset iteration can continue using Internet Archive plus `yt-dlp`, but should prioritize source-matched negatives or charted audio with comparable duration, codec, sample rate, bitrate mode, and file-generation pipeline.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m unittest tests.test_research_insights -v
```

Expected: PASS for all six tests.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/research_insights.py tests/test_research_insights.py
git commit -m "test: add research insight helpers"
```

If `src/` has unrelated pre-existing edits, inspect `git diff --cached` before committing and unstage unrelated files.

## Task 2: Generate the Popular Feature Research Report

**Files:**
- Create: `src/write_popular_feature_research.py`
- Create: `reports/popular_feature_research.md`
- Test: `tests/test_research_insights.py`

- [ ] **Step 1: Add a CLI test for markdown report generation**

Append this test method inside `ResearchInsightsTest` in `tests/test_research_insights.py`:

```python
    def test_format_research_markdown_includes_data_roadmap(self):
        summary = {
            "snapshot": {
                "feature_rows": 3,
                "charted_rows": 2,
                "low_download_rows": 1,
                "source_auc": 0.9488,
                "quality_source_auc": 1.0,
            },
            "feature_profiles": [],
            "key_distribution": [],
        }

        markdown = format_research_markdown(summary)

        self.assertIn("Data Roadmap", markdown)
        self.assertIn("Internet Archive", markdown)
        self.assertIn("yt-dlp", markdown)
```

- [ ] **Step 2: Run tests to verify they pass before adding the CLI**

Run:

```bash
python -m unittest tests.test_research_insights -v
```

Expected: PASS. The markdown formatter already supports this behavior from Task 1.

- [ ] **Step 3: Create the report writer script**

Create `src/write_popular_feature_research.py` with this content:

```python
"""Write the popular-reference feature research report from current artifacts."""

from pathlib import Path

from research_insights import build_research_summary, format_research_markdown


OUTPUT_PATH = Path("reports/popular_feature_research.md")


def main() -> None:
    summary = build_research_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(format_research_markdown(summary), encoding="utf-8")
    print(f"Research report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the report writer**

Run:

```bash
python src/write_popular_feature_research.py
```

Expected: command exits 0 and prints `Research report saved to reports/popular_feature_research.md`.

- [ ] **Step 5: Inspect the generated report**

Run:

```bash
sed -n '1,220p' reports/popular_feature_research.md
```

Expected: report contains `Features of Popular-Reference Tracks`, a feature profile table, bias-aware interpretation, and a data roadmap. If `N/A` appears for every feature row, stop and inspect `data/processed/extended_features.csv` path/columns before continuing.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/write_popular_feature_research.py tests/test_research_insights.py reports/popular_feature_research.md
git commit -m "docs: add popular feature research report"
```

If report files had unrelated pre-existing edits, inspect `git diff --cached -- reports` before committing.

## Task 3: Add Dataset Snapshot, Research, and Method Sections to Streamlit

**Files:**
- Modify: `app.py`
- Uses: `src/research_insights.py`

- [ ] **Step 1: Add app-level imports**

In `app.py`, add these imports after the existing `plotly.express` import:

```python
from src.research_insights import (
    build_dataset_snapshot,
    build_research_summary,
    compare_track_to_reference,
    load_json_report,
)
```

- [ ] **Step 2: Add cached artifact loaders**

Add these functions below `load_models()` in `app.py`:

```python
@st.cache_data
def load_reports():
    return {
        "qc": load_json_report("reports/dataset_qc.json"),
        "model": load_json_report("reports/model_observability.json"),
    }


@st.cache_data
def load_research_summary():
    return build_research_summary()
```

Then after `model, feature_cols, metadata = load_models()`, add:

```python
reports = load_reports()
research_summary = load_research_summary()
dataset_snapshot = build_dataset_snapshot(reports["qc"], reports["model"])
```

- [ ] **Step 3: Add small formatting helpers**

Add these helper functions above the sidebar block:

```python
def format_optional_number(value, digits=3):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_optional_percent(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def score_band(score):
    if score >= 70:
        return "High similarity to the charted audio reference profile", "success"
    if score >= 40:
        return "Middle range: some traits resemble the charted reference", "warning"
    return "Low similarity to the charted audio reference profile", "info"
```

- [ ] **Step 4: Replace the current intro with reviewer-focused tabs**

Replace the current main intro, `How to read the score` expander, warning, uploader declaration, and current upload block entry with this structure. Keep the existing analysis internals for now; Task 4 rewrites the post-upload report.

```python
st.title("Music Hit-Likeness Analyzer")
st.write(
    "Upload a track and get a guided audio-similarity report against the current charted reference dataset."
)

if model is None:
    st.error("Model artifacts are missing. Train locally with `src/train_extended_model.py` first.")
    st.stop()

analyze_tab, research_tab, method_tab = st.tabs(["Analyze", "Research insights", "Method & limits"])

with analyze_tab:
    st.subheader("Analyze one track")
    st.write(
        "The headline score is a percentile of closeness to the weighted charted audio-feature profile. "
        "It is not a calibrated probability of commercial success."
    )

    snap_cols = st.columns(4)
    snap_cols[0].metric("Feature rows", dataset_snapshot.get("feature_rows") or "N/A")
    snap_cols[1].metric("Charted reference", dataset_snapshot.get("charted_rows") or "N/A")
    snap_cols[2].metric("Low-download reference", dataset_snapshot.get("low_download_rows") or "N/A")
    snap_cols[3].metric(
        "Extraction success",
        format_optional_percent(dataset_snapshot.get("feature_extraction_success_rate")),
    )

    source_auc = dataset_snapshot.get("source_auc")
    if source_auc is not None and source_auc >= 0.8:
        st.warning(
            "Current research caveat: audio features still separate charted vs low-download sources strongly "
            f"(source ROC-AUC {source_auc:.3f}). Interpret scores as reference-set similarity, not proof of popularity."
        )
    else:
        st.info("Source-bias diagnostics are unavailable or below the warning threshold.")

    uploaded_file = st.file_uploader("Upload a track (MP3, WAV)", type=["mp3", "wav"])
```

Move the existing `if uploaded_file is not None:` block under `with analyze_tab:` so upload analysis remains inside the Analyze tab.

- [ ] **Step 5: Add the Research insights tab content**

Add this block after the `with analyze_tab:` block and before `with method_tab:`:

```python
with research_tab:
    st.subheader("Features of popular-reference tracks")
    st.write(
        "This section summarizes patterns in the current dataset. It is exploratory and should be read with the source-bias warning in mind."
    )

    feature_profiles = research_summary.get("feature_profiles", [])
    if feature_profiles:
        profile_df = pd.DataFrame(feature_profiles)
        display_cols = [
            "label",
            "charted_median",
            "charted_q25",
            "charted_q75",
            "top20_median",
            "chart_21_50_median",
            "chart_51_100_median",
            "low_stream_median",
        ]
        display_cols = [col for col in display_cols if col in profile_df.columns]
        st.dataframe(profile_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("Research summary is unavailable. Regenerate `data/processed/extended_features.csv` first.")

    key_distribution = research_summary.get("key_distribution", [])
    if key_distribution:
        key_df = pd.DataFrame(key_distribution)
        key_fig = px.bar(
            key_df,
            x="key",
            y=["all_count", "charted_count"],
            barmode="group",
            title="Most common estimated keys in the current dataset",
        )
        st.plotly_chart(key_fig, use_container_width=True)

    st.caption(
        "Full write-up: `reports/popular_feature_research.md`. "
        "Findings are phrased as dataset-specific because source/domain artifacts remain strong."
    )
```

- [ ] **Step 6: Add the Method & limits tab content**

Add this block after the research tab:

```python
with method_tab:
    st.subheader("Method and limits")
    st.write(
        "The app extracts local audio features from the first 30 seconds of the uploaded file, then compares those features "
        "with a weighted charted-song reference profile. XGBoost and SHAP are diagnostic signals, not the headline score."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Distance ROC-AUC", format_optional_number(dataset_snapshot.get("distance_score_auc")))
    metric_cols[1].metric("XGBoost OOF ROC-AUC", format_optional_number(dataset_snapshot.get("xgboost_oof_auc")))
    metric_cols[2].metric("Audio source ROC-AUC", format_optional_number(dataset_snapshot.get("source_auc")))
    metric_cols[3].metric("Quality source ROC-AUC", format_optional_number(dataset_snapshot.get("quality_source_auc")))

    st.warning(
        "The model does not know marketing, artist reputation, release timing, playlist exposure, platform dynamics, or culture. "
        "Use the result as a comparative audio score, not a chart-success forecast."
    )
    st.write(
        "Next data milestone: reduce source/domain bias with source-matched negatives or charted audio collected through a comparable audio pipeline."
    )
```

- [ ] **Step 7: Run syntax check**

Run:

```bash
python -m py_compile app.py src/research_insights.py
```

Expected: exits 0.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add app.py
git commit -m "feat: add guided app context sections"
```

Because `app.py` is already dirty in the current worktree, inspect `git diff -- app.py` and `git diff --cached -- app.py` before committing.

## Task 4: Upgrade the Post-Upload Result Into a Guided Report

**Files:**
- Modify: `app.py`
- Uses: `src/research_insights.py`

- [ ] **Step 1: Add feature-comparison rendering helper**

Add this function near the other app helper functions:

```python
def render_feature_comparison(row, metadata, research_summary):
    feature_frame = pd.DataFrame(research_summary.get("feature_profiles", []))
    if feature_frame.empty:
        feature_names = [name for name in ["tempo", "rms_mean", "chroma_mean", "tonnetz_mean", "zcr_std"] if name in row.columns]
    else:
        feature_names = feature_frame["feature"].dropna().astype(str).head(10).tolist()

    comparisons = compare_track_to_reference(row, metadata, feature_names)
    if not comparisons:
        st.info("Feature comparison is unavailable for this track and metadata combination.")
        return

    comparison_df = pd.DataFrame(comparisons)
    st.dataframe(
        comparison_df[["label", "value", "reference_median", "reference_low", "reference_high", "status"]],
        use_container_width=True,
        hide_index=True,
    )
```

- [ ] **Step 2: Replace the current result header and score block**

Inside the existing `else:` branch after scoring, replace the current `Analysis Result` section through the `Audio Features` metric block with this:

```python
                st.markdown("---")
                st.subheader("Guided analysis report")

                band_text, band_type = score_band(score)
                score_cols = st.columns(3)
                score_cols[0].metric("Hit-likeness percentile", f"{score:.1f}%")
                score_cols[1].metric("XGBoost diagnostic percentile", f"{model_percentile:.1f}%")
                score_cols[2].metric("Charted-reference distance", f"{charted_distance:.2f}")

                st.progress(int(max(0, min(score, 100))))
                if band_type == "success":
                    st.success(band_text)
                elif band_type == "warning":
                    st.warning(band_text)
                else:
                    st.info(band_text)

                if in_distribution:
                    st.success(f"Reliability: within training feature range ({distance:.2f}/{threshold:.2f}).")
                else:
                    st.warning(
                        f"Low reliability: this track is far from the training feature range ({distance:.2f}/{threshold:.2f})."
                    )

                st.caption(
                    f"Raw XGBoost score: {raw_score * 100:.1f}%. "
                    "The XGBoost signal is shown as a diagnostic model output, not as the headline score."
                )

                st.write("**Extracted audio profile**")
                audio_cols = st.columns(6)
                audio_cols[0].metric("BPM", f"{features['tempo']:.0f}")
                audio_cols[1].metric("Key", features["key"])
                audio_cols[2].metric("RMS mean", f"{features['rms_mean']:.3f}")
                audio_cols[3].metric("Chroma mean", f"{features.get('chroma_mean', 0):.3f}")
                audio_cols[4].metric("Tonnetz mean", f"{features.get('tonnetz_mean', 0):.3f}")
                audio_cols[5].metric("ZCR std", f"{features.get('zcr_std', 0):.3f}")

                st.write("**Uploaded track vs charted reference band**")
                render_feature_comparison(row, metadata, research_summary)
```

- [ ] **Step 3: Rename and reframe the SHAP section**

Replace the current SHAP section heading and chart title strings with:

```python
                st.markdown("---")
                st.subheader("Diagnostic model contributions")
                st.caption(
                    "SHAP values explain the secondary XGBoost diagnostic model in its raw feature space. "
                    "They are not causal production advice."
                )
```

In the Plotly call, change `title` to:

```python
                    title="Top diagnostic XGBoost feature contributions (SHAP)",
```

- [ ] **Step 4: Add a research teaser after SHAP**

After `st.plotly_chart(fig, use_container_width=True)`, add:

```python
                st.info(
                    "Research context: the feature comparison is anchored to the current charted-reference dataset. "
                    "Source/domain bias remains high, so treat this as exploratory audio similarity rather than a popularity forecast."
                )
```

- [ ] **Step 5: Run syntax check**

Run:

```bash
python -m py_compile app.py src/research_insights.py
```

Expected: exits 0.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add app.py
git commit -m "feat: guide uploaded track report"
```

Inspect staged `app.py` before committing because the file was dirty before this plan.

## Task 5: Update README With Research Report Pointer

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add research report to the project capabilities list**

In `README.md`, add this bullet under `What It Does` after the observability bullet:

```markdown
- Adds a concise research write-up on popular-reference audio features in `reports/popular_feature_research.md`.
```

- [ ] **Step 2: Add the report to Observability**

In the `Observability` file list, add:

```markdown
- `reports/popular_feature_research.md`: bias-aware mini study of audio features in the current charted-reference dataset.
```

- [ ] **Step 3: Add a short Research Findings section**

Add this section after `Current Dataset Snapshot`:

```markdown
## Research Findings

See `reports/popular_feature_research.md` for a compact, bias-aware study of the current dataset's popular-reference audio features. The report summarizes charted-reference medians and interquartile ranges for tempo, energy, chroma, tonnetz, zero-crossing variation, and top diagnostic MFCC features, then compares them with chart tiers and low-download reference tracks.

The findings are intentionally phrased as dataset-specific. Current source/domain diagnostics still show strong technical mismatch between charted 30-second previews and low-download Internet Archive tracks, so the report should be read as exploratory EDA, not as a universal formula for popular songs.
```

- [ ] **Step 4: Commit Task 5**

Run:

```bash
git add README.md
git commit -m "docs: summarize popular feature research"
```

Inspect staged `README.md` before committing because the file was dirty before this plan.

## Task 6: Final Verification

**Files:**
- Verify: `app.py`
- Verify: `src/research_insights.py`
- Verify: `src/write_popular_feature_research.py`
- Verify: `tests/test_research_insights.py`
- Verify: `reports/popular_feature_research.md`
- Verify: `README.md`

- [ ] **Step 1: Run unit tests**

Run:

```bash
python -m unittest tests.test_research_insights -v
```

Expected: PASS for all tests.

- [ ] **Step 2: Run Python syntax checks**

Run:

```bash
python -m py_compile app.py src/research_insights.py src/write_popular_feature_research.py tests/test_research_insights.py
```

Expected: exits 0.

- [ ] **Step 3: Regenerate the research report**

Run:

```bash
python src/write_popular_feature_research.py
```

Expected: exits 0 and updates `reports/popular_feature_research.md`.

- [ ] **Step 4: Start Streamlit**

Run:

```bash
streamlit run app.py
```

Expected: Streamlit starts and prints a local URL. If the shell environment does not have dependencies installed, rerun with the project's configured interpreter, for example `.venv_torch\\Scripts\\streamlit.exe run app.py` on the Windows-style environment documented in `README.md`.

- [ ] **Step 5: Manually inspect the app**

In the browser:

- Confirm `Analyze`, `Research insights`, and `Method & limits` tabs render.
- Confirm the Analyze tab shows dataset metrics and source-bias warning before upload.
- Confirm the Research insights tab shows a feature-profile table and key chart.
- Upload a known local MP3/WAV if one is available and click Analyze track.
- Confirm the post-upload report shows hit-likeness percentile, reliability, XGBoost diagnostic percentile, extracted audio profile, feature comparison table, SHAP chart, and research-context warning.

- [ ] **Step 6: Stop Streamlit**

Stop the running Streamlit process with `Ctrl+C` in the terminal session.

- [ ] **Step 7: Check final git state**

Run:

```bash
git status --short
```

Expected: no unexpected new files. Existing user changes that predated this plan may still be present; list them in the final implementation summary instead of reverting them.
