import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.research_insights import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_MODEL_REPORT_PATH,
    DEFAULT_QC_REPORT_PATH,
    PROJECT_ROOT,
    build_dataset_snapshot,
    build_research_summary,
    compare_track_to_reference,
    format_research_markdown,
    select_research_features,
    summarize_feature_profiles,
    summarize_key_distribution,
    top_model_mfcc_features,
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

    def test_top_model_mfcc_features_preserves_order_and_limit(self):
        model_report = {
            "top_features": [
                {"feature": "zcr_std", "importance": 0.09},
                {"feature": "mfcc_11_std", "importance": 0.07},
                {"feature": "mfcc_1_mean", "importance": 0.06},
                {"feature": "mfcc_2_std", "importance": 0.05},
            ]
        }

        features = top_model_mfcc_features(model_report, limit=2)

        self.assertEqual(features, ["mfcc_11_std", "mfcc_1_mean"])

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

    def test_dataset_snapshot_preserves_zero_qc_source_auc(self):
        qc_report = {
            "source_bias": {"source_separability_auc_audio_features": 0.0},
        }

        snapshot = build_dataset_snapshot(qc_report, self.model_report)

        self.assertEqual(snapshot["source_auc"], 0.0)

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

    def test_compare_track_to_reference_returns_app_required_keys(self):
        row = pd.DataFrame([{"tempo": 140}])
        metadata = {
            "charted_feature_median": {"tempo": 120.0},
            "charted_feature_iqr": {"tempo": 20.0},
        }

        comparison = compare_track_to_reference(row, metadata, ["tempo"])

        self.assertTrue(
            {"label", "value", "reference_median", "reference_low", "reference_high", "status"}
            <= comparison[0].keys()
        )
        self.assertEqual(comparison[0]["label"], "Tempo")
        self.assertEqual(comparison[0]["reference_low"], 110.0)
        self.assertEqual(comparison[0]["reference_high"], 130.0)

    def test_build_research_summary_accepts_injected_reports(self):
        qc_report = {
            "dataset": {
                "rows_extracted": 3,
                "label_source_counts": {"charted": 2, "archive_low_download": 1},
            }
        }

        summary = build_research_summary(
            features_df=self.df,
            qc_report=qc_report,
            model_report=self.model_report,
            key_top_n=1,
        )

        self.assertEqual(summary["snapshot"]["feature_rows"], 3)
        self.assertEqual(summary["key_distribution"], [{"key": "C major", "all_count": 2, "charted_count": 1}])
        self.assertIn("mfcc_11_std", summary["features"])

    def test_build_research_summary_loads_from_temp_paths(self):
        qc_report = {
            "dataset": {
                "rows_extracted": 3,
                "label_source_counts": {"charted": 2, "archive_low_download": 1},
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            features_path = f"{tmp_dir}/features.csv"
            qc_path = f"{tmp_dir}/qc.json"
            model_path = f"{tmp_dir}/model.json"
            self.df.to_csv(features_path, index=False)
            with open(qc_path, "w", encoding="utf-8") as qc_file:
                json.dump(qc_report, qc_file)
            with open(model_path, "w", encoding="utf-8") as model_file:
                json.dump(self.model_report, model_file)

            summary = build_research_summary(
                features_path=features_path,
                qc_report_path=qc_path,
                model_report_path=model_path,
                key_top_n=1,
            )

        self.assertEqual(summary["snapshot"]["charted_rows"], 2)
        tempo_row = next(row for row in summary["feature_profiles"] if row["feature"] == "tempo")
        self.assertEqual(tempo_row["charted_median"], 126.0)

    def test_default_paths_are_absolute_and_repo_root_anchored(self):
        self.assertTrue(PROJECT_ROOT.is_absolute())
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(DEFAULT_FEATURES_PATH, PROJECT_ROOT / "data/processed/extended_features.csv")
        self.assertEqual(DEFAULT_QC_REPORT_PATH, PROJECT_ROOT / "reports/dataset_qc.json")
        self.assertEqual(DEFAULT_MODEL_REPORT_PATH, PROJECT_ROOT / "reports/model_observability.json")

    def test_build_research_summary_explicit_paths_are_cwd_independent(self):
        qc_report = {
            "dataset": {
                "rows_extracted": 3,
                "label_source_counts": {"charted": 2, "archive_low_download": 1},
            }
        }

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            features_path = temp_root / "features.csv"
            qc_path = temp_root / "qc.json"
            model_path = temp_root / "model.json"
            other_cwd = temp_root / "other"
            other_cwd.mkdir()
            self.df.to_csv(features_path, index=False)
            with qc_path.open("w", encoding="utf-8") as qc_file:
                json.dump(qc_report, qc_file)
            with model_path.open("w", encoding="utf-8") as model_file:
                json.dump(self.model_report, model_file)

            os.chdir(other_cwd)
            try:
                summary = build_research_summary(
                    features_path=features_path,
                    qc_report_path=qc_path,
                    model_report_path=model_path,
                    key_top_n=1,
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(summary["snapshot"]["feature_rows"], 3)

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

    def test_write_popular_feature_research_output_path_is_repo_anchored(self):
        from src.write_popular_feature_research import OUTPUT_PATH

        self.assertTrue(OUTPUT_PATH.is_absolute())
        self.assertEqual(OUTPUT_PATH, PROJECT_ROOT / "reports/popular_feature_research.md")


if __name__ == "__main__":
    unittest.main()
