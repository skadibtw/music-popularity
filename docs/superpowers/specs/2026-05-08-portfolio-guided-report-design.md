# Portfolio Guided Report Design

Date: 2026-05-08

## Context

Music Hit-Likeness Analyzer is a research-first audio-feature project. It scores how close an uploaded track is to a weighted charted-song audio reference distribution. The score is a charted-reference similarity percentile, not a calibrated probability of commercial success.

The project already has the key research artifacts: fixed 30-second feature extraction, a charted-vs-low-download dataset, XGBoost diagnostic model, SHAP explanations, dataset QC reports, source-bias reports, and model observability reports. The current Streamlit app exposes only a narrow upload-and-score flow, so a portfolio reviewer can miss the project's strongest parts: honest score semantics, reliability checks, dataset caveats, and the feature study.

## Target User

Primary user for this pass: a portfolio or demo reviewer who opens the Streamlit app, uploads one MP3/WAV file, and needs to understand the project in about two minutes.

Secondary reader: someone browsing README/reports who wants a concise research summary of audio features in the current dataset.

## Goals

- Make Streamlit the main entry point for the demo.
- Turn the upload result into a guided one-page analysis report.
- Add a compact research section about features of popular-reference tracks in the current dataset.
- Keep all claims scientifically honest by showing source/domain bias and score limitations inside the UI.
- Add a fuller research write-up in `reports/popular_feature_research.md` and summarize it from `README.md`.
- Leave new dataset collection as a later phase, not part of this implementation pass.

## Non-Goals

- Do not present the score as hit probability or commercial-success prediction.
- Do not add production advice such as "change this feature to make a hit."
- Do not add batch analysis or multi-track library workflows.
- Do not collect new audio or run `yt-dlp` in this pass.
- Do not perform broad refactors outside the app/report helpers needed for the guided report.

## Product Shape

Use the "Guided Analysis Report" approach.

Before upload, the app should show:

- A clear upload panel for MP3/WAV files.
- A short explanation of what the score means.
- A visible dataset snapshot: feature rows, charted reference rows, low-download reference rows, and current source-bias warning.
- Navigation or sections for Analyze, Research insights, and Method/limits.

After upload and analysis, the app should show one structured report:

- Main hit-likeness percentile with a plain-language interpretation band.
- Reliability status from the existing robust feature distance/OOD check.
- Secondary XGBoost percentile, clearly labeled as diagnostic.
- Basic extracted audio features: BPM, key, RMS energy, chroma mean/std, tonnetz mean/std, and the top available diagnostic MFCC/ZCR features from the model observability report.
- Feature comparison against the charted reference profile using "inside / above / below reference range" language.
- SHAP contribution chart renamed and framed as diagnostic model contributions, not causal explanation.
- A research insight teaser linking the uploaded track back to the feature study and limitations.

## Research Insights

Add a small research section in Streamlit, a fuller written version in `reports/popular_feature_research.md`, and a concise link/summary in `README.md`.

Recommended content:

- Popular-reference feature profile: median and interquartile range for interpretable traits such as tempo, RMS energy, chroma, tonnetz, and selected MFCC summaries.
- Tier comparison: compare `top20`, `chart_21_50`, `chart_51_100`, and `low_stream` where the data supports it.
- Key/mode distribution summary using the existing key extraction.
- Reuse existing static plots where helpful: tempo by source, duration by source, top keys, and feature importance.
- Bias-aware interpretation: every summary should be phrased as "in this dataset" and mention that source/domain artifacts remain strong.

First-pass feature list:

- `tempo`
- `rms_mean`
- `rms_std`
- `chroma_mean`
- `chroma_std`
- `tonnetz_mean`
- `tonnetz_std`
- `zcr_std`
- the top three available MFCC features from `reports/model_observability.json`

Avoid:

- Universal claims like "popular songs are always..."
- Causal claims like "increase energy to become popular."
- Hiding the source mismatch between charted previews and low-download netlabels files.

## Data Roadmap

The implementation should not download more data, but the UI/docs should make the next scientific step clear:

- Continue improving the dataset through Internet Archive plus `yt-dlp` in a separate iteration.
- Prefer source-matched negatives or charted audio with comparable duration, codec, sample rate, bitrate mode, and file-generation pipeline.
- Treat source/domain bias reduction as the next major research milestone before score calibration.

## Architecture

Keep `app.py` as the Streamlit entry point, but avoid growing it into a large mixed UI/data file.

Recommended structure:

- Add small helper functions for loading JSON/CSV report artifacts.
- Add small helper functions for deriving research summaries from existing processed data and metadata.
- Keep existing scoring helpers in `src/music_success_predictor.py`.
- Only introduce a new module if the Streamlit file becomes hard to read; likely candidate: `src/research_insights.py`.

The implementation should follow existing project conventions and use current artifacts:

- `data/processed/extended_features.csv`
- `reports/dataset_qc.json`
- `reports/model_observability.json`
- `reports/popular_feature_research.md`
- `models/xgboost_score_metadata.pkl`
- existing plots under `plots/`

## Data Flow

1. App startup loads model artifacts and observability/QC artifacts.
2. Before upload, the app renders dataset snapshot and method limitations from existing reports.
3. Research insights are derived from `extended_features.csv` and cached with Streamlit caching.
4. On upload, the app saves the file to a temporary path, extracts the first 30 seconds of features, and computes the existing charted-reference score, XGBoost diagnostic score, SHAP values, and OOD distance.
5. The report compares uploaded-track features to charted-reference medians/IQRs from model metadata and research summaries.
6. Temporary upload files are removed after processing.

## Error Handling

- If model artifacts are missing, show the existing training instruction and stop analysis.
- If report artifacts are missing, keep upload analysis usable and show a warning that research/QC panels are unavailable.
- If `extended_features.csv` is missing, hide the research section and explain which pipeline step regenerates it.
- If audio processing fails, keep the current user-facing error and avoid showing partial model output.
- If a feature is missing from uploaded extraction or metadata, omit that comparison row instead of crashing.

## Verification

Minimum checks for implementation:

- Run a syntax/import check for touched Python files.
- Start Streamlit locally and verify that model/report artifacts load.
- Upload or analyze a known local MP3/WAV if available and confirm the guided report renders.
- Verify the research section renders from current CSV/JSON artifacts.
- Verify missing-report behavior by reasoning through helper defaults or with targeted tests if practical.

## Implementation Choices

- Final visual polish can stay modest; this is a working analysis tool, not a landing page.
- The research section should use the first-pass feature list above and omit individual missing columns instead of failing.
- A separate `src/research_insights.py` module should be created only if helper logic would make `app.py` too large.
