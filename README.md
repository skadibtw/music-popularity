# Music Hit-Likeness Analyzer

This project analyzes local audio files and scores how similar they are to a weighted reference distribution of charted songs. It is an exploratory ML/research project, not a real-world guarantee that an unknown song will chart.

## What It Does

- Extracts local audio features with `librosa` from a fixed 30-second preview window.
- Builds a dataset from chart-matched MP3 files plus low-download Internet Archive netlabels tracks.
- Treats all charted tracks as the popular reference class and low-download tracks as the low-popularity reference class.
- Uses a weighted charted-reference distance score as the headline score; XGBoost is a diagnostic model.
- Uses artist-grouped train/test splitting to reduce artist leakage.
- Encodes musical key with circular pitch-class features instead of ordinal labels.
- Provides a Streamlit UI for uploading a track and viewing a charted-reference audio similarity percentile.
- Shows SHAP explanations in the model's raw feature space.
- Warns when an uploaded track is outside the training feature distribution.
- Writes observability reports with dataset composition, validation metrics, score semantics, and feature importance.

## Important Limitation

The app reports a `hit potential score`, not a calibrated success probability.

The percentile is anchored to the weighted feature distribution of charted tracks in the local dataset. If the app shows `80%`, it means the uploaded track is closer to the charted audio reference profile than roughly 80% of tracks in the current reference set. The XGBoost percentile is shown separately as a secondary signal.

The current low-popularity reference comes from Internet Archive netlabels tracks with low item-level download counts. This is a proxy, not proof of universal low popularity. The latest dataset uses matched 30-second analysis windows, but the QC report still shows strong charted-vs-netlabels source/domain bias in the audio features. Read `reports/dataset_qc.md` before interpreting model metrics. Artist fame, marketing, release timing, playlist exposure, culture, and platform dynamics are not modeled.

## Project Structure

- `app.py`: Streamlit upload-and-score interface.
- `src/extract_extended_features.py`: builds `data/processed/extended_features.csv` from local MP3s.
- `src/train_extended_model.py`: trains the XGBoost model and exports model artifacts.
- `src/music_success_predictor.py`: audio feature extraction and shared feature helpers.
- `src/build_low_stream_manifest.py`: builds a low-download Internet Archive netlabels manifest.
- `src/sample_low_stream_manifest.py`: creates a capped reproducible staged sample.
- `src/qc_extended_dataset.py`: writes dataset QC and source-bias reports.
- `src/analyze_extended.py`: CLI analysis for one audio file.
- `data/processed/`: processed chart and extracted feature data.
- `models/`: exported model files.
- `plots/`: EDA and feature-importance plots.

## Current Dataset Snapshot

- Feature rows: `3975`
- Charted reference rows: `1996`
- Low-download netlabels rows: `1979`
- Feature extraction success rate: `99.85%`
- Median analyzed duration: `30.0s`
- Source separability ROC-AUC, audio features only: `0.9493`
- Model-feature source separability ROC-AUC after conservative feature exclusion: `0.9344`
- Excluded source-confounded model features: `spectral_bandwidth_mean`, `mfcc_0_mean`, `mfcc_0_std`, `spectral_rolloff_mean`, `spectral_centroid_mean`
- Main scientific risk: model metrics are still strongly confounded by source/domain differences.

## Quickstart

Install dependencies, then build features and train:

```bash
.venv_torch\Scripts\python.exe -m pip install yt-dlp
.venv_torch\Scripts\python.exe src\build_low_stream_manifest.py --target-rows 50000 --output data\raw\low_stream_tracks.csv --allow-missing-license --workers 12 --checkpoint-every 5000
.venv_torch\Scripts\python.exe src\sample_low_stream_manifest.py --input data\raw\low_stream_tracks.csv --output data\raw\low_stream_tracks_sample_2000.csv --target-rows 2000
.venv_torch\Scripts\python.exe src\download_low_stream_tracks.py --manifest data\raw\low_stream_tracks_sample_2000.csv --workers 4
.venv_torch\Scripts\python.exe src\extract_extended_features.py --low-streams-csv data\raw\low_stream_tracks_sample_2000.csv --checkpoint-every 100 --preview-seconds 30
.venv_torch\Scripts\python.exe src\qc_extended_dataset.py
.venv_torch\Scripts\python.exe src\train_extended_model.py
```

If `yt-dlp.exe` is stored in the project root, the downloader will use it automatically. The binary itself is ignored by Git.

`data/raw/low_stream_tracks.csv` contains an Internet Archive netlabels queue with `downloads <= 1000`, used as a practical proxy for low popularity. `data/raw/low_stream_tracks_sample_2000.csv` is the capped staged sample used for local downloads. Manifest format:

```csv
file_path,artist,title,stream_count,source_url,archive_item,license_url
music\low_stream\Some Artist - Some Song.mp3,Some Artist,Some Song,742,https://archive.org/download/.../song.mp3,https://archive.org/details/...,https://creativecommons.org/licenses/...
```

Run the web app:

```bash
.venv_torch\Scripts\streamlit.exe run app.py
```

## Observability

Training writes these files:

- `reports/model_observability.md`: human-readable dataset and model report.
- `reports/model_observability.json`: machine-readable version of the same report.
- `reports/dataset_qc.md`: dataset quality, source-bias, and staged-sample checks.
- `reports/dataset_qc.json`: machine-readable QC report.
- `plots/feature_importance_xgb.png`: top XGBoost feature importance chart.

The reports include label-source counts, target base rate, holdout metrics, out-of-fold ROC-AUC, OOD distance threshold, source separability, QC checks, and top model features. Read these before interpreting any app score.

## Next Scientific Improvements

- Fix source/domain bias by collecting full-length popular/charted audio or matching low-download snippets to the same duration/quality rules.
- Evaluate by release year, genre, and artist holdout to quantify generalization.
- Calibrate scores only after the training population matches the intended inference population.
- Store source quality metadata such as bitrate, codec, loudness normalization, and version type.
