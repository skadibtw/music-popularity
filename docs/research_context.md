# Research Context

## Goal

Build a research-first audio-feature popularity project. The main output is a `hit potential score`: a percentile-style score from 0 to 1 showing how close a track's audio features are to a popular/charted reference distribution. It is not a guarantee of commercial success.

## Current Direction

- Research and ML quality come before portfolio presentation.
- Study how audio features relate to popularity, with correlation/EDA as a consequence of building a better prediction/reference dataset.
- Keep the project honest about limitations: marketing, artist reputation, release timing, platforms, playlists, and cultural context are not modeled.

## Label Semantics

- All charted tracks are treated as `popular = 1`, including tracks with `peak_rank = 100`.
- Low-stream / low-download tracks are treated as `popular = 0`.
- `local_unmatched_proxy` is excluded from the core dataset because it is too noisy as a negative label.
- Keep source labels for diagnostics and source-bias checks.

## Popularity Strength

Use a smooth chart weight for charted tracks:

```text
popularity_weight = 0.25 + 0.75 * (1 - ((peak_rank - 1) / 99) ** 0.5)
```

- Rank 1 has weight 1.0.
- Rank 100 has weight 0.25, still positive/popular.
- Low-stream tracks have weight 0.0.
- Use this weight for the distance/reference score first, not as the main supervised target.

## Current Extraction Protocol

- Current dataset extraction uses `--preview-seconds 30`.
- The fixed 30-second window keeps analyzed duration consistent across charted and low-download sources.
- Raw full-file duration is retained as metadata for QC, but it is excluded from model features and source-bias audio diagnostics.
- Model training excludes audio features with univariate source-separability abs ROC-AUC >= 0.65.

## Main Score

- Main score: distance-to-weighted-chart-reference percentile.
- XGBoost remains a secondary diagnostic, not the headline score.
- The app should be updated later, after offline reports stabilize.

## Low-Stream Data Plan

- Audio files are never committed.
- Manifests and scripts can be committed if the CSV size is reasonable.
- Preserve the current licensed manifest separately.
- Build a broader 50,000-row Internet Archive `collection:netlabels` pool with missing-license rows allowed.
- Create a reproducible staged sample of 2,000 rows using fixed-seed randomized sampling.
- Sampling constraints:
  - max 10 tracks per `archive_item`;
  - max 20 tracks per artist;
  - max 50 tracks for `Unknown` artist.

## First 2,000-Track Iteration Results

- Download success rate: 99.05%.
- Feature extraction success rate: 99.85%.
- Median analyzed duration: 30.0 seconds.
- No archive item has more than 10 tracks in the staged sample.
- No artist has more than 20 tracks, except `Unknown` <= 50.
- Distance score ROC-AUC: 0.8847 on the current charted-vs-low-download proxy task.
- Conservative distance score ROC-AUC after source-confounded feature exclusion: 0.8561.
- Conservative XGBoost out-of-fold ROC-AUC: 0.9524.
- Audio-feature source separability remains high at 0.9493 ROC-AUC; model-feature source separability is 0.9344 after exclusion, so popularity claims remain source/domain-confounded.

## Cleanup Rule

During migration, delete functions, scripts, artifacts, or files that only support stale `success = peak_rank <= 20` semantics, if they are no longer used. Do not delete raw/source data, manifest snapshots, useful diagnostics, or baseline code that is explicitly still reported.
