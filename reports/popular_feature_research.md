# Features of Popular-Reference Tracks

This report describes patterns in this dataset, not universal rules for music popularity.
Because charted tracks and Internet Archive low-download tracks come from different collection paths, source/domain effects must be treated as a core caveat.

## Dataset Snapshot

- Feature rows: 988
- Charted rows: 500
- Low-download rows: 488
- Feature extraction success rate: 0.9980
- Audio feature source/domain AUC: 0.9378
- Format-only source/domain AUC: 0.9982
- Distance score ROC AUC: 0.8623
- XGBoost OOF ROC AUC: 0.9231

## Feature Profiles

| Feature | Charted median | Charted q25 | Charted q75 | Top 20 median | 21-50 median | 51-100 median | Low-stream median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tempo | 117.454 | 99.384 | 135.999 | 117.454 | 112.347 | 117.454 | 117.454 |
| RMS mean | 0.121 | 0.09 | 0.177 | 0.116 | 0.132 | 0.124 | 0.114 |
| RMS std | 0.044 | 0.033 | 0.06 | 0.042 | 0.045 | 0.043 | 0.053 |
| Chroma mean | 0.373 | 0.338 | 0.411 | 0.379 | 0.373 | 0.371 | 0.403 |
| Chroma std | 0.296 | 0.29 | 0.303 | 0.296 | 0.296 | 0.297 | 0.295 |
| Tonnetz mean | 0.006 | -0.01 | 0.029 | 0.009 | 0.003 | 0.008 | 0.007 |
| Tonnetz std | 0.146 | 0.117 | 0.179 | 0.142 | 0.144 | 0.149 | 0.124 |
| ZCR std | 0.05 | 0.037 | 0.066 | 0.049 | 0.052 | 0.048 | 0.04 |
| Mfcc 10 Std | 8.299 | 7.314 | 9.442 | 8.427 | 8.446 | 8.156 | 7.677 |
| Mfcc 1 Std | 27.567 | 22.723 | 33.449 | 27.28 | 28.06 | 27.14 | 32.051 |
| Mfcc 1 Mean | 93.145 | 77.101 | 110.747 | 92.523 | 92.074 | 93.581 | 110.314 |

## Key Distribution

| Key | All tracks | Charted tracks |
| --- | ---: | ---: |
| A# major | 70 | 48 |
| F major | 67 | 46 |
| C major | 63 | 37 |
| G# minor | 59 | 25 |
| G# major | 58 | 34 |
| D# major | 57 | 36 |
| D# minor | 52 | 20 |
| A# minor | 47 | 18 |

## Interpretation Caveats

High validation AUC can reflect real audio-pattern signal, source/domain separation, or both. Treat these summaries as descriptive reference statistics in this dataset until the collection design is balanced.

## Data Roadmap

- Expand Internet Archive sampling to cover more genres, years, formats, and download bands.
- Pair archive tracks with additional chart and non-chart references acquired through yt-dlp only where rights, terms, and project policy allow it.
- Track source/domain diagnostics beside model metrics for every dataset refresh.
