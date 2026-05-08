# Features of Popular-Reference Tracks

This report describes patterns in this dataset, not universal rules for music popularity.
Because charted tracks and Internet Archive low-download tracks come from different collection paths, source/domain effects must be treated as a core caveat.

## Dataset Snapshot

- Feature rows: 3975
- Charted rows: 1996
- Low-download rows: 1979
- Feature extraction success rate: 0.9985
- Audio feature source/domain AUC: 0.9488
- Format-only source/domain AUC: 1.0000
- Distance score ROC AUC: 0.8613
- XGBoost OOF ROC AUC: 0.9535

## Feature Profiles

| Feature | Charted median | Charted q25 | Charted q75 | Top 20 median | 21-50 median | 51-100 median | Low-stream median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tempo | 117.454 | 99.384 | 135.999 | 117.454 | 117.454 | 117.454 | 117.454 |
| RMS mean | 0.127 | 0.089 | 0.184 | 0.123 | 0.122 | 0.133 | 0.111 |
| RMS std | 0.043 | 0.03 | 0.062 | 0.043 | 0.042 | 0.044 | 0.051 |
| Chroma mean | 0.375 | 0.337 | 0.419 | 0.374 | 0.373 | 0.376 | 0.406 |
| Chroma std | 0.296 | 0.289 | 0.302 | 0.296 | 0.296 | 0.296 | 0.295 |
| Tonnetz mean | 0.008 | -0.011 | 0.03 | 0.008 | 0.007 | 0.009 | 0.006 |
| Tonnetz std | 0.143 | 0.114 | 0.176 | 0.142 | 0.144 | 0.144 | 0.126 |
| ZCR std | 0.05 | 0.039 | 0.064 | 0.052 | 0.051 | 0.049 | 0.042 |
| Mfcc 11 Std | 7.753 | 6.87 | 8.744 | 7.813 | 7.713 | 7.746 | 7.131 |
| Mfcc 1 Std | 26.847 | 22.287 | 32.843 | 27.632 | 26.67 | 26.343 | 32.851 |
| Mfcc 12 Std | 7.556 | 6.717 | 8.63 | 7.525 | 7.473 | 7.662 | 6.876 |

## Key Distribution

| Key | All tracks | Charted tracks |
| --- | ---: | ---: |
| C major | 322 | 161 |
| F major | 285 | 163 |
| A# major | 266 | 174 |
| D# major | 217 | 133 |
| G# minor | 195 | 87 |
| D# minor | 195 | 70 |
| G# major | 190 | 121 |
| C minor | 185 | 62 |

## Interpretation Caveats

High validation AUC can reflect real audio-pattern signal, source/domain separation, or both. Treat these summaries as descriptive reference statistics in this dataset until the collection design is balanced.

## Data Roadmap

- Expand Internet Archive sampling to cover more genres, years, formats, and download bands.
- Pair archive tracks with additional chart and non-chart references acquired through yt-dlp only where rights, terms, and project policy allow it.
- Track source/domain diagnostics beside model metrics for every dataset refresh.
