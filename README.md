# Music Popularity

A small research project exploring why some songs become hits by combining historical chart data (US Billboard, UK charts) with track‑level features. The work happens in a single Jupyter notebook that loads, cleans, and normalizes the datasets, then prepares helper files for later enrichment (e.g., Spotify, YouTube).

## What’s Here

- `music-popularity/main.ipynb`: the end‑to‑end workflow (load → clean → normalize → export).
- `music-popularity/charts.csv`: Billboard Hot 100 weekly history (US).
- `music-popularity/top_100_songs_1952_to_2024.xlsx`: UK Top 100 weekly history (1952–2024).
- Outputs written by the notebook (after running it):
  - `billboard_clean.parquet`, `uk_clean.parquet`: cleaned, normalized tables.
  - `songs_for_spotify.csv`: unique (title, artist) pairs for ID lookups.
  - `spotify_cache.json`: optional local cache for API results.

## Setup

1) Use a recent Python (3.10+ recommended). Create and activate a virtual env.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

2) Install the core libraries used by the notebook (the notebook also does `%pip install ...`, but installing up front speeds things up):

```
pip install pandas numpy matplotlib seaborn openpyxl pyarrow
# Optional enrichment:
pip install python-dotenv spotipy yt-dlp youtube-search-python
```

3) Launch Jupyter (or open the notebook directly in VS Code):

```
python -m pip install jupyterlab  # if needed
jupyter lab  # or: jupyter notebook
```

## Running the Notebook

Open `music-popularity/main.ipynb` and run cells top‑to‑bottom:

- Load data: reads `charts.csv` (US) and the UK Excel workbook; prints heads and `.info()`.
- Normalize schema: renames columns, adds `country` and a stable `track_id` derived from normalized title/artist.
- De‑duplicate and clean: drops missing titles/artists and duplicate (title, artist, date) rows.
- Enrich basics: derives `year`, `week`; exports cleaned parquet files and `songs_for_spotify.csv`.

Expected sizes (approx.):
- US (Billboard): ~330k rows, ~20 MB after normalization
- UK: ~286k rows, ~18 MB after normalization

Outputs appear next to the notebook after the export cells execute.

## Optional: Spotify Enrichment

Some cells use the Spotify API (via `spotipy`) to look up track IDs/features. To enable:

1) Create a free Spotify developer app and get `Client ID` and `Client Secret`.
2) Add a `.env` file next to the notebook with:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

Notes:
- Network access is required for these cells. If you’re offline or in a restricted environment, skip them.
- The notebook uses a simple JSON cache (`spotify_cache.json`) to avoid repeated lookups.

## Optional: YouTube Search / Audio

Later cells install `yt-dlp` and `youtube-search-python` to locate reference audio. These are purely optional and require outbound network access. If you see connection timeouts, either rerun later with network enabled or skip those cells.

## Troubleshooting

- Datetime errors (e.g., “Can only use .dt accessor with datetimelike values”): ensure `date` columns are parsed
  correctly, for example: `df["date"] = pd.to_datetime(df["date"], errors="coerce")` before using `.dt`.
- Memory: if you’re memory‑constrained, run on a 64‑bit Python and consider reading in chunks or filtering years.
- Parquet export: `pyarrow` is required; install with `pip install pyarrow` if missing.

## Project Structure

```
music-popularity/
├─ main.ipynb
├─ charts.csv
├─ top_100_songs_1952_to_2024.xlsx
└─ README.md
```

The notebook also writes: `billboard_clean.parquet`, `uk_clean.parquet`, `songs_for_spotify.csv`, and optionally `spotify_cache.json`.

## Roadmap

- Clean and standardize historical US/UK charts ✓
- Prepare unique track list for external ID lookups ✓
- Fetch audio features via APIs or offline extraction
- Analyze relationships between features and chart success
- Publish a video/write‑up of findings
- Train a model to get the "popularity probability" of your music

## Notes on Data & Attribution

The repository includes pre‑extracted chart data files for convenience. If you publish results, please credit the original chart sources and any APIs used (Spotify, YouTube) and review their terms of service.

