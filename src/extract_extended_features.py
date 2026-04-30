import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from music_success_predictor import AudioFeatureExtractor, DEFAULT_PREVIEW_SECONDS
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_artist_title_from_filename(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    if " - " in basename:
        artist, title = basename.split(" - ", 1)
        return artist.strip(), title.strip()
    return "Unknown", basename.strip()


def track_keys(artist, title):
    artist = str(artist).strip()
    title = str(title).strip()
    return (
        f"{artist} - {title}".lower(),
        f"{title} - {artist}".lower(),
    )


def normalize_path(path):
    return os.path.normcase(os.path.abspath(path))


def popularity_tier(peak_rank):
    if pd.isna(peak_rank):
        return "low_stream"
    peak_rank = int(peak_rank)
    if peak_rank <= 20:
        return "top20"
    if peak_rank <= 50:
        return "chart_21_50"
    return "chart_51_100"


def chart_popularity_weight(peak_rank):
    if pd.isna(peak_rank):
        return 0.0
    rank = min(max(float(peak_rank), 1.0), 100.0)
    return float(0.25 + 0.75 * (1 - ((rank - 1) / 99) ** 0.5))


def make_chart_record(path, row):
    peak_rank = row["peak_rank"]
    return {
        "file_path": path,
        "popular": 1,
        "peak_rank": peak_rank,
        "weeks-on-board": row.get("weeks-on-board", np.nan),
        "artist": row["artist"],
        "title": row["title"],
        "label_source": "charted",
        "stream_count": np.nan,
        "popularity_tier": popularity_tier(peak_rank),
        "popularity_weight": chart_popularity_weight(peak_rank),
    }


def load_low_stream_tracks(low_streams_csv, low_stream_threshold, chart_keys):
    if not low_streams_csv or not os.path.exists(low_streams_csv):
        print(f"Low-stream manifest not found: {low_streams_csv}. Skipping this source.")
        return [], set()

    low_df = pd.read_csv(low_streams_csv)
    required_cols = {"file_path", "stream_count"}
    missing_cols = required_cols - set(low_df.columns)
    if missing_cols:
        raise ValueError(f"{low_streams_csv} is missing columns: {sorted(missing_cols)}")

    low_df["stream_count"] = pd.to_numeric(low_df["stream_count"], errors="coerce")
    low_df = low_df[
        low_df["stream_count"].notna() & (low_df["stream_count"] <= low_stream_threshold)
    ]

    low_stream_tracks = []
    low_stream_paths = set()
    skipped_charted = 0
    skipped_missing_file = 0

    for _, row in low_df.iterrows():
        path = str(row["file_path"]).strip()
        if not path or not os.path.exists(path):
            skipped_missing_file += 1
            continue

        normalized = normalize_path(path)
        if normalized in low_stream_paths:
            continue
        low_stream_paths.add(normalized)

        artist, title = parse_artist_title_from_filename(path)
        if "artist" in low_df.columns and pd.notna(row.get("artist")) and str(row.get("artist")).strip():
            artist = str(row["artist"]).strip()
        if "title" in low_df.columns and pd.notna(row.get("title")) and str(row.get("title")).strip():
            title = str(row["title"]).strip()

        if set(track_keys(artist, title)) & chart_keys:
            skipped_charted += 1
            continue

        low_stream_tracks.append(
            {
                "file_path": path,
                "popular": 0,
                "peak_rank": np.nan,
                "weeks-on-board": 0,
                "artist": artist,
                "title": title,
                "label_source": "archive_low_download",
                "stream_count": float(row["stream_count"]),
                "popularity_tier": "low_stream",
                "popularity_weight": 0.0,
            }
        )

    print(
        f"Low-stream <= {low_stream_threshold}: {len(low_stream_tracks)} tracks "
        f"(skipped charted: {skipped_charted}, missing file: {skipped_missing_file})"
    )
    return low_stream_tracks, low_stream_paths


def build_mp3_map():
    all_mp3s = glob.glob("music/*.mp3")
    mp3_map = {}
    for path in all_mp3s:
        basename = os.path.splitext(os.path.basename(path))[0]
        mp3_map[basename.lower()] = path
    return mp3_map


def build_extended_dataset(
    max_chart_tracks=0,
    low_streams_csv="data/raw/low_stream_tracks_sample_2000.csv",
    low_stream_threshold=1000,
    max_low_stream_tracks=0,
    checkpoint_every=100,
    resume=False,
    preview_seconds=DEFAULT_PREVIEW_SECONDS,
):
    print("Loading chart data...")
    df = pd.read_csv("data/processed/tracks_with_features.csv")
    df = df.drop_duplicates(subset=["artist", "title"])

    chart_keys = set()
    for _, row in df.iterrows():
        chart_keys.update(track_keys(row["artist"], row["title"]))

    print(f"Unique charted tracks: {len(df)}")

    print("Looking for chart MP3 files in music/...")
    mp3_map = build_mp3_map()
    print(f"Found {len(mp3_map)} root-level chart MP3 files.")

    chart_records = []
    for _, row in df.iterrows():
        name1, name2 = track_keys(row["artist"], row["title"])
        if name1 in mp3_map:
            chart_records.append(make_chart_record(mp3_map[name1], row))
        elif name2 in mp3_map:
            chart_records.append(make_chart_record(mp3_map[name2], row))

    print(f"Matched charted tracks: {len(chart_records)}")

    low_stream_tracks, _ = load_low_stream_tracks(
        low_streams_csv, low_stream_threshold, chart_keys
    )

    rng = np.random.default_rng(42)
    if max_chart_tracks and len(chart_records) > max_chart_tracks:
        chart_records = [chart_records[i] for i in rng.choice(len(chart_records), max_chart_tracks, replace=False)]
    if max_low_stream_tracks and len(low_stream_tracks) > max_low_stream_tracks:
        low_stream_tracks = [
            low_stream_tracks[i]
            for i in rng.choice(len(low_stream_tracks), max_low_stream_tracks, replace=False)
        ]

    final_sample = chart_records + low_stream_tracks
    if not chart_records or not low_stream_tracks:
        raise RuntimeError(
            "Core dataset needs both matched charted tracks and downloaded low-stream tracks. "
            "Check music/ and the low-stream manifest/download step."
        )

    print(f"Target distribution: {pd.Series([item['popular'] for item in final_sample]).value_counts().to_dict()}")
    print(f"Source distribution: {pd.Series([item['label_source'] for item in final_sample]).value_counts().to_dict()}")
    print(f"Tier distribution: {pd.Series([item['popularity_tier'] for item in final_sample]).value_counts().to_dict()}")
    print(
        f"Extracting features for {len(final_sample)} tracks "
        f"with preview_seconds={preview_seconds}..."
    )

    output_path = "data/processed/extended_features.csv"
    failure_path = "reports/feature_extraction_failures.csv"
    features_list = []
    extracted_paths = set()
    if resume and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        features_list = existing_df.to_dict(orient="records")
        extracted_paths = set(existing_df.get("file_path", pd.Series(dtype=str)).astype(str))
        print(f"Resuming from {len(features_list)} existing extracted rows.")

    extraction_failures = []

    for idx, record in enumerate(tqdm(final_sample), start=1):
        if record["file_path"] in extracted_paths:
            continue

        feat = AudioFeatureExtractor.extract_features(
            record["file_path"],
            preview_seconds=preview_seconds,
        )
        if feat:
            feat.update(record)
            features_list.append(feat)
            extracted_paths.add(record["file_path"])
        else:
            extraction_failures.append(record)

        if checkpoint_every and idx % checkpoint_every == 0:
            pd.DataFrame(features_list).to_csv(output_path, index=False)
            os.makedirs("reports", exist_ok=True)
            pd.DataFrame(extraction_failures).to_csv(failure_path, index=False)
            print(f"Checkpoint: saved {len(features_list)} extracted rows.")

    res_df = pd.DataFrame(features_list)
    res_df.to_csv(output_path, index=False)

    os.makedirs("reports", exist_ok=True)
    failures_df = pd.DataFrame(extraction_failures)
    failures_df.to_csv(failure_path, index=False)

    print(f"Saved {len(res_df)} extracted rows to data/processed/extended_features.csv")
    print(f"Feature extraction failures: {len(failures_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local MP3 audio-feature dataset.")
    parser.add_argument(
        "--max-chart-tracks",
        type=int,
        default=0,
        help="Maximum matched chart tracks to extract. Use 0 for all.",
    )
    parser.add_argument(
        "--low-streams-csv",
        default="data/raw/low_stream_tracks_sample_2000.csv",
        help="CSV with file_path, stream_count and optional artist/title columns for low-stream negatives.",
    )
    parser.add_argument(
        "--low-stream-threshold",
        type=int,
        default=1000,
        help="Maximum stream_count/download proxy for low-stream tracks.",
    )
    parser.add_argument(
        "--max-low-stream-tracks",
        type=int,
        default=0,
        help="Maximum low-stream tracks to extract. Use 0 for all downloaded rows.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Rows between checkpoint writes.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing extended_features.csv.")
    parser.add_argument(
        "--preview-seconds",
        type=int,
        default=DEFAULT_PREVIEW_SECONDS,
        help="Seconds to analyze when extraction mode is preview.",
    )
    args = parser.parse_args()
    build_extended_dataset(
        max_chart_tracks=args.max_chart_tracks,
        low_streams_csv=args.low_streams_csv,
        low_stream_threshold=args.low_stream_threshold,
        max_low_stream_tracks=args.max_low_stream_tracks,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        preview_seconds=args.preview_seconds,
    )
