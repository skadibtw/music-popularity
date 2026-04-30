import argparse
from collections import Counter

import pandas as pd


def normalize_artist(value):
    if pd.isna(value) or not str(value).strip():
        return "Unknown"
    return str(value).strip()


def sample_manifest(
    input_path,
    output_path,
    target_rows,
    seed,
    max_per_archive_item,
    max_per_artist,
    max_unknown_artist,
):
    df = pd.read_csv(input_path)
    required_cols = {"file_path", "artist", "source_url", "archive_item"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = df.drop_duplicates(subset=["source_url"]).copy()
    df["artist"] = df["artist"].apply(normalize_artist)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    item_counts = Counter()
    artist_counts = Counter()
    selected_indices = []

    for idx, row in df.iterrows():
        archive_item = row["archive_item"]
        artist = row["artist"]
        artist_limit = max_unknown_artist if artist == "Unknown" else max_per_artist

        if item_counts[archive_item] >= max_per_archive_item:
            continue
        if artist_counts[artist] >= artist_limit:
            continue

        selected_indices.append(idx)
        item_counts[archive_item] += 1
        artist_counts[artist] += 1

        if len(selected_indices) >= target_rows:
            break

    sample = df.loc[selected_indices].copy()
    sample.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Selected rows: {len(sample)}")
    print(f"Unique source_url: {sample['source_url'].nunique()}")
    print(f"Unique archive_item: {sample['archive_item'].nunique()}")
    print(f"Unique artist: {sample['artist'].nunique()}")
    print(f"Max per archive_item: {sample['archive_item'].value_counts().max()}")
    print(f"Max per artist: {sample['artist'].value_counts().max()}")
    print(f"Unknown artist rows: {int((sample['artist'] == 'Unknown').sum())}")

    return len(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a capped staged sample from a low-stream manifest.")
    parser.add_argument("--input", default="data/raw/low_stream_tracks.csv", help="Input low-stream manifest.")
    parser.add_argument(
        "--output",
        default="data/raw/low_stream_tracks_sample_2000.csv",
        help="Output sampled manifest.",
    )
    parser.add_argument("--target-rows", type=int, default=2000, help="Target sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--max-per-archive-item", type=int, default=10, help="Maximum rows per archive item.")
    parser.add_argument("--max-per-artist", type=int, default=20, help="Maximum rows per artist.")
    parser.add_argument("--max-unknown-artist", type=int, default=50, help="Maximum rows for Unknown artist.")
    args = parser.parse_args()

    count = sample_manifest(
        input_path=args.input,
        output_path=args.output,
        target_rows=args.target_rows,
        seed=args.seed,
        max_per_archive_item=args.max_per_archive_item,
        max_per_artist=args.max_per_artist,
        max_unknown_artist=args.max_unknown_artist,
    )
    raise SystemExit(0 if count >= args.target_rows else 1)
