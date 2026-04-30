import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_yt_dlp(yt_dlp):
    if os.path.exists(yt_dlp):
        return yt_dlp

    found = shutil.which(yt_dlp)
    if found:
        return found

    in_cwd = os.path.join(os.getcwd(), yt_dlp)
    if os.path.exists(in_cwd):
        return in_cwd

    script_name = "yt-dlp.exe" if os.name == "nt" else "yt-dlp"
    next_to_python = os.path.join(os.path.dirname(sys.executable), script_name)
    if os.path.exists(next_to_python):
        return next_to_python

    return None


def run_download(row, yt_dlp_path, dry_run, yt_dlp):
    source_url = (row.get("source_url") or "").strip()
    file_path = (row.get("file_path") or "").strip()
    if not source_url or not file_path:
        return "skipped", row

    if os.path.exists(file_path):
        print(f"skip existing: {file_path}")
        return "ready", row

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    cmd = [
        yt_dlp_path or yt_dlp,
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--no-part",
        "-o",
        file_path,
        source_url,
    ]
    print("download:", file_path)
    if dry_run:
        print(" ".join(cmd))
        return "ready", row

    result = subprocess.run(cmd, check=False)
    if result.returncode == 0 and os.path.exists(file_path):
        return "ready", row
    return "failed", row


def write_failures(failed_rows, failure_csv):
    if not failure_csv:
        return
    os.makedirs(os.path.dirname(failure_csv) or ".", exist_ok=True)
    fieldnames = sorted({key for row in failed_rows for key in row.keys()})
    with open(failure_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)


def download_low_stream_tracks(
    manifest_path,
    limit=None,
    dry_run=False,
    yt_dlp="yt-dlp",
    workers=1,
    failure_csv="reports/low_stream_download_failures.csv",
):
    yt_dlp_path = resolve_yt_dlp(yt_dlp)
    if not dry_run and yt_dlp_path is None:
        raise RuntimeError(
            f"{yt_dlp} не найден. Установите его: python -m pip install yt-dlp"
        )

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit is not None:
        rows = rows[:limit]

    downloaded = 0
    failed_rows = []
    skipped = 0
    if workers <= 1:
        for row in rows:
            status, result_row = run_download(row, yt_dlp_path, dry_run, yt_dlp)
            if status == "ready":
                downloaded += 1
            elif status == "failed":
                failed_rows.append(result_row)
            else:
                skipped += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_download, row, yt_dlp_path, dry_run, yt_dlp) for row in rows]
            for future in as_completed(futures):
                status, result_row = future.result()
                if status == "ready":
                    downloaded += 1
                elif status == "failed":
                    failed_rows.append(result_row)
                else:
                    skipped += 1

    write_failures(failed_rows, failure_csv)
    print(f"Done. Ready or existing files: {downloaded}. Failed: {len(failed_rows)}. Skipped: {skipped}.")
    return len(failed_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download low-stream tracks from manifest via yt-dlp.")
    parser.add_argument(
        "--manifest",
        default="data/raw/low_stream_tracks.csv",
        help="CSV with file_path and source_url columns.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading.")
    parser.add_argument("--yt-dlp", default="yt-dlp", help="yt-dlp executable name or path.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent download workers.")
    parser.add_argument(
        "--failure-csv",
        default="reports/low_stream_download_failures.csv",
        help="CSV path for failed download rows.",
    )
    args = parser.parse_args()

    try:
        failures = download_low_stream_tracks(
            args.manifest,
            limit=args.limit,
            dry_run=args.dry_run,
            yt_dlp=args.yt_dlp,
            workers=args.workers,
            failure_csv=args.failure_csv,
        )
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    sys.exit(1 if failures else 0)
