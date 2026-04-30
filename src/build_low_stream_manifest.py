import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


ADVANCED_SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
USER_AGENT = "music-popularity-dataset-builder/1.0"
FIELDNAMES = [
    "file_path",
    "artist",
    "title",
    "stream_count",
    "source_url",
    "archive_item",
    "license_url",
]


def fetch_json(url, retries=3, sleep_seconds=1.0):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def safe_filename_part(value, fallback):
    value = str(value or fallback).strip()
    value = re.sub(r"[<>:\"/\\|?*]+", "-", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:120] or fallback


def build_search_url(query, page, rows_per_page):
    params = [
        ("q", query),
        ("fl[]", "identifier"),
        ("fl[]", "title"),
        ("fl[]", "creator"),
        ("fl[]", "downloads"),
        ("fl[]", "licenseurl"),
        ("sort[]", "downloads desc"),
        ("rows", str(rows_per_page)),
        ("page", str(page)),
        ("output", "json"),
    ]
    return f"{ADVANCED_SEARCH_URL}?{urllib.parse.urlencode(params)}"


def first_text(value, fallback=""):
    if isinstance(value, list):
        return str(value[0]) if value else fallback
    return str(value) if value not in (None, "") else fallback


def is_audio_file(file_record):
    name = str(file_record.get("name") or "")
    fmt = str(file_record.get("format") or "").lower()
    return name.lower().endswith(".mp3") or "mp3" in fmt


def make_row(item_doc, file_record, item_metadata):
    identifier = item_doc["identifier"]
    item_title = first_text(item_doc.get("title"), identifier)
    creator = first_text(item_doc.get("creator"), first_text(item_metadata.get("creator"), "Unknown"))
    license_url = first_text(item_doc.get("licenseurl"), first_text(item_metadata.get("licenseurl"), ""))
    downloads = int(float(item_doc.get("downloads") or 0))

    source_name = str(file_record.get("name") or "")
    title = os.path.splitext(os.path.basename(source_name))[0] or item_title
    artist_part = safe_filename_part(creator, "Unknown")
    title_part = safe_filename_part(title, "Untitled")
    filename = f"{artist_part} - {title_part}.mp3"

    return {
        "file_path": os.path.join("music", "low_stream", filename),
        "artist": creator,
        "title": title,
        "stream_count": downloads,
        "source_url": f"https://archive.org/download/{identifier}/{urllib.parse.quote(source_name)}",
        "archive_item": f"https://archive.org/details/{identifier}",
        "license_url": license_url,
    }


def build_item_rows(item_doc, allow_missing_license):
    identifier = item_doc["identifier"]
    try:
        metadata = fetch_json(METADATA_URL.format(identifier=urllib.parse.quote(identifier)))
    except RuntimeError as error:
        print(error, flush=True)
        return []

    rows = []
    item_metadata = metadata.get("metadata", {})
    for file_record in metadata.get("files", []):
        if not is_audio_file(file_record):
            continue

        row = make_row(item_doc, file_record, item_metadata)
        if not allow_missing_license and not row["license_url"]:
            continue
        rows.append(row)
    return rows


def write_manifest(rows, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_manifest(output_path):
    if not os.path.exists(output_path):
        return []
    with open(output_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_manifest(
    target_rows,
    output_path,
    base_query,
    max_downloads,
    rows_per_page,
    max_pages,
    allow_missing_license,
    workers,
    checkpoint_every,
    resume,
    start_page,
):
    rows = load_existing_manifest(output_path) if resume else []
    seen_urls = {row["source_url"] for row in rows if row.get("source_url")}
    seen_paths = {row["file_path"].lower() for row in rows if row.get("file_path")}
    seen_items = set()
    last_checkpoint_count = len(rows)
    query = f"{base_query} AND downloads:[0 TO {max_downloads}]"

    if rows:
        print(f"Resuming from {len(rows)} existing rows in {output_path}", flush=True)

    for page in range(start_page, max_pages + 1):
        if len(rows) >= target_rows:
            break

        data = fetch_json(build_search_url(query, page, rows_per_page))
        docs = []
        for doc in data.get("response", {}).get("docs", []):
            identifier = doc.get("identifier")
            if identifier and identifier not in seen_items:
                seen_items.add(identifier)
                docs.append(doc)
        if not docs:
            break

        print(
            f"Page {page}: scanning {len(docs)} items; collected {len(rows)} rows...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(build_item_rows, doc, allow_missing_license) for doc in docs]
            for future in as_completed(futures):
                for row in future.result():
                    if len(rows) >= target_rows:
                        break
                    if row["source_url"] in seen_urls:
                        continue

                    base_path = row["file_path"]
                    path = base_path
                    suffix = 2
                    while path.lower() in seen_paths:
                        stem, ext = os.path.splitext(base_path)
                        path = f"{stem} ({suffix}){ext}"
                        suffix += 1

                    row["file_path"] = path
                    seen_urls.add(row["source_url"])
                    seen_paths.add(path.lower())
                    rows.append(row)

                if len(rows) >= target_rows:
                    break

        if len(rows) - last_checkpoint_count >= checkpoint_every:
            write_manifest(rows, output_path)
            last_checkpoint_count = len(rows)
            print(f"Checkpoint: wrote {len(rows)} rows to {output_path}", flush=True)

    write_manifest(rows, output_path)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a low-stream MP3 manifest from Internet Archive metadata.")
    parser.add_argument("--target-rows", type=int, default=10000, help="Target number of MP3 rows.")
    parser.add_argument("--output", default="data/raw/low_stream_tracks.csv", help="Output CSV path.")
    parser.add_argument(
        "--base-query",
        default="mediatype:audio AND collection:netlabels",
        help="Internet Archive query before the downloads filter.",
    )
    parser.add_argument("--max-downloads", type=int, default=1000, help="Maximum item-level downloads proxy.")
    parser.add_argument("--rows-per-page", type=int, default=500, help="Internet Archive search page size.")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum search pages to scan.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent metadata fetch workers.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Rows between checkpoint writes.")
    parser.add_argument("--resume", action="store_true", help="Seed rows and duplicate checks from existing output CSV.")
    parser.add_argument("--start-page", type=int, default=1, help="First Internet Archive search page to scan.")
    parser.add_argument(
        "--allow-missing-license",
        action="store_true",
        help="Keep rows without Internet Archive license metadata.",
    )
    args = parser.parse_args()

    count = collect_manifest(
        target_rows=args.target_rows,
        output_path=args.output,
        base_query=args.base_query,
        max_downloads=args.max_downloads,
        rows_per_page=args.rows_per_page,
        max_pages=args.max_pages,
        allow_missing_license=args.allow_missing_license,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        start_page=args.start_page,
    )
    raise SystemExit(0 if count >= args.target_rows else 1)
