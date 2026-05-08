import argparse
import csv
import os
import shutil


def normalize_file_path(path):
    return os.path.normpath(str(path).replace("\\", os.sep))


def normalize_manifest(input_path, output_path=None, move_files=False):
    output_path = output_path or input_path
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    moved = 0
    existing = 0
    missing = 0
    changed = 0
    for row in rows:
        original = (row.get("file_path") or "").strip()
        normalized = normalize_file_path(original)
        if original != normalized:
            changed += 1
        row["file_path"] = normalized

        if not move_files:
            continue

        if os.path.exists(normalized):
            existing += 1
            continue
        if os.path.exists(original):
            os.makedirs(os.path.dirname(normalized) or ".", exist_ok=True)
            shutil.move(original, normalized)
            moved += 1
        else:
            missing += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Rows: {len(rows)}. Paths normalized: {changed}. "
        f"Moved files: {moved}. Existing files: {existing}. Missing files: {missing}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize low-stream manifest file paths for the current OS.")
    parser.add_argument("--input", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", default=None, help="Output manifest CSV. Defaults to overwriting input.")
    parser.add_argument("--move-files", action="store_true", help="Move existing files from old paths to normalized paths.")
    args = parser.parse_args()
    normalize_manifest(args.input, args.output, args.move_files)
