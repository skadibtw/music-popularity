# Torchaudio Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `torchaudio` feature extraction backend that can batch audio feature computation on AMD ROCm/Linux while preserving the existing `librosa` path as the compatibility baseline.

**Architecture:** Keep the current single-track `librosa` extractor as the app/default compatibility path, then add a focused torch backend for dataset generation. The torch backend decodes audio on CPU, pads fixed 30-second previews into batches, moves tensors to `cuda` when available, computes spectral/RMS/ZCR/MFCC/chroma-like features with PyTorch/torchaudio, and falls back to CPU for non-GPU environments.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, ROCm wheels on Linux/AMD, pandas, numpy, librosa parity tests, unittest.

---

## Current Constraints

- `src/music_success_predictor.py` currently uses `librosa` and processes one file at a time.
- `src/extract_extended_features.py` owns dataset assembly, checkpointing, and failure CSVs.
- `requirements.txt` is UTF-16 LE and currently contains Windows ROCm wheel URLs. We should not blindly rewrite it during the first implementation step.
- PyTorch wheel index currently exposes Linux ROCm 7.2 wheels for `torch`, `torchvision`, and `torchaudio`; the preferred project target is the 2.11 family via `--index-url https://download.pytorch.org/whl/rocm7.2`.
- `torchaudio.transforms` are `torch.nn.Module` transforms and can be moved to CUDA/ROCm with `.to(device)`.

## File Structure

- Create `src/torch_audio_features.py`
  - Owns torch/torchaudio feature extraction only.
  - Exposes `TorchAudioFeatureExtractor`, `extract_batch_features`, and a single-file convenience wrapper.
  - Does not import app/model code.

- Modify `src/extract_extended_features.py`
  - Add CLI flags: `--backend`, `--device`, `--batch-size`, `--num-decode-workers`.
  - Keep existing `librosa` code path unchanged for default behavior.
  - Add batched torch path with the same output schema.

- Modify `src/music_success_predictor.py`
  - Keep `AudioFeatureExtractor` as the compatibility wrapper.
  - Optionally allow `backend="torchaudio"` for single-file app smoke usage only after parity tests pass.

- Create `tests/test_torch_audio_features.py`
  - Unit tests on synthetic waveforms, no external audio files required.
  - Tests schema, CPU fallback, duration handling, and key format.

- Create `scripts/check_torch_audio_env.py`
  - Prints torch/torchaudio versions, CUDA/ROCm availability, selected device, and a small transform smoke result.

- Modify `README.md`
  - Add Linux AMD ROCm install instructions and CPU fallback instructions.

- Optionally modify `requirements.txt`
  - Add normal PyPI package names `torch` and `torchaudio` only if we decide to normalize this file encoding first.
  - Keep ROCm index installation documented separately because pip requirement files cannot reliably express “use this index only for torch wheels” without making the whole environment fragile.

---

### Task 1: Torch/Torchaudio Environment Probe

**Files:**
- Create: `scripts/check_torch_audio_env.py`
- Test: manual command

- [ ] **Step 1: Add the environment probe script**

Create `scripts/check_torch_audio_env.py`:

```python
import json

import torch
import torchaudio


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform = torch.randn(2, 22050, device=device)
    transform = torchaudio.transforms.MFCC(
        sample_rate=22050,
        n_mfcc=13,
        melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
    ).to(device)
    mfcc = transform(waveform)
    payload = {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mfcc_shape": list(mfcc.shape),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run with normal PyPI/CPU packages first**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 --with torch --with torchaudio python scripts/check_torch_audio_env.py
```

Expected: JSON prints `torch`, `torchaudio`, `device`, and `mfcc_shape`.

- [ ] **Step 3: Run with ROCm wheels on Linux AMD**

Use the project target:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  --with torch --with torchvision --with torchaudio \
  python scripts/check_torch_audio_env.py
```

Fallback if ROCm 7.2 wheels or local driver are incompatible:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 \
  --index-url https://download.pytorch.org/whl/rocm6.4 \
  --with torch==2.9.1 --with torchvision==0.24.1 --with torchaudio==2.9.1 \
  python scripts/check_torch_audio_env.py
```

Expected on AMD ROCm machine: `cuda_available: true`. PyTorch exposes ROCm through the `cuda` API.

- [ ] **Step 4: Commit**

```bash
git add scripts/check_torch_audio_env.py
git commit -m "chore: add torchaudio environment probe"
```

---

### Task 2: Torch Feature Extractor Unit

**Files:**
- Create: `src/torch_audio_features.py`
- Test: `tests/test_torch_audio_features.py`

- [ ] **Step 1: Write tests for torch feature schema**

Create `tests/test_torch_audio_features.py`:

```python
import unittest

import torch

from src.torch_audio_features import TorchAudioFeatureExtractor


class TorchAudioFeatureExtractorTest(unittest.TestCase):
    def test_extract_waveform_batch_returns_current_schema(self):
        sr = 22050
        waveform = torch.sin(2 * torch.pi * 440 * torch.arange(sr * 2) / sr)
        batch = waveform.unsqueeze(0)

        extractor = TorchAudioFeatureExtractor(sample_rate=sr, device="cpu")
        rows = extractor.extract_waveform_batch(batch, [2.0])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        expected = {
            "duration_seconds",
            "analyzed_duration_seconds",
            "tempo",
            "key",
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_rolloff_mean",
            "spectral_bandwidth_mean",
            "zcr_mean",
            "zcr_std",
            "chroma_mean",
            "chroma_std",
            "rms_mean",
            "rms_std",
            "tonnetz_mean",
            "tonnetz_std",
        }
        for i in range(13):
            expected.add(f"mfcc_{i}_mean")
            expected.add(f"mfcc_{i}_std")

        self.assertTrue(expected.issubset(row.keys()))
        self.assertEqual(row["feature_backend"], "torchaudio")
        self.assertGreater(row["rms_mean"], 0)
        self.assertRegex(row["key"], r"^[A-G]#? (major|minor)$")

    def test_short_waveform_is_handled(self):
        extractor = TorchAudioFeatureExtractor(sample_rate=22050, device="cpu")
        rows = extractor.extract_waveform_batch(torch.zeros(1, 1000), [1000 / 22050])

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["analyzed_duration_seconds"], 1000 / 22050, places=4)
        self.assertEqual(rows[0]["feature_backend"], "torchaudio")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity \
  uv run --python 3.12 --with torch --with torchaudio python -m unittest tests.test_torch_audio_features -v
```

Expected: FAIL because `src.torch_audio_features` does not exist.

- [ ] **Step 3: Implement torch feature extractor**

Create `src/torch_audio_features.py` with:

```python
import math

import torch
import torchaudio


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class TorchAudioFeatureExtractor:
    def __init__(
        self,
        sample_rate=22050,
        device="auto",
        n_fft=2048,
        hop_length=512,
        n_mfcc=13,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = int(sample_rate)
        self.device = torch.device(device)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mfcc = int(n_mfcc)

        self.window = torch.hann_window(self.n_fft, device=self.device)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": 128,
                "center": True,
                "power": 2.0,
            },
        ).to(self.device)

    def extract_waveform_batch(self, waveforms, analyzed_durations):
        waveforms = waveforms.to(self.device, dtype=torch.float32)
        if waveforms.ndim == 3:
            waveforms = waveforms.mean(dim=1)
        if waveforms.ndim != 2:
            raise ValueError("Expected waveform tensor with shape [batch, samples] or [batch, channels, samples].")

        spec_complex = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        magnitude = spec_complex.abs().clamp_min(1e-10)
        power = magnitude.square()
        freqs = torch.linspace(0, self.sample_rate / 2, magnitude.shape[1], device=self.device)

        centroid = (freqs[None, :, None] * magnitude).sum(dim=1) / magnitude.sum(dim=1).clamp_min(1e-10)
        bandwidth = torch.sqrt(
            (((freqs[None, :, None] - centroid[:, None, :]).abs() ** 2) * magnitude).sum(dim=1)
            / magnitude.sum(dim=1).clamp_min(1e-10)
        )
        cumulative = torch.cumsum(power, dim=1)
        thresholds = 0.85 * cumulative[:, -1:, :]
        rolloff_bins = (cumulative >= thresholds).float().argmax(dim=1)
        rolloff = freqs[rolloff_bins]

        zcr = ((waveforms[:, 1:] * waveforms[:, :-1]) < 0).float()
        rms_frames = torch.sqrt(
            torch.nn.functional.avg_pool1d(
                waveforms.square().unsqueeze(1),
                kernel_size=self.n_fft,
                stride=self.hop_length,
                padding=self.n_fft // 2,
            ).squeeze(1).clamp_min(1e-10)
        )
        mfcc = self.mfcc(waveforms)
        chroma = self._chroma_from_spectrogram(magnitude, freqs)

        rows = []
        for i in range(waveforms.shape[0]):
            chroma_i = chroma[i]
            row = {
                "feature_backend": "torchaudio",
                "duration_seconds": float(analyzed_durations[i]),
                "analyzed_duration_seconds": float(analyzed_durations[i]),
                "tempo": 0.0,
                "key": self._estimate_key(chroma_i),
                "spectral_centroid_mean": float(centroid[i].mean().detach().cpu()),
                "spectral_centroid_std": float(centroid[i].std(unbiased=False).detach().cpu()),
                "spectral_rolloff_mean": float(rolloff[i].mean().detach().cpu()),
                "spectral_bandwidth_mean": float(bandwidth[i].mean().detach().cpu()),
                "zcr_mean": float(zcr[i].mean().detach().cpu()) if zcr.shape[1] else 0.0,
                "zcr_std": float(zcr[i].std(unbiased=False).detach().cpu()) if zcr.shape[1] else 0.0,
                "chroma_mean": float(chroma_i.mean().detach().cpu()),
                "chroma_std": float(chroma_i.std(unbiased=False).detach().cpu()),
                "rms_mean": float(rms_frames[i].mean().detach().cpu()),
                "rms_std": float(rms_frames[i].std(unbiased=False).detach().cpu()),
                "tonnetz_mean": 0.0,
                "tonnetz_std": 0.0,
            }
            for j in range(self.n_mfcc):
                row[f"mfcc_{j}_mean"] = float(mfcc[i, j].mean().detach().cpu())
                row[f"mfcc_{j}_std"] = float(mfcc[i, j].std(unbiased=False).detach().cpu())
            rows.append(row)
        return rows

    def _chroma_from_spectrogram(self, magnitude, freqs):
        chroma = torch.zeros(
            magnitude.shape[0],
            12,
            magnitude.shape[2],
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        valid = freqs > 0
        midi = torch.round(69 + 12 * torch.log2(freqs[valid] / 440.0)).long()
        pitch_class = torch.remainder(midi, 12)
        chroma.index_add_(1, pitch_class, magnitude[:, valid, :])
        return chroma / chroma.sum(dim=1, keepdim=True).clamp_min(1e-10)

    def _estimate_key(self, chroma):
        pitch_profile = chroma.mean(dim=1).detach().cpu()
        pitch_profile = pitch_profile / pitch_profile.sum().clamp_min(1e-10)
        major = MAJOR_PROFILE / MAJOR_PROFILE.sum()
        minor = MINOR_PROFILE / MINOR_PROFILE.sum()

        def score(profile, shift):
            rolled = torch.roll(pitch_profile, shifts=shift)
            x = profile - profile.mean()
            y = rolled - rolled.mean()
            return float((x * y).sum() / (torch.sqrt((x.square()).sum() * (y.square()).sum()) + 1e-10))

        major_scores = [score(major, i) for i in range(12)]
        minor_scores = [score(minor, i) for i in range(12)]
        major_idx = int(max(range(12), key=lambda i: major_scores[i]))
        minor_idx = int(max(range(12), key=lambda i: minor_scores[i]))
        if major_scores[major_idx] >= minor_scores[minor_idx]:
            return f"{NOTES[major_idx]} major"
        return f"{NOTES[minor_idx]} minor"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity \
  uv run --python 3.12 --with torch --with torchaudio python -m unittest tests.test_torch_audio_features -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/torch_audio_features.py tests/test_torch_audio_features.py
git commit -m "feat: add torchaudio feature extractor"
```

---

### Task 3: Batched Dataset Extraction Backend

**Files:**
- Modify: `src/extract_extended_features.py`
- Test: `tests/test_torch_audio_features.py`

- [ ] **Step 1: Add a waveform loader helper**

Add to `src/torch_audio_features.py`:

```python
def load_audio_preview(path, sample_rate=22050, preview_seconds=30):
    waveform, source_sr = torchaudio.load(path)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {path}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    max_samples = int(sample_rate * preview_seconds)
    waveform = waveform[:, :max_samples]
    analyzed_duration = waveform.shape[-1] / sample_rate
    if waveform.shape[-1] < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[-1]))
    return waveform.squeeze(0), float(analyzed_duration)
```

- [ ] **Step 2: Add batched torch extraction to `extract_extended_features.py`**

Add imports:

```python
from torch_audio_features import TorchAudioFeatureExtractor, load_audio_preview
```

Add function:

```python
def extract_records_torchaudio(records, preview_seconds, batch_size, device, checkpoint_every, output_path, failure_path, resume_rows=None):
    import torch

    extractor = TorchAudioFeatureExtractor(device=device)
    features_list = list(resume_rows or [])
    extracted_paths = {row["file_path"] for row in features_list if "file_path" in row}
    extraction_failures = []
    pending_waveforms = []
    pending_durations = []
    pending_records = []

    def flush_batch():
        if not pending_records:
            return
        batch = torch.stack(pending_waveforms, dim=0)
        rows = extractor.extract_waveform_batch(batch, pending_durations)
        for feat, record in zip(rows, pending_records):
            feat.update(record)
            features_list.append(feat)
            extracted_paths.add(record["file_path"])
        pending_waveforms.clear()
        pending_durations.clear()
        pending_records.clear()

    for idx, record in enumerate(tqdm(records), start=1):
        if record["file_path"] in extracted_paths:
            continue
        try:
            waveform, analyzed_duration = load_audio_preview(record["file_path"], preview_seconds=preview_seconds)
            pending_waveforms.append(waveform)
            pending_durations.append(analyzed_duration)
            pending_records.append(record)
            if len(pending_records) >= batch_size:
                flush_batch()
        except Exception as exc:
            failed = dict(record)
            failed["failure_reason"] = str(exc)
            extraction_failures.append(failed)

        if checkpoint_every and idx % checkpoint_every == 0:
            flush_batch()
            pd.DataFrame(features_list).to_csv(output_path, index=False)
            os.makedirs("reports", exist_ok=True)
            pd.DataFrame(extraction_failures).to_csv(failure_path, index=False)
            print(f"Checkpoint: saved {len(features_list)} extracted rows.")

    flush_batch()
    return features_list, extraction_failures
```

- [ ] **Step 3: Wire CLI arguments**

Add parser args:

```python
parser.add_argument("--backend", choices=["librosa", "torchaudio"], default="librosa")
parser.add_argument("--device", default="auto", help="Torch device for torchaudio backend: auto, cpu, cuda.")
parser.add_argument("--batch-size", type=int, default=16, help="Torch extraction batch size.")
```

Pass them into `build_extended_dataset(...)` and branch after `final_sample` is built:

```python
if backend == "torchaudio":
    features_list, extraction_failures = extract_records_torchaudio(
        final_sample,
        preview_seconds=preview_seconds,
        batch_size=batch_size,
        device=device,
        checkpoint_every=checkpoint_every,
        output_path=output_path,
        failure_path=failure_path,
        resume_rows=features_list if resume else None,
    )
else:
    # existing librosa loop
```

- [ ] **Step 4: Run a tiny CPU smoke extraction**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity/src \
  uv run --python 3.12 --with pandas --with numpy --with torch --with torchaudio --with tqdm \
  python src/extract_extended_features.py \
    --backend torchaudio \
    --device cpu \
    --batch-size 2 \
    --low-streams-csv data/raw/low_stream_tracks_sample_500.csv \
    --max-chart-tracks 2 \
    --max-low-stream-tracks 2 \
    --checkpoint-every 2 \
    --preview-seconds 5
```

Expected: output CSV has 4 rows and `feature_backend=torchaudio`.

- [ ] **Step 5: Commit**

```bash
git add src/extract_extended_features.py src/torch_audio_features.py
git commit -m "feat: add batched torchaudio dataset extraction"
```

---

### Task 4: Parity and Benchmark Report

**Files:**
- Create: `src/benchmark_audio_backends.py`
- Create: `reports/audio_backend_benchmark.md`

- [ ] **Step 1: Add benchmark script**

Create `src/benchmark_audio_backends.py`:

```python
import argparse
import glob
import time

import pandas as pd

from music_success_predictor import AudioFeatureExtractor
from torch_audio_features import TorchAudioFeatureExtractor, load_audio_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-seconds", type=int, default=30)
    parser.add_argument("--output", default="reports/audio_backend_benchmark.md")
    args = parser.parse_args()

    paths = glob.glob("music/*.mp3")[: args.limit]
    torch_extractor = TorchAudioFeatureExtractor(device=args.device)

    start = time.perf_counter()
    librosa_rows = [AudioFeatureExtractor.extract_features(path, preview_seconds=args.preview_seconds) for path in paths]
    librosa_seconds = time.perf_counter() - start

    start = time.perf_counter()
    torch_rows = []
    for path in paths:
        waveform, duration = load_audio_preview(path, preview_seconds=args.preview_seconds)
        torch_rows.extend(torch_extractor.extract_waveform_batch(waveform.unsqueeze(0), [duration]))
    torch_seconds = time.perf_counter() - start

    comparison_features = ["rms_mean", "zcr_mean", "spectral_centroid_mean", "mfcc_0_mean", "mfcc_1_mean", "chroma_mean"]
    rows = []
    for feature in comparison_features:
        left = pd.Series([row.get(feature) for row in librosa_rows if row])
        right = pd.Series([row.get(feature) for row in torch_rows if row])
        rows.append(
            {
                "feature": feature,
                "librosa_median": float(left.median()),
                "torchaudio_median": float(right.median()),
                "median_abs_delta": float((left - right).abs().median()),
            }
        )

    report = [
        "# Audio Backend Benchmark",
        "",
        f"- Tracks: {len(paths)}",
        f"- Librosa seconds: {librosa_seconds:.2f}",
        f"- Torchaudio seconds: {torch_seconds:.2f}",
        f"- Speedup: {librosa_seconds / torch_seconds:.2f}x" if torch_seconds else "- Speedup: N/A",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "Note: torchaudio features are not bit-identical to librosa. Treat backend changes as a dataset-version change.",
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved benchmark report to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity/src \
  uv run --python 3.12 --with pandas --with numpy --with librosa --with torch --with torchaudio --with tabulate \
  python src/benchmark_audio_backends.py --limit 25 --device auto --preview-seconds 30
```

Expected: `reports/audio_backend_benchmark.md` exists and includes timing plus median deltas.

- [ ] **Step 3: Commit**

```bash
git add src/benchmark_audio_backends.py reports/audio_backend_benchmark.md
git commit -m "test: benchmark librosa and torchaudio backends"
```

---

### Task 5: Regenerate Torch Dataset and Reports

**Files:**
- Modify generated: `data/processed/extended_features.csv`
- Modify generated: `reports/dataset_qc.json`
- Modify generated: `reports/dataset_qc.md`
- Modify generated: `reports/model_observability.json`
- Modify generated: `reports/model_observability.md`
- Modify generated: `reports/popular_feature_research.md`
- Modify generated: `models/xgboost_music_model.pkl`
- Modify generated: `models/xgboost_features.pkl`
- Modify generated: `models/xgboost_score_metadata.pkl`
- Modify generated: `plots/feature_importance_xgb.png`

- [ ] **Step 1: Extract the operational torch dataset**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity/src \
  uv run --python 3.12 --with pandas --with numpy --with torch --with torchaudio --with tqdm \
  python src/extract_extended_features.py \
    --backend torchaudio \
    --device auto \
    --batch-size 16 \
    --low-streams-csv data/raw/low_stream_tracks_sample_500.csv \
    --max-chart-tracks 500 \
    --max-low-stream-tracks 500 \
    --checkpoint-every 100 \
    --preview-seconds 30
```

Expected: `data/processed/extended_features.csv` has approximately 988 rows and includes `feature_backend=torchaudio`.

- [ ] **Step 2: Regenerate QC**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 \
  --with pandas --with numpy --with scikit-learn --with soundfile \
  python src/qc_extended_dataset.py \
    --features data/processed/extended_features.csv \
    --sample-manifest data/raw/low_stream_tracks_sample_500.csv \
    --failures reports/feature_extraction_failures.csv \
    --download-failures reports/low_stream_download_failures_sample_3000.csv \
    --quality-output reports/audio_source_quality.csv \
    --output-json reports/dataset_qc.json \
    --output-md reports/dataset_qc.md
```

Expected: `QC report saved to reports/dataset_qc.md`.

- [ ] **Step 3: Retrain model**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity/src \
  uv run --python 3.12 \
    --with pandas --with numpy --with scikit-learn --with xgboost \
    --with matplotlib --with seaborn --with joblib --with librosa \
  python src/train_extended_model.py
```

Expected: model artifacts saved and ROC-AUC/source-bias metrics printed.

- [ ] **Step 4: Regenerate research summary**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 --with pandas python src/write_popular_feature_research.py
```

Expected: `Research report saved`.

- [ ] **Step 5: Commit generated torch dataset artifacts**

```bash
git add data/processed/extended_features.csv reports/dataset_qc.json reports/dataset_qc.md \
  reports/feature_extraction_failures.csv reports/audio_source_quality.csv \
  reports/model_observability.json reports/model_observability.md reports/popular_feature_research.md \
  models/xgboost_features.pkl models/xgboost_music_model.pkl models/xgboost_score_metadata.pkl \
  plots/feature_importance_xgb.png
git commit -m "data: regenerate features with torchaudio backend"
```

---

### Task 6: Documentation and Dependency Notes

**Files:**
- Modify: `README.md`
- Optional modify: `requirements.txt`

- [ ] **Step 1: Document install modes**

Add to `README.md`:

```markdown
### Torchaudio / AMD ROCm extraction

The dataset extractor supports a batched `torchaudio` backend:

```bash
python src/extract_extended_features.py --backend torchaudio --device auto --batch-size 16
```

For normal CPU development:

```bash
pip install torch torchaudio
```

For Linux AMD ROCm, install matching PyTorch ROCm wheels. Project target:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

If ROCm 7.2 does not match the local driver stack:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
```

PyTorch exposes AMD ROCm devices through the `cuda` API, so `--device cuda` is expected on AMD GPUs when ROCm is working.
```

- [ ] **Step 2: Decide whether to normalize `requirements.txt`**

If we edit `requirements.txt`, convert it from UTF-16 LE to UTF-8 in a separate commit and replace Windows ROCm direct URLs with plain package names:

```text
torch
torchaudio
```

Keep ROCm-specific `--index-url` out of requirements unless this project adopts a single locked Linux ROCm environment.

- [ ] **Step 3: Commit docs**

```bash
git add README.md requirements.txt
git commit -m "docs: document torchaudio rocm setup"
```

---

### Task 7: Final Verification

**Files:**
- No new source files

- [ ] **Step 1: Compile Python files**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 \
  --with pandas --with numpy --with librosa --with streamlit --with plotly \
  --with joblib --with shap --with xgboost --with torch --with torchaudio \
  python -m py_compile \
    app.py \
    src/music_success_predictor.py \
    src/download_low_stream_tracks.py \
    src/extract_extended_features.py \
    src/torch_audio_features.py \
    src/normalize_low_stream_paths.py \
    src/research_insights.py \
    src/write_popular_feature_research.py \
    tests/test_research_insights.py \
    tests/test_torch_audio_features.py
```

Expected: exit code 0.

- [ ] **Step 2: Run tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity \
  uv run --python 3.12 --with pandas --with torch --with torchaudio \
  python -m unittest tests.test_research_insights tests.test_torch_audio_features -v
```

Expected: all tests pass.

- [ ] **Step 3: App import smoke**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/home/skadibtw/backup/music-popularity \
  uv run --python 3.12 \
    --with pandas --with numpy --with librosa --with streamlit --with plotly \
    --with joblib --with shap --with xgboost --with torch --with torchaudio \
  python -c "import app; print('imported', app.model is not None, len(app.feature_cols or []))"
```

Expected: `imported True <feature_count>`.

- [ ] **Step 4: Dataset count check**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 --with pandas python -c "import pandas as pd; df=pd.read_csv('data/processed/extended_features.csv'); print(len(df)); print(df['feature_backend'].value_counts(dropna=False).to_dict()); print(df['label_source'].value_counts().to_dict())"
```

Expected: `feature_backend` is mostly or entirely `torchaudio`, and label counts are balanced enough for model training.

- [ ] **Step 5: Commit any missed verification/doc updates**

```bash
git status --short
```

Expected: only pre-existing unrelated dirty files remain, or clean if those were handled separately.

---

## Self-Review

- Spec coverage: The plan covers `torchaudio`, AMD ROCm/Linux setup, normal CPU torch requirements, batched GPU extraction, dataset regeneration, QC, model retraining, and docs.
- Risk control: The `librosa` backend remains available until torchaudio parity and generated reports are reviewed.
- Known tradeoff: `tempo` and `tonnetz` are initially placeholder-compatible in the torch backend. If those become important model features after retraining, add a follow-up task for torch tempo estimation and harmonic/tonnetz replacement rather than blocking the GPU migration.
