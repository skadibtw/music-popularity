"""
Audio feature extraction and scoring helpers for chart-track hit-likeness analysis.
"""

import warnings
import sys

import librosa
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


DEFAULT_PREVIEW_SECONDS = 30


PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def add_key_features(df):
    """Represent musical key as circular pitch-class features plus mode."""
    key_parts = df["key"].fillna("C major").str.split(" ", n=1, expand=True)
    root = key_parts[0].map(PITCH_CLASS).fillna(0).astype(float)
    if key_parts.shape[1] > 1:
        mode = key_parts[1].fillna("major")
    else:
        mode = pd.Series("major", index=df.index)
    radians = 2 * np.pi * root / 12
    df["key_root_sin"] = np.sin(radians)
    df["key_root_cos"] = np.cos(radians)
    df["key_is_minor"] = mode.eq("minor").astype(int)
    return df


def score_percentile(raw_score, metadata):
    """Map a model score to its percentile among out-of-fold reference scores."""
    reference_scores = np.asarray(metadata.get("oof_scores", []), dtype=float)
    if reference_scores.size == 0:
        return float(raw_score * 100)
    return float((reference_scores <= raw_score).mean() * 100)


def robust_feature_distance(row, metadata):
    """Estimate whether a song is inside the training feature distribution."""
    medians = metadata.get("feature_median", {})
    iqrs = metadata.get("feature_iqr", {})
    if not medians or not iqrs:
        return 0.0, 0.0, True

    values = row[list(medians.keys())].iloc[0].astype(float)
    median = pd.Series(medians, dtype=float)
    iqr = pd.Series(iqrs, dtype=float).replace(0, 1.0)
    distance = float(((values - median).abs() / iqr).mean())
    threshold = float(metadata.get("ood_distance_p95", distance))
    return distance, threshold, distance <= threshold


def charted_similarity_percentile(row, metadata):
    """Score closeness to the weighted charted audio-feature reference distribution."""
    medians = metadata.get("charted_feature_median", {})
    iqrs = metadata.get("charted_feature_iqr", {})
    reference_distances = np.asarray(metadata.get("charted_reference_distances", []), dtype=float)
    if not medians or not iqrs or reference_distances.size == 0:
        return 0.0, 0.0

    values = row[list(medians.keys())].iloc[0].astype(float)
    median = pd.Series(medians, dtype=float)
    iqr = pd.Series(iqrs, dtype=float).replace(0, 1.0)
    distance = float(((values - median).abs() / iqr).mean())
    percentile = float((reference_distances >= distance).mean() * 100)
    return percentile, distance


class AudioFeatureExtractor:
    """Extract a compact audio-feature set from MP3/WAV files."""

    def _load_preview_audio(audio_path, preview_seconds=30):
        """Load the same fixed preview window for every source to reduce duration/source bias."""
        return librosa.load(audio_path, mono=True, duration=preview_seconds)

    @staticmethod
    def get_duration(audio_path):
        try:
            return float(librosa.get_duration(path=audio_path))
        except TypeError:
            return float(librosa.get_duration(filename=audio_path))

    @staticmethod
    def extract_features(
        audio_path,
        preview_seconds=DEFAULT_PREVIEW_SECONDS,
    ):
        """Extract audio features from a local file."""
        try:
            y, sr = AudioFeatureExtractor._load_preview_audio(audio_path, preview_seconds)

            features = {}
            features["duration_seconds"] = AudioFeatureExtractor.get_duration(audio_path)
            features["analyzed_duration_seconds"] = float(len(y) / sr) if sr else 0.0

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            features["key"] = AudioFeatureExtractor._estimate_key(chroma)

            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_std"] = float(np.std(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
                features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma_mean"] = float(np.mean(chroma_stft))
            features["chroma_std"] = float(np.std(chroma_stft))

            rms = librosa.feature.rms(y=y)[0]
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features["tonnetz_mean"] = float(np.mean(tonnetz))
            features["tonnetz_std"] = float(np.std(tonnetz))

            return features

        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
            return None

    @staticmethod
    def _estimate_key(chroma):
        """Estimate musical key from a chromagram using Krumhansl profiles."""
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
        pitch_profile = chroma.mean(axis=1)

        def best_match(profile):
            scores = []
            p = profile / profile.sum()
            for i in range(12):
                scores.append(np.corrcoef(p, np.roll(pitch_profile, i))[0, 1])
            key_index = int(np.argmax(scores))
            return key_index, float(np.max(scores))

        major_root, major_score = best_match(major_profile)
        minor_root, minor_score = best_match(minor_profile)

        if major_score >= minor_score:
            return f"{notes[major_root]} major"
        return f"{notes[minor_root]} minor"
