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
