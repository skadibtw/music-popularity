import torch
import torchaudio


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = torch.tensor(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = torch.tensor(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


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
            raise ValueError(
                "Expected waveform tensor with shape [batch, samples] or "
                "[batch, channels, samples]."
            )
        if waveforms.shape[1] < self.n_fft:
            waveforms = torch.nn.functional.pad(waveforms, (0, self.n_fft - waveforms.shape[1]))

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
        freqs = torch.linspace(
            0, self.sample_rate / 2, magnitude.shape[1], device=self.device
        )

        centroid = (freqs[None, :, None] * magnitude).sum(dim=1) / magnitude.sum(
            dim=1
        ).clamp_min(1e-10)
        bandwidth = torch.sqrt(
            (((freqs[None, :, None] - centroid[:, None, :]).abs() ** 2) * magnitude).sum(
                dim=1
            )
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
            )
            .squeeze(1)
            .clamp_min(1e-10)
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
                "spectral_centroid_std": float(
                    centroid[i].std(unbiased=False).detach().cpu()
                ),
                "spectral_rolloff_mean": float(rolloff[i].mean().detach().cpu()),
                "spectral_bandwidth_mean": float(bandwidth[i].mean().detach().cpu()),
                "zcr_mean": float(zcr[i].mean().detach().cpu()) if zcr.shape[1] else 0.0,
                "zcr_std": float(zcr[i].std(unbiased=False).detach().cpu())
                if zcr.shape[1]
                else 0.0,
                "chroma_mean": float(chroma_i.mean().detach().cpu()),
                "chroma_std": float(chroma_i.std(unbiased=False).detach().cpu()),
                "rms_mean": float(rms_frames[i].mean().detach().cpu()),
                "rms_std": float(rms_frames[i].std(unbiased=False).detach().cpu()),
                "tonnetz_mean": 0.0,
                "tonnetz_std": 0.0,
            }
            for j in range(self.n_mfcc):
                row[f"mfcc_{j}_mean"] = float(mfcc[i, j].mean().detach().cpu())
                row[f"mfcc_{j}_std"] = float(
                    mfcc[i, j].std(unbiased=False).detach().cpu()
                )
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
            return float(
                (x * y).sum()
                / (torch.sqrt((x.square()).sum() * (y.square()).sum()) + 1e-10)
            )

        major_scores = [score(major, i) for i in range(12)]
        minor_scores = [score(minor, i) for i in range(12)]
        major_idx = int(max(range(12), key=lambda i: major_scores[i]))
        minor_idx = int(max(range(12), key=lambda i: minor_scores[i]))
        if major_scores[major_idx] >= minor_scores[minor_idx]:
            return f"{NOTES[major_idx]} major"
        return f"{NOTES[minor_idx]} minor"
