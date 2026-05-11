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
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "mfcc_shape": list(mfcc.shape),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
