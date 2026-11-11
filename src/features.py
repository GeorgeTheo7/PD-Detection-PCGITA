"""Feature computation for the three branches.

* Wav2Vec2 (frozen)  -> computed once and cached to disk (.npy, float16)
* Spectrogram image  -> [3, 224, 224]: log-Mel, delta, delta-delta (ImageNet-normalised)
* MFCC sequence      -> [T, 40]
Spectrograms and MFCCs are deterministic and cheap, so they are recomputed in memory at train time.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import config


def load_wav(path: str) -> np.ndarray:
    """Read a chunk as mono float32, padded/truncated to exactly 5 s."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != config.TARGET_SR:
        raise ValueError(f"{path}: expected {config.TARGET_SR} Hz, got {sr} Hz (run `prepare` first)")
    wav = np.nan_to_num(wav)
    n = config.CHUNK_SAMPLES
    return wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))


def _batches(df, chunk_dir, bs):
    for i in range(0, len(df), bs):
        names = df["filename"].iloc[i:i + bs].tolist()
        yield i, np.stack([load_wav(os.path.join(chunk_dir, n)) for n in names])


@torch.no_grad()
def extract_wav2vec2(csv_path, chunk_dir, out_path, device="cuda", batch_size=16):
    """Run the frozen Wav2Vec2 model once over all chunks; save [N, T, 1024] float16 in CSV row order."""
    from transformers import Wav2Vec2Model

    df = pd.read_csv(csv_path)
    model = Wav2Vec2Model.from_pretrained(config.WAV2VEC2_ID).to(device).eval()
    feats, t0 = None, time.time()
    use_amp = device.startswith("cuda")
    for i, x in _batches(df, chunk_dir, batch_size):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(input_values=torch.from_numpy(x).to(device)).last_hidden_state
        if feats is None:
            feats = np.zeros((len(df), out.shape[1], out.shape[2]), dtype=np.float16)
        feats[i:i + len(x)] = out.float().cpu().numpy().astype(np.float16)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, feats)
    print(f"Wav2Vec2 features {feats.shape} saved to {out_path} ({time.time() - t0:.0f}s)")
    return out_path


@torch.no_grad()
def compute_spec_mfcc(csv_path, chunk_dir, device="cuda", batch_size=64):
    """Return (SPEC [N,3,224,224] float16, MFCC [N,T,40] float32) on CPU, in CSV row order."""
    import torchaudio.functional as AF
    import torchaudio.transforms as AT
    import torchvision.transforms as VT

    df = pd.read_csv(csv_path)
    mel_kw = dict(n_fft=config.N_FFT, win_length=config.WIN_LENGTH, hop_length=config.HOP_LENGTH,
                  n_mels=config.N_MELS, power=2.0)
    mel = AT.MelSpectrogram(sample_rate=config.TARGET_SR, **mel_kw).to(device)
    to_db = AT.AmplitudeToDB(stype="power").to(device)
    mfcc_t = AT.MFCC(sample_rate=config.TARGET_SR, n_mfcc=config.N_MFCC, log_mels=True,
                     melkwargs=mel_kw).to(device)
    imnet = VT.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    spec = torch.empty((len(df), 3, *config.IMG_SIZE), dtype=torch.float16)
    mfcc = None
    for i, x in _batches(df, chunk_dir, batch_size):
        x = torch.from_numpy(x).to(device)                                   # [B, L]
        lm = to_db(mel(x) + 1e-6)                                            # [B, 200, T]
        d1 = AF.compute_deltas(lm)
        d2 = AF.compute_deltas(d1)
        rgb = torch.stack([lm, d1, d2], dim=1)                               # [B, 3, 200, T]
        rgb = F.interpolate(rgb, size=config.IMG_SIZE, mode="bilinear", align_corners=False)
        cmin = rgb.amin(dim=(2, 3), keepdim=True)
        cmax = rgb.amax(dim=(2, 3), keepdim=True)
        rgb = imnet((rgb - cmin) / (cmax - cmin).clamp(min=1e-6))           # per-image min-max, then ImageNet
        spec[i:i + len(x)] = rgb.half().cpu()

        m = mfcc_t(x).transpose(1, 2)                                        # [B, T, 40]
        if mfcc is None:
            mfcc = torch.empty((len(df), m.shape[1], config.N_MFCC))
        mfcc[i:i + len(x)] = m.cpu()
    return spec, mfcc
