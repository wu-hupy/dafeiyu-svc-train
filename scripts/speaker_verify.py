#!/usr/bin/env python3
"""Speaker-verification check with CAM++ (3D-Speaker, funasr).

Run with the sing venv (has torch + torchaudio + funasr):
    python scripts/speaker_verify.py f1.wav f2.wav ...

Embeddings are averaged over 6 s windows (3 s hop), L2-normalised.
Prints the cosine-similarity matrix and a same/different verdict per pair.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

MODEL_DIR = Path(os.environ.get("CAMPP_DIR", "models/campplus"))
MODEL_BIN = MODEL_DIR / "campplus_cn_common.bin"
TARGET_SR = 16000
WIN = 6.0
HOP = 3.0


def load_model():
    from funasr.models.campplus.model import CAMPPlus
    from funasr.frontends.wav_frontend import WavFrontend

    conf = dict(
        feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4,
        init_channels=128, config_str="batchnorm-relu", memory_efficient=True,
        output_level="segment",
    )
    model = CAMPPlus(**conf)
    state = torch.load(MODEL_BIN, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} -> {missing[:3]}")
    model.eval()
    frontend = WavFrontend(fs=TARGET_SR, dither=0.0).eval()
    return model, frontend


def read16k(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != TARGET_SR:
        t = torch.from_numpy(y)[None, :]
        y = torchaudio.functional.resample(t, sr, TARGET_SR)[0].numpy()
    return y.astype(np.float32)


@torch.inference_mode()
def embed(model, frontend, path: Path) -> np.ndarray:
    y = read16k(path)
    win = int(WIN * TARGET_SR)
    hop = int(HOP * TARGET_SR)
    starts = list(range(0, max(1, len(y) - win + 1), hop)) or [0]
    vecs = []
    for s in starts:
        seg = y[s:s + win]
        if len(seg) < int(0.7 * TARGET_SR):
            continue
        t = torch.from_numpy(seg)[None, :]
        feats, _ = frontend(t, torch.tensor([t.shape[1]]))
        if feats.shape[1] < 10:
            continue
        emb = model(feats)
        vecs.append(torch.nn.functional.normalize(emb, dim=1)[0].numpy())
    if not vecs:
        raise SystemExit(f"no usable audio in {path}")
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> int:
    files = [Path(p) for p in sys.argv[1:]]
    if not files:
        print(__doc__)
        return 1
    model, frontend = load_model()
    names, embs = [], []
    for f in files:
        e = embed(model, frontend, f)
        names.append(f.stem[:34])
        embs.append(e)
        print(f"embedded {f.name} ({len(e)}d)")
    E = np.stack(embs)
    S = E @ E.T
    w = max(len(n) for n in names) + 2
    print("\nCOSINE SIMILARITY")
    print(" " * w + "".join(f"{i:>8d}" for i in range(len(names))))
    for i, n in enumerate(names):
        print(f"{n:<{w}}" + "".join(f"{S[i, j]:8.3f}" for j in range(len(names))))
    print("\n" + " " * 1 + "legend: 0 " + "\n          ".join(f"{i}={n}" for i, n in enumerate(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
