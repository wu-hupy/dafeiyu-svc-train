# -*- coding: utf-8 -*-
"""三方对照指标：f0/音高误差、频带、细节%、CAM++（到 prompt / 到源）。

用法（sing venv 跑）：
  $PY _three_way_eval.py --ref _three_way/src --dirs _three_way/soulx _three_way/v2
"""
import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf

SR = 44100


def load(p, sr=SR):
    x, s = sf.read(p, always_2d=True)
    x = x.mean(1).astype(np.float32)
    if s != sr:
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    return x


def f0_track(x, sr=SR, hop=441, n=2048):
    """自相关逐帧 f0（Hz），返回 (f0, rms)。"""
    f0s, es = [], []
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] * np.hanning(n)
        e = float(np.sqrt((fr ** 2).mean()))
        es.append(e)
        if e < 1e-4:
            f0s.append(0.0)
            continue
        c = np.correlate(fr, fr, "full")[n - 1:]
        lo, hi = int(sr / 700), int(sr / 70)
        seg = c[lo:hi]
        if len(seg) == 0 or seg.max() <= 0:
            f0s.append(0.0)
            continue
        lag = lo + int(np.argmax(seg))
        f0s.append(sr / lag)
    return np.array(f0s), np.array(es)


def spec_stats(x, sr=SR, n=2048, hop=1024):
    win = np.hanning(n)
    P = []
    for i in range(0, len(x) - n, hop):
        P.append(np.abs(np.fft.rfft(x[i:i + n] * win)) ** 2)
    P = np.array(P) + 1e-12
    freqs = np.fft.rfftfreq(n, 1 / sr)
    tot = P.sum(1, keepdims=True)
    Pn = P / tot
    cent = float(np.median((Pn * freqs).sum(1) / Pn.sum(1)))
    def band(a, b):
        m = (freqs >= a) & (freqs < b)
        return float((P[:, m].sum(1) / tot[:, 0]).mean())

    keep = (freqs >= 100) & (freqs <= 10000)
    db = 10 * np.log10(P.mean(0) + 1e-20)
    a, b = np.polyfit(np.log10(freqs[keep]), db[keep], 1)
    voiced = Pn[:, (freqs >= 300) & (freqs < 3000)].sum(1).mean()
    return dict(centroid=cent, tilt=a, hi10k=band(10000, 20000),
                b12_16=band(12000, 16000), b2_4=band(2000, 4000),
                b4_6=band(4000, 6000), b6_8=band(6000, 8000),
                b8_12=band(8000, 12000), voiced_frac=voiced)


def detail_frac(x, sr=SR, n=2048, hop=441):
    """非谐波能量占比（气声/摩擦）。"""
    fr = []
    for i in range(0, len(x) - n, hop):
        seg = x[i:i + n] * np.hanning(n)
        c = np.correlate(seg, seg, "full")[n - 1:]
        lo, hi = int(sr / 700), int(sr / 70)
        if c[lo:hi].max() <= 0:
            continue
        lag = lo + int(np.argmax(c[lo:hi]))
        f0 = sr / lag
        S = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(n, 1 / sr)
        harm = np.zeros_like(S, dtype=bool)
        k = 1
        while k * f0 < sr / 2:
            harm |= np.abs(freqs - k * f0) <= 2 * (sr / n)
            k += 1
        if S.sum() <= 0:
            continue
        fr.append(1 - S[harm].sum() / S.sum())
    return float(np.median(fr)) if fr else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="源干声目录（同文件名）")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--prompt", default=r"D:\dafeiyu\sing\voice\dafeiyu-prompt.wav")
    args = ap.parse_args()

    import sys
    from pathlib import Path
    sys.path.insert(0, r"D:\AI-Song")
    from speaker_verify import load_model as load_campplus, read16k

    cpp, frontend = load_campplus()
    import torch

    def emb(p):
        y = read16k(Path(p))
        t = torch.from_numpy(y)[None, :]
        feats, _ = frontend(t, torch.tensor([t.shape[1]]))
        with torch.inference_mode():
            v = cpp(feats)
        v = torch.nn.functional.normalize(v, dim=1)[0].cpu().numpy()
        return v

    E = {}
    def get_emb(p):
        if p not in E:
            E[p] = emb(p)
        return E[p]

    pe = get_emb(args.prompt)
    pe = pe / np.linalg.norm(pe)

    files = sorted(f for f in os.listdir(args.ref) if f.endswith(".wav"))
    for f in files:
        src = load(os.path.join(args.ref, f))
        se = get_emb(os.path.join(args.ref, f))
        se = se / np.linalg.norm(se)
        s_f0, s_e = f0_track(src)
        s_st = spec_stats(src)
        print(f"\n== {f}  (源)")
        print(f"   f0med {np.median(s_f0[s_f0>0]):.1f}  centroid {s_st['centroid']:.0f} "
              f"tilt {s_st['tilt']:.2f}  hi10k {s_st['hi10k']*100:.2f}%  "
              f"12-16k {s_st['b12_16']*100:.2f}%  detail {detail_frac(src)*100:.2f}%")
        for d in args.dirs:
            cand = [p for p in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
                    if f.split(".")[0].split("_")[0].lower() in p.lower()]
            if not cand:
                print(f"   [miss] {d}")
                continue
            p = cand[0]
            y = load(p)
            n = min(len(y), len(src))
            o_f0, _ = f0_track(y[:n])
            k = min(len(s_f0), len(o_f0))
            m = (s_f0[:k] > 0) & (o_f0[:k] > 0)
            err = float(np.median(np.abs(12 * np.log2((o_f0[:k][m] + 1e-9) /
                                                    (s_f0[:k][m] + 1e-9)))))
            st = spec_stats(y)
            oe = get_emb(p)
            oe = oe / np.linalg.norm(oe)
            t = os.path.basename(os.path.dirname(p))
            print(f"   [{t:6s}] f0err {err*100:5.0f}音分  centroid {st['centroid']:.0f}  "
                  f"tilt {st['tilt']:.2f}  hi10k {st['hi10k']*100:.2f}%  "
                  f"12-16k {st['b12_16']*100:.2f}%  2-4k {st['b2_4']*100:.1f}%  "
                  f"detail {detail_frac(y)*100:.1f}%  "
                  f"cam++→prompt {float(oe@pe):.3f}  →源 {float(oe@se):.3f}")


if __name__ == "__main__":
    main()
