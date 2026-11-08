"""Reference-guided noise-floor removal for Seed-VC converted clips.

We have the real source (clean, same content, same timing) next to every
converted clip.  The hiss added by the converter is roughly additive and
roughly stationary in 6-10 kHz, so:

  1. estimate how the converter maps the source spectrum to the output
     (per-bin least squares over loud frames) -> expected signal S = g * |X|
  2. residual R = |Y| - S; take the median of R over frames where the
     source is nearly silent -> that is the noise floor N(f)
  3. Wiener-style subtraction with a soft floor, applied only inside the
     noise band (5.5-11.5 kHz, 500 Hz ramps), so sibilance carried by the
     source survives.

usage:  python _ref_dn.py --src <dir-or-file> --conv <dir> -o <outdir> [--over 1.2 --floor 0.06]
"""
import os
import argparse
import numpy as np
import soundfile as sf

SR = 44100
N = 2048
HOP = 512
WIN = np.hanning(N)


def load(p):
    y, sr = sf.read(p, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def stft(y):
    pad = np.zeros(N // 2, np.float32)
    yp = np.concatenate([pad, y, pad])
    k = 1 + (len(yp) - N) // HOP
    idx = np.arange(N)[None, :] + HOP * np.arange(k)[:, None]
    return np.fft.rfft(yp[idx] * WIN, axis=1), yp


def istft(S, length):
    fr = np.fft.irfft(S, axis=1).astype(np.float32) * WIN
    out = np.zeros(HOP * (len(S) - 1) + N, np.float32)
    wsum = np.zeros_like(out)
    for i in range(len(S)):
        out[i * HOP:i * HOP + N] += fr[i]
        wsum[i * HOP:i * HOP + N] += WIN ** 2
    out = out / np.maximum(wsum, 1e-8)
    return out[N // 2:N // 2 + length]


def band_mask(f, lo, hi, ramp=500.0):
    m = np.zeros(len(f))
    m[(f >= lo) & (f < hi)] = 1.0
    a = (f >= lo - ramp) & (f < lo)
    m[a] = (f[a] - (lo - ramp)) / ramp
    b = (f >= hi) & (f < hi + ramp)
    m[b] = (hi + ramp - f[b]) / ramp
    return m


def frames_mid(absX, f):
    return (absX[:, (f >= 1000) & (f < 4000)]) @ np.ones(((f >= 1000) & (f < 4000)).sum(), np.float32)


def denoise(x, y, over=1.2, floor=0.06, f_lo=5500.0, f_hi=11500.0, verbose=False):
    X, _ = stft(x)
    Y, _ = stft(y)
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    f = np.fft.rfftfreq(N, 1.0 / SR)
    absX, absY = np.abs(X), np.abs(Y)

    mid = (absX[:, (f >= 1000) & (f < 4000)] ** 2).sum(1)
    order = np.argsort(mid)
    k_loud = max(8, int(0.40 * len(mid)))
    k_quiet = max(8, int(0.15 * len(mid)))
    sig = np.zeros(len(mid), bool)
    sig[order[-k_loud:]] = True
    qui = np.zeros(len(mid), bool)
    qui[order[:k_quiet]] = True

    g = (absX[sig] * absY[sig]).sum(0) / ((absX[sig] ** 2).sum(0) + 1e-9)
    g = np.clip(g, 0.0, 3.0)
    S = g[None, :] * absX
    R = np.clip(absY - S, 0.0, None)
    Nf = np.median(R[qui], axis=0)
    Nf = np.nan_to_num(Nf, nan=0.0, posinf=0.0, neginf=0.0)
    Nf = np.convolve(Nf, np.ones(9) / 9.0, mode="same")

    mask = band_mask(f, f_lo, f_hi)
    G = 1.0 - over * mask[None, :] * Nf[None, :] / (absY + 1e-9)
    G = np.clip(G, floor, 1.0)
    out = istft(Y * G, len(y))

    rms_in = np.sqrt((y ** 2).mean())
    rms_out = np.sqrt((out ** 2).mean() + 1e-12)
    out = out * (rms_in / rms_out)
    if verbose:
        nb = (f >= 6000) & (f < 10000)
        print("  floor 6-10k %.1f dBFS, mean gain %.2f" %
              (10 * np.log10(Nf[nb].mean() ** 2 + 1e-20), float(G.mean())))
    return np.clip(out, -1.0, 1.0)


def stats(y):
    Y, _ = stft(y)
    f = np.fft.rfftfreq(N, 1.0 / SR)
    P = (np.abs(Y) ** 2).sum(0)
    tot = P.sum()
    bands = [P[(f >= a) & (f < b)].sum() / tot * 100 for a, b in
             [(6000, 8000), (8000, 10000), (10000, 12000), (12000, 16000)]]
    mid = (np.abs(Y[:, (f >= 1000) & (f < 4000)]) ** 2).sum(1)
    ok = mid > 1e-9
    lu = ok & (mid > np.percentile(mid[ok], 80))
    qu = ok & (mid < np.percentile(mid[ok], 30))
    dyn = {}
    for a, b in [(6000, 10000), (12000, 16000)]:
        e = (np.abs(Y[:, (f >= a) & (f < b)]) ** 2).sum(1)
        dyn[(a, b)] = 10 * np.log10(np.median(e[lu] / mid[lu])) - 10 * np.log10(np.median(e[qu] / mid[qu]))
    return bands, dyn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--over", type=float, default=1.2)
    ap.add_argument("--floor", type=float, default=0.06)
    ap.add_argument("--ref-fill", default="", help="可选：补过空气的版本，说明差异")
    a = ap.parse_args()

    os.makedirs(a.output, exist_ok=True)
    if os.path.isdir(a.src):
        names = sorted(os.path.basename(p) for p in
                       [os.path.join(a.src, x) for x in os.listdir(a.src)] if p.endswith(".wav"))
    else:
        names = [os.path.basename(a.src)]
        a.src = os.path.dirname(a.src)

    rows = {"源": [], "SeedVC": [], "去底噪": []}
    done = 0
    for nm in names:
        sp = os.path.join(a.src, nm)
        cp = os.path.join(a.conv, nm)
        if not (os.path.exists(sp) and os.path.exists(cp)):
            continue
        x, y = load(sp), load(cp)
        z = denoise(x, y, a.over, a.floor)
        op = os.path.join(a.output, nm)
        sf.write(op, z, SR, subtype="PCM_16")
        for k, v in (("源", x), ("SeedVC", y), ("去底噪", z)):
            rows[k].append(stats(v))
        done += 1
    print("done", done)
    for k, rs in rows.items():
        b = np.median(np.array([r[0] for r in rs]), axis=0)
        d1 = np.median([r[1][(6000, 10000)] for r in rs])
        d2 = np.median([r[1][(12000, 16000)] for r in rs])
        print("%-6s 6-8k %.2f 8-10k %.2f 10-12k %.2f 12-16k %.2f | 动态 6-10k %+.1f 12-16k %+.1f"
              % (k, b[0], b[1], b[2], b[3], d1, d2))


if __name__ == "__main__":
    main()
