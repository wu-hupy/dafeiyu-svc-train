"""Paired spectral correction for converted clips (reference-guided).

Idea
----
Every converted clip has its real source (same id, same words, same timing).
Comparing the two in the *loud* frames gives the converter's spectral
transfer error; correcting it on the output removes the 6-8 kHz sizzle that
makes the data sound hissy, without touching the timbre itself.

Steps
-----
1. band levels of source and output over loud frames (anchored on mid band)
2. correction gain per band, only above --from, capped to +-cap dB
3. gentle frequency smoothing -> zero phase EQ
4. optional HF-dynamics gate: where the source's 6-16k sits far below its own
   loud-frame level (breaths, gaps), pull the output's HF down toward the same
   relative level; only attenuating, 120 ms smoothed
5. loudness anchor on the mid band (300-4000 Hz), not on RMS, so the EQ does
   not shift the whole spectrum

usage:
  python _ref_eq.py --src <src-dir> --conv <conv-dir> -o <out-dir>
                    [--from 4000 --cap 10 --gate 1 --limit 100]
"""
import os
import argparse
import numpy as np
import soundfile as sf

from ref_dn import SR, N, HOP, load, stft, istft

BANDS = [(300, 600), (600, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
         (4000, 6000), (6000, 8000), (8000, 10000), (10000, 12000), (12000, 16000)]
F = np.fft.rfftfreq(N, 1.0 / SR)


def mid_energy(X):
    return (np.abs(X[:, (F >= 1000) & (F < 4000)]) ** 2).sum(1)


def band_levels(X, sel):
    P = np.abs(X[sel]) ** 2
    out = []
    for a, b in BANDS:
        m = (F >= a) & (F < b)
        out.append(10 * np.log10(P[:, m].mean() + 1e-20))
    return np.array(out)


def band_response(gain_db, smooth_hz=500.0):
    resp = np.zeros(len(F))
    for (a, b), g in zip(BANDS, gain_db):
        resp[(F >= a) & (F < b)] = g
    k = max(1, int(smooth_hz / (SR / N)))
    if k > 1:
        resp = np.convolve(resp, np.ones(k) / k, mode="same")
    return resp


def hf_env(X, lo=6000.0, hi=16000.0):
    e = (np.abs(X[:, (F >= lo) & (F < hi)]) ** 2).sum(1)
    return 10 * np.log10(e + 1e-20)


def shares_of(X):
    P = (np.abs(X) ** 2).sum(0)
    tot = P.sum() + 1e-20
    return np.array([100.0 * P[(F >= a) & (F < b)].sum() / tot for a, b in BANDS])


def correct(x, y, frm=4000.0, cap=10.0, gate=1.0, smooth_hz=500.0,
            low_cap=4.0, low_blend=0.6, low_to=4000.0, iters=3):
    """按整段频带占比与源对齐：每轮把 |src_share - out_share| 折成 dB 修正。"""
    X, _ = stft(x)
    Y, _ = stft(y)
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    sx = shares_of(X)
    Z = Y.copy()
    resp = np.zeros(len(F))
    for _ in range(iters):
        sz = shares_of(Z)
        step = np.zeros(len(BANDS))
        for i, (a, b) in enumerate(BANDS):
            d = 10 * np.log10((sx[i] + 1e-9) / (sz[i] + 1e-9))
            lim = cap if a >= frm else low_cap
            if a >= frm:
                step[i] = np.clip(d, -lim, lim)
            elif b <= low_to:
                step[i] = np.clip(d * low_blend, -lim, lim)
        resp = resp + band_response(step, smooth_hz)
        Z = Y * (10 ** (resp / 20.0))[None, :]

    if gate > 0:
        hf_s = (np.abs(X[:, (F >= 6000) & (F < 8000)]) ** 2).sum(1)
        hf_z = (np.abs(Z[:, (F >= 6000) & (F < 8000)]) ** 2).sum(1)
        midX = (np.abs(X[:, (F >= 300) & (F < 4000)]) ** 2).sum(1) + 1e-12
        midZ = (np.abs(Z[:, (F >= 300) & (F < 4000)]) ** 2).sum(1) + 1e-12
        want = 10 * np.log10((hf_s / midX) + 1e-20) - 10 * np.log10((hf_z / midZ) + 1e-20)
        want = np.clip(want, -10.0, 0.0) * gate
        k = max(1, int(0.06 * SR / HOP))
        want = np.convolve(want, np.ones(k) / k, mode="same")
        gm = (F >= 6000) & (F < 16000)
        Z = Z * (10 ** (want[:, None] * gm[None, :] / 20.0))

    z = istft(Z, len(y))
    Z2, _ = stft(np.clip(z, -4, 4))
    m0 = (F >= 300) & (F < 4000)
    L = min(len(Y), len(Z2))
    e_in = (np.abs(Y[:L][:, m0]) ** 2).mean()
    e_out = (np.abs(Z2[:L][:, m0]) ** 2).mean() + 1e-12
    z = z * float(np.sqrt(e_in / e_out))
    return np.clip(z, -1.0, 1.0)


def dynamics(X, lu, qu):
    """6-10k 相对中频：低电平帧 - 有声帧；lu/qu 用源的帧选择，三版可比"""
    P = np.abs(X) ** 2
    mid = (P[:, (F >= 1000) & (F < 4000)]).sum(1)
    e = (P[:, (F >= 6000) & (F < 10000)]).sum(1)
    d = 10 * np.log10(np.median(e[lu] / mid[lu]) + 1e-20) - \
        10 * np.log10(np.median(e[qu] / mid[qu]) + 1e-20)
    lv = band_levels(X, lu)
    return d, lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--from", dest="frm", type=float, default=4000.0)
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--low-cap", type=float, default=4.0)
    ap.add_argument("--low-blend", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.output, exist_ok=True)
    names = sorted(n for n in os.listdir(a.src) if n.lower().endswith(".wav"))
    if a.limit:
        names = names[:a.limit]

    acc = {"源": [], "SeedVC": [], "校正后": []}
    dyn = {"源": [], "SeedVC": [], "校正后": []}
    done = 0
    for nm in names:
        sp, cp = os.path.join(a.src, nm), os.path.join(a.conv, nm)
        if not (os.path.exists(sp) and os.path.exists(cp)):
            continue
        x, y = load(sp), load(cp)
        z = correct(x, y, a.frm, a.cap, a.gate, low_cap=a.low_cap, low_blend=a.low_blend)
        sf.write(os.path.join(a.output, nm), z, SR, subtype="PCM_16")

        Xs, _ = stft(x)
        mid = mid_energy(Xs)
        ok = mid > 1e-9
        lu = ok & (mid > np.percentile(mid[ok], 80))
        qu = ok & (mid < np.percentile(mid[ok], 30))
        for k, v in (("源", x), ("SeedVC", y), ("校正后", z)):
            V, _ = stft(v)
            n = min(len(V), len(Xs))
            d, lv = dynamics(V[:n], lu[:n], qu[:n])
            acc[k].append(shares_of(V[:n]))
            dyn[k].append(d)
        done += 1
        if done % 20 == 0:
            print("  %d" % done, flush=True)

    print("\ndone %d" % done)
    print("band %    " + " ".join("%6d-%d" % (a_, b_) for a_, b_ in BANDS))
    for k in ("源", "SeedVC", "校正后"):
        m = np.median(np.array(acc[k]), axis=0)
        print("%-9s" % k + " ".join("%8.2f" % v for v in m))
    print("6-10k 动态: 源 %+.1f | SeedVC %+.1f | 校正后 %+.1f" %
          (np.median(dyn["源"]), np.median(dyn["SeedVC"]), np.median(dyn["校正后"])))


if __name__ == "__main__":
    main()
