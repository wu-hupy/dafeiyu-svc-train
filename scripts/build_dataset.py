# -*- coding: utf-8 -*-
"""v4 数据集构建：①按「转换后片段两两之间的音色一致」筛 ②按音高分层保证高低音都覆盖 ③控制在 1–2 小时。

流程：
  1. CAM++ 嵌入（L2 归一）→ 两两余弦，迭代砍掉最不一致的，直到平均两两余弦 ≥ --target
  2. 再过一道「到 medoid 余弦 ≥ --min-to-medoid」，剔离群
  3. 逐段估 f0 中位（自相关，60–800Hz，只看有声帧）
  4. 若总时长 > --max-minutes，按 f0 分层抽（每层等比例），保证 p5/p95 不被砍
  5. 按整首歌留验证集（默认 2081/2095，与 v2 可比）

用法：
  HF_HUB_OFFLINE=1 D:/dafeiyu/sing/venv/Scripts/python.exe _svc_v4_build.py \
      --in-dir svc_train/dataset_svc_v4 --out data/dafeiyu_v4 \
      --target 0.82 --min-to-medoid 0.72 --max-minutes 110
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, r"D:\AI-Song")
from _speaker_verify import embed, load_model  # noqa: E402

TRIM_DB = -45.0
TARGET_RMS_DB = -20.0
PEAK_LIMIT = 10 ** (-1.0 / 20)
HOP = 441
FRAME = 1024
F0_MIN, F0_MAX = 60.0, 800.0


def trim(x, sr):
    n = len(x) // FRAME
    if n < 3:
        return x
    e = np.array([np.sqrt(np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2) + 1e-12) for i in range(n)])
    keep = np.where(20 * np.log10(e + 1e-12) > TRIM_DB)[0]
    if len(keep) == 0:
        return x
    return x[keep[0] * FRAME:min(len(x), (keep[-1] + 1) * FRAME)]


def normalize(x):
    x = x * (10 ** (TARGET_RMS_DB / 20) / (np.sqrt(np.mean(x ** 2) + 1e-12)))
    peak = float(np.max(np.abs(x)))
    if peak > PEAK_LIMIT:
        x = x / peak * PEAK_LIMIT
    return x.astype(np.float32)


def f0_median(y, sr, min_rms_db=-40.0):
    """自相关估 f0，返回有声帧 f0 中位（Hz）。"""
    y = y / (np.max(np.abs(y)) + 1e-9)
    n = (len(y) - FRAME) // HOP + 1
    if n < 3:
        return 0.0
    lo = max(2, int(sr / F0_MAX))
    hi = min(int(sr / F0_MIN), FRAME // 2)
    out = []
    win = np.hanning(FRAME)
    for i in range(n):
        fr = y[i * HOP:i * HOP + FRAME]
        if np.sqrt(np.mean(fr ** 2) + 1e-12) < 10 ** (min_rms_db / 20):
            continue
        fr = (fr - fr.mean()) * win
        r = np.correlate(fr, fr, mode="full")[FRAME - 1:]
        if r[0] <= 0:
            continue
        r = r / r[0]
        seg = r[lo:hi]
        if len(seg) < 3:
            continue
        k = int(np.argmax(seg)) + lo
        if r[k] < 0.45:
            continue
        out.append(sr / k)
    if not out:
        return 0.0
    return float(np.median(out))


def pairwise_stats(S):
    n = S.shape[0]
    v = S[np.triu_indices(n, 1)]
    v = v[~np.isnan(v)]
    return float(v.mean()), float(np.percentile(v, 10)), float(np.percentile(v, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=r"svc_train\dataset_svc_v4")
    ap.add_argument("--out", default="data/dafeiyu_v4")
    ap.add_argument("--target", type=float, default=0.82, help="平均两两余弦目标")
    ap.add_argument("--min-to-medoid", type=float, default=0.72)
    ap.add_argument("--step", type=float, default=0.06)
    ap.add_argument("--min-keep", type=float, default=0.30)
    ap.add_argument("--min-minutes", type=float, default=60.0)
    ap.add_argument("--max-minutes", type=float, default=110.0)
    ap.add_argument("--val-songs", default="2081,2095")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    src_dir = Path(args.in_dir)
    files = sorted(p for p in src_dir.glob("*.wav") if not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"{src_dir} 里没有 wav")
    print(f"[build] 输入 {len(files)} 个", flush=True)

    model, frontend = load_model()
    names, embs = [], []
    for i, p in enumerate(files, 1):
        try:
            e = embed(model, frontend, p)
        except SystemExit:
            continue
        names.append(p.stem)
        embs.append(e)
        if i % 200 == 0:
            print(f"[build] embedded {i}/{len(files)}", flush=True)
    E = np.stack(embs)
    n = len(names)
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    m, p10, p50 = pairwise_stats(S)
    print(f"[build] 嵌入 {n} 条；初始两两余弦 mean {m:.4f} p10 {p10:.4f} p50 {p50:.4f}", flush=True)

    keep = np.ones(n, dtype=bool)
    idx = np.arange(n)
    while True:
        Sk = S[np.ix_(idx, idx)]
        m, p10, p50 = pairwise_stats(Sk)
        ratio = keep.sum() / n
        print(f"[build] {int(keep.sum())} 条 ({ratio*100:.1f}%)  mean {m:.4f} p10 {p10:.4f} p50 {p50:.4f}", flush=True)
        if m >= args.target or ratio <= args.min_keep:
            break
        mps = np.nanmean(Sk, axis=1)
        k = max(1, int(round(len(idx) * args.step)))
        drop = idx[np.argsort(mps)[:k]]
        keep[drop] = False
        idx = np.where(keep)[0]

    Sk = S[np.ix_(idx, idx)]
    mps = np.nanmean(Sk, axis=1)
    medoid = idx[int(np.argmax(mps))]
    to_med = np.nan_to_num(S[:, medoid], nan=1.0)
    ok = keep & (to_med >= args.min_to_medoid)
    ok[medoid] = True
    idx = np.where(ok)[0]
    print(f"[build] 过 medoid 门槛 {len(idx)} 条  medoid={names[medoid]}", flush=True)

    # 逐段 f0 + 时长（按 44.1k 目标时长估）
    rows = []
    for j, i in enumerate(idx, 1):
        p = src_dir / f"{names[i]}.wav"
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y.mean(1)
        dur = len(y) / sr
        f0 = f0_median(y, sr)
        rows.append({"name": names[i], "song": names[i][:4], "sec": dur, "f0": f0,
                     "mps": float(np.nanmean(S[i, idx])), "to_medoid": float(to_med[i])})
        if j % 300 == 0:
            print(f"[build] f0 {j}/{len(idx)}", flush=True)

    voiced = [r for r in rows if r["f0"] > 0]
    f0s = np.array([r["f0"] for r in voiced])
    print(f"[build] f0 med {np.median(f0s):.1f}  p5 {np.percentile(f0s,5):.1f} "
          f"p95 {np.percentile(f0s,95):.1f}  min {f0s.min():.1f} max {f0s.max():.1f}", flush=True)

    total_min = sum(r["sec"] for r in rows) / 60.0
    print(f"[build] 筛后总时长 {total_min:.1f} 分钟", flush=True)
    if total_min > args.max_minutes:
        rng = np.random.default_rng(args.seed)
        rows_sorted = sorted(rows, key=lambda r: (r["f0"] if r["f0"] > 0 else 1e9))
        nb = 10
        per = len(rows_sorted) / nb
        want = args.max_minutes / total_min
        picked = []
        for b in range(nb):
            seg = rows_sorted[int(b * per):int((b + 1) * per)]
            k = max(1, int(round(len(seg) * want)))
            take = rng.choice(len(seg), size=min(k, len(seg)), replace=False)
            picked += [seg[t] for t in take]
        rows = picked
        total_min = sum(r["sec"] for r in rows) / 60.0
        f0s = np.array([r["f0"] for r in rows if r["f0"] > 0])
        print(f"[build] 分层抽到 {len(rows)} 段 / {total_min:.1f} 分钟；"
              f"f0 med {np.median(f0s):.1f} p5 {np.percentile(f0s,5):.1f} p95 {np.percentile(f0s,95):.1f}",
              flush=True)

    val_songs = [s.strip() for s in args.val_songs.split(",") if s.strip()]
    for split in ("train", "val"):
        (Path(args.out) / split / "audio").mkdir(parents=True, exist_ok=True)

    stats = {"train": {"n": 0, "sec": 0.0}, "val": {"n": 0, "sec": 0.0}}
    for r in rows:
        split = "val" if r["song"] in val_songs else "train"
        y, sr = sf.read(src_dir / f"{r['name']}.wav", dtype="float32", always_2d=True)
        y = y.mean(1)
        if sr != 44100:
            g = np.gcd(int(sr), 44100)
            y = resample_poly(y, 44100 // g, int(sr) // g)
        y = trim(y, 44100)
        if len(y) < 44100 * 1.5:
            continue
        y = normalize(y)
        sf.write(Path(args.out) / split / "audio" / f"{r['name']}.wav", y, 44100, subtype="PCM_16")
        stats[split]["n"] += 1
        stats[split]["sec"] += len(y) / 44100.0
        r["split"] = split

    for split in ("train", "val"):
        print(f"[build] {split}: {stats[split]['n']} 段 / {stats[split]['sec']/60:.1f} 分钟", flush=True)

    with open(Path(args.out) / "selection.json", "w", encoding="utf-8") as f:
        json.dump({"source": str(src_dir), "target_pairwise": args.target,
                   "min_to_medoid": args.min_to_medoid, "val_songs": val_songs,
                   "stats": stats, "items": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
