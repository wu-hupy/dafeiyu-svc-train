"""按 v6 的口径挑片，生成一份"每行一个绝对路径"的清单。

规则（和 v6 完全一致，改之前想清楚为什么）：
  - 只要 3–8 秒的段；
  - 按 id 排序后用 seed 7 置换；
  - 卡总时长上限（默认 320 分钟），超了就跳过下一条而不是截断。

用法：
    python scripts/make_list.py --src /path/to/opencpop/segments/wavs --out clips.txt
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Opencpop segments/wavs 目录")
    ap.add_argument("--out", required=True, help="输出清单路径（UTF-8，无 BOM）")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=8.0)
    ap.add_argument("--limit-min", type=float, default=320.0, help="总时长上限（分钟）")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*.wav")))
    rows = []
    for f in files:
        try:
            info = sf.info(f)
        except Exception:
            continue
        d = info.frames / float(info.samplerate)
        if a.min_dur <= d <= a.max_dur:
            rows.append((f, d))

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(rows))
    picked, total = [], 0.0
    for i in order:
        f, d = rows[i]
        if total + d > a.limit_min * 60.0 and picked:
            continue
        picked.append((f, d))
        total += d
    picked.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for f, _ in picked:
            fh.write(f + "\n")

    print(f"candidates {len(rows)}  picked {len(picked)}  {total/60:.1f} min -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
