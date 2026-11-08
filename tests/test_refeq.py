"""refeq 的自测：造一个"已知的频谱搬迁错误"，看校正能不能把它掰回来。

不依赖任何真实数据，也不需要 GPU。跑法：

    python tests/test_refeq.py

判据（全部基于能量占比，与响度无关）：
  1. 4–6 kHz 占比：校正后比"错的那个"更接近干净源；
  2. 10 带的总 L1 距离：校正后必须比校正前小；
  3. 输出必须可读、时长不变。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BANDS = [300, 600, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000]


def make_source(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """类人声：基频 220 Hz 的谐波丛 + 少量噪声 + 幅度包络。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 220.0 * (1.0 + 0.01 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = np.zeros_like(t)
    for k in range(1, 30):
        if f0.mean() * k > 16000:
            break
        x += np.sin(k * phase) / k ** 1.1
    x += 0.02 * rng.standard_normal(x.size)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t - np.pi / 2)
    x *= env
    return (x / np.abs(x).max() * 0.5).astype(np.float64)


def band_share(x: np.ndarray) -> np.ndarray:
    """整段单次 FFT 的分带能量占比（响度无关）。"""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    tot = X[(f >= 300) & (f < 16000)].sum() + 1e-20
    return np.array([X[(f >= a) & (f < b)].sum() / tot for a, b in zip(BANDS[:-1], BANDS[1:])])


def apply_transfer(x: np.ndarray) -> np.ndarray:
    """模拟"包络搬迁"：低频抽空、中频堆高、4–6k 亮一截。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    g = np.ones_like(f)
    g[(f >= 300) & (f < 600)] = 10 ** (-6 / 20)
    g[(f >= 600) & (f < 2000)] = 10 ** (6 / 20)
    g[(f >= 4000) & (f < 6000)] = 10 ** (8 / 20)
    return np.fft.irfft(X * g, n=len(x))


def main() -> int:
    here = Path(__file__).resolve().parent
    script = here.parent / "scripts" / "refeq.py"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src_dir, conv_dir, out_dir = td / "src", td / "conv", td / "out"
        for d in (src_dir, conv_dir):
            d.mkdir()
        src = make_source()
        conv = apply_transfer(src)
        sf.write(src_dir / "0001.wav", src, SR, subtype="PCM_16")
        sf.write(conv_dir / "0001.wav", conv, SR, subtype="PCM_16")

        r = subprocess.run([sys.executable, str(script), "--src", str(src_dir),
                            "--conv", str(conv_dir), "-o", str(out_dir),
                            "--from", "4000", "--cap", "10"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-800:])
            print(r.stderr[-800:])
            print("[] refeq 跑了但失败")
            return 1

        fixed, sr = sf.read(out_dir / "0001.wav")
        assert sr == SR and len(fixed) == len(src), "时长/采样率不符"

        s_src, s_conv, s_fix = band_share(src), band_share(conv), band_share(fixed)
        i46 = BANDS.index(4000)
        d_before = np.abs(s_conv - s_src).sum()
        d_after = np.abs(s_fix - s_src).sum()
        e_before = abs(s_conv[i46] - s_src[i46])
        e_after = abs(s_fix[i46] - s_src[i46])

        print(f"4–6k 占比   源 {s_src[i46]:.4f}  错 {s_conv[i46]:.4f}  校正后 {s_fix[i46]:.4f}")
        print(f"4–6k 误差   {e_before:.4f} -> {e_after:.4f}")
        print(f"10 带 L1    {d_before:.4f} -> {d_after:.4f}")

        ok = (e_after < e_before) and (d_after < d_before)
        print("SELFTEST: PASS" if ok else "SELFTEST: FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
