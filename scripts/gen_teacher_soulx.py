"""批量换声：模型只加载一次，把一批 wav 换成人声音色，输出 44.1k 单声道 PCM16。

用法：
  HF_HUB_OFFLINE=1 $PY _svc_batch.py --inputs <目录或通配> --out <目录> \
      --prompt <音色参考 wav> [--limit-min 60] [--min-dur 3 --max-dur 8] [--seed 7]

特点：
  - 断点续跑：输出已存在则跳过（--overwrite 覆盖）
  - f0 提取器与模型都只初始化一次
  - 记录 manifest.json（源/输出/时长/耗时/移调量）
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

REPO = r"D:\dafeiyu\sing\SoulX-Singer-main"
RMVPE = os.path.join(REPO, "pretrained_models", "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt")
CKPT = os.path.join(REPO, "pretrained_models", "SoulX-Singer", "model-svc.pt")


def pick_inputs(pattern, min_dur, max_dur, limit_min, seed):
    if os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, "*.wav")))
    else:
        files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"没有匹配到文件: {pattern}")

    rows = []
    for f in files:
        try:
            info = sf.info(f)
        except Exception:
            continue
        d = info.frames / float(info.samplerate)
        if min_dur <= d <= max_dur:
            rows.append((f, d))
    if not rows:
        raise SystemExit("按 min_dur/max_dur 过滤后一个文件都不剩")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    picked, total = [], 0.0
    for i in order:
        f, d = rows[i]
        if total + d > limit_min * 60.0 and picked:
            continue
        picked.append((f, d))
        total += d
    picked.sort(key=lambda x: x[0])
    print(f"[batch] 候选 {len(rows)} 个，按 {limit_min} 分钟挑中 {len(picked)} 个，共 {total/60:.1f} 分钟", flush=True)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="目录或 glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--limit-min", type=float, default=60.0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # 之后会 chdir 到 SoulX 仓库，先把所有路径绝对化
    args.out = os.path.abspath(args.out)
    args.prompt = os.path.abspath(args.prompt)
    args.inputs = os.path.abspath(args.inputs)

    os.makedirs(args.out, exist_ok=True)
    picked = pick_inputs(args.inputs, args.min_dur, args.max_dur, args.limit_min, args.seed)

    os.chdir(REPO)
    sys.path.insert(0, REPO)

    from soulxsinger.utils.file_utils import load_config
    from soulxsinger.utils.audio_utils import load_wav
    from cli.inference_svc import build_model
    from preprocess.tools.f0_extraction import F0Extractor
    from scipy.signal import resample_poly

    cfg = load_config(os.path.join(REPO, "soulxsinger", "config", "soulxsinger.yaml"))
    sr_out = cfg.audio.sample_rate

    t0 = time.time()
    fx = F0Extractor(RMVPE, device="cuda", verbose=False)
    n_steps = args.steps or cfg.infer.n_steps

    prompt_f0_path = os.path.join(args.out, "_prompt_f0.npy")
    fx.process(args.prompt, f0_path=prompt_f0_path)
    pt_f0 = torch.from_numpy(np.load(prompt_f0_path)).unsqueeze(0).to("cuda").float()
    pt_wav = load_wav(args.prompt, sr_out).to("cuda")
    print(f"[batch] f0 extractor + prompt 就绪 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    model = build_model(model_path=CKPT, config=cfg, device="cuda", use_fp16=True)
    print(f"[batch] model load {time.time()-t0:.1f}s", flush=True)

    manifest = []
    total_audio = total_infer = 0.0
    t_start = time.time()

    for i, (src, dur) in enumerate(picked, 1):
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(args.out, f"{name}.wav")
        if os.path.exists(dst) and not args.overwrite:
            try:
                manifest.append({"src": src, "dst": dst, "dur": sf.info(dst).frames / 44100.0,
                                 "skipped": True})
            except Exception:
                pass
            continue

        f0_path = os.path.join(args.out, "_tmp_f0.npy")
        fx.process(src, f0_path=f0_path)
        g_f0 = torch.from_numpy(np.load(f0_path)).unsqueeze(0).to("cuda").float()
        gt_wav = load_wav(src, sr_out).to("cuda")

        torch.cuda.synchronize()
        ti = time.time()
        with torch.no_grad():
            out, shift = model.infer(
                pt_wav=pt_wav, gt_wav=gt_wav, pt_f0=pt_f0, gt_f0=g_f0,
                auto_shift=False, pitch_shift=0,
                n_steps=n_steps, cfg=cfg.infer.cfg, use_fp16=True,
            )
        torch.cuda.synchronize()
        dt = time.time() - ti

        audio = out.squeeze().float().cpu().numpy()
        up = resample_poly(audio, 44100, sr_out).astype(np.float32)
        peak = float(np.max(np.abs(up))) or 1.0
        up = up / peak * 0.89
        sf.write(dst, up, 44100, subtype="PCM_16")

        sec = len(audio) / sr_out
        total_audio += sec
        total_infer += dt
        manifest.append({"src": src, "dst": dst, "src_dur": round(dur, 3),
                         "dur": round(len(up) / 44100.0, 3), "infer_s": round(dt, 2),
                         "shift": int(shift), "skipped": False})
        if i % 10 == 0 or i == len(picked):
            el = time.time() - t_start
            print(f"[batch] {i}/{len(picked)}  已换 {total_audio/60:.1f}min  用时 {el/60:.1f}min  "
                  f"预计剩余 {el/max(total_audio,1e-9)*(sum(d for _, d in picked)-total_audio)/60:.1f}min",
                  flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt": args.prompt, "n_steps": n_steps,
                   "items": manifest}, f, ensure_ascii=False, indent=1)
    print(f"[batch] 完成 {total_audio/60:.1f} 分钟音频，推理 {total_infer/60:.1f} 分钟，"
          f"总耗时 {(time.time()-t_start)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
