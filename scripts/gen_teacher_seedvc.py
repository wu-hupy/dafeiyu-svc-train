# -*- coding: utf-8 -*-
"""Seed-VC 批量换声：模型只加载一次，循环处理整批片段（避免每条 1 分钟加载）。

用法（用 D:/dafeiyu/sing/venv 的 python，cwd 任意）：
  $PY _seedvc_batch.py --list list.txt --ref dafeiyu_speech --out outdir [--steps 50]
list.txt 每行一个源 wav 绝对路径。
"""
import argparse
import os
import sys
import time
import types

ROOT = r"D:\AI-Song\_seedvc\seed-vc-main"
sys.path.insert(0, ROOT)
sys.path.insert(0, r"D:\AI-Song\_stubs")

REFS = {
    "dafeiyu_speech": r"D:\dafeiyu\sing\voice\dafeiyu-prompt.wav",
    "dafeiyu_sing": r"D:\AI-Song\_three_way\ref\dafeiyu_sing_ref.wav",
}
CKPT = os.path.join(ROOT, "checkpoints",
                    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth")
CONFIG = os.path.join(ROOT, "configs", "presets",
                      "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--ref", default="dafeiyu_speech")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=0.7)
    a = ap.parse_args()

    import inference

    a.list = os.path.abspath(a.list)
    a.out = os.path.abspath(a.out)
    srcs = [l.strip() for l in open(a.list, encoding="utf-8") if l.strip()]
    ref = REFS.get(a.ref, a.ref)
    os.makedirs(a.out, exist_ok=True)
    os.chdir(ROOT)   # 有些模块按相对路径找配置，必须在读文件之后切

    args = types.SimpleNamespace(
        source=srcs[0], target=ref, output=a.out,
        diffusion_steps=a.steps, length_adjust=1.0, inference_cfg_rate=a.cfg,
        f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
        checkpoint=CKPT, config=CONFIG, fp16=True,
    )

    t0 = time.time()
    models = inference.load_models(args)
    inference.fp16 = True
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    # 后续每条直接复用已加载模型
    inference.load_models = lambda _a: models

    done = fail = 0
    t_all = time.time()
    for i, s in enumerate(srcs, 1):
        stem = os.path.splitext(os.path.basename(s))[0]
        dst = os.path.join(a.out, stem + ".wav")
        if os.path.exists(dst):
            done += 1
            continue
        args.source = s
        args.output = a.out
        t1 = time.time()
        try:
            inference.main(args)
        except Exception as e:
            print(f"[{i}/{len(srcs)}] FAIL {stem}: {type(e).__name__} {e}", flush=True)
            fail += 1
            continue
        # inference 写出的是 vc_<src>_<ref>_1.0_<steps>_<cfg>.wav，改名成 <stem>.wav
        name = f"vc_{stem}_{os.path.basename(ref).split('.')[0]}_1.0_{a.steps}_{a.cfg}.wav"
        p = os.path.join(a.out, name)
        if os.path.exists(p):
            os.replace(p, dst)
            done += 1
        else:
            fail += 1
        if i % 10 == 0:
            el = time.time() - t_all
            print(f"[{i}/{len(srcs)}] ok={done} fail={fail} {el/60:.1f}min "
                  f"eta={el/i*(len(srcs)-i)/60:.1f}min", flush=True)
    print(f"[done] ok={done} fail={fail} total={(time.time()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
