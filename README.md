# dafeiyu-svc-train

**造数据 + 训练**那一半。推理与权重在隔壁 **[`dafeiyu-svc`](https://github.com/<you>/dafeiyu-svc)**（MIT）。

> 为什么拆两个仓：造数据要 import **Seed-VC（GPL-3.0）**，一旦合并，推理仓也会被传染成 GPL，
> 别人就不能在闭源项目里依赖它了。拆开以后：推理仓 MIT、本仓 GPL-3.0，谁要哪一半拿哪一半。

## 本仓包含什么

| 文件 | 作用 |
|---|---|
| `scripts/make_list.py` | 从 Opencpop segments 里挑 3–8 秒的段，卡总时长，出清单 |
| `scripts/gen_teacher_seedvc.py` | 用 Seed-VC（**import，本仓就是 GPL 所以没问题**）批量造教师数据 |
| `scripts/gen_teacher_soulx.py` | 同一件事，教师换成 SoulX-Singer-SVC（24 kHz，会封顶高频） |
| `scripts/refeq.py` | **配对频响校正**：逐条对着该条真人原件，把包络搬迁掰回来 |
| `scripts/build_dataset.py` | 两两余弦一致性剪枝 + f0 分层抽样 + 整首留出验证 |
| `scripts/eval_metrics.py` | 质心 / tilt / hi10k / 12–16k / 非谐波细节 / f0 误差 / CAM++ |
| `scripts/speaker_verify.py` | CAM++ 说话人嵌入（`CAMPP_DIR` 环境变量指向模型目录） |
| `configs/dafeiyu_v6.yaml` | v6 的训练配置（改 `data.*_path` 与 `env.expdir` 即可） |

## 端到端（v6 实际走的那条，数字是实测）

```bash
# 0) 依赖
pip install -r requirements.txt

# 1) 选片：3756 段 -> 3540 段 / 304.2 分钟
python scripts/make_list.py --src /data/opencpop/segments/wavs --out clips.txt

# 2) 教师换声：3540 段 / 154.8 分钟（RTF 0.5，模型只加载一次）
python scripts/gen_teacher_seedvc.py --list clips.txt --ref ref.wav \
       --out data/teacher --steps 50 --cfg 0.7

# 3) 配对频响校正（C 档：只修 4k 以上，v6 用的就是这档）
python scripts/refeq.py --src /data/opencpop/segments/wavs \
       --conv data/teacher -o data/teacher_eqc --from 4000 --cap 10 --gate 0.7

# 4) 一致性剪枝：3540 -> 1092 段（均值两两余弦 0.8213）
python scripts/build_dataset.py --in-dir data/teacher_eqc --out /path/to/ddsp/data/dafeiyu_v6 \
       --target 0.82 --min-to-medoid 0.72 --step 0.06 --max-minutes 110 --val-songs 2081,2095

# 5) 训练（在 DDSP-SVC 仓里）
python preprocess.py -c configs/dafeiyu_v6.yaml
python train_reflow.py -c configs/dafeiyu_v6.yaml      # 20k 步，约 50 分钟
```

## 自测

```bash
python tests/test_refeq.py
# 4–6k 占比   源 0.0202  错 0.0601  校正后 0.0225
# 4–6k 误差   0.0399 -> 0.0023
# 10 带 L1    0.8770 -> 0.4770
# SELFTEST: PASS
```

这条自测不碰真实数据：造一个"低频抽空 + 中频堆高 + 4–6k 亮 8 dB"的已知错误，
看校正能不能把它掰回来。**它只验证 ≥4 kHz 那一段**，低频不动是 C 档的设计（D 档用 `--from 2000`）。

## 许可

- 本仓代码：**GPL-3.0**（见 `LICENSE`），因为直接链接 Seed-VC。
- 训练数据 Opencpop：**non-commercial research and education only**。
- 因此训练出来的**权重**只能 CC BY-NC 4.0——权重本身发布在推理仓。
- 第三方组件许可见 `NOTICE.md`。
