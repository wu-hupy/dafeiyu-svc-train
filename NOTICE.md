# NOTICE — 组件与数据

本仓代码以 **GPL-3.0** 发布。下列组件/数据**不随本仓分发**，需使用者自行获取。

| 名称 | 许可 | 用途 |
|---|---|---|
| Seed-VC (`Plachtaa/seed-vc`) | **GPL-3.0** | 教师换声（本仓直接 import，故本仓也是 GPL） |
| DDSP-SVC 6.2 (`yxlllc/DDSP-SVC`) | MIT | 训练/推理框架（`configs/dafeiyu_v6.yaml` 用它） |
| ContentVec (`TechInterMezzo/contentvec-500`) | MIT | 内容特征（768 维 / 第 12 层） |
| rmvpe (`lj1995/VoiceConversionWebUI`) | 见上游 | F0 提取 |
| NSF-HiFiGAN 44.1k **2024.02** (openvpi/vocoders) | 见随包 NOTICE | 声码器；2025.02 的 pc 版架构不同，别混用 |
| SoulX-Singer | Apache-2.0 | 备用教师（24 kHz） |
| CAM++ (`iic/speech_campplus_sv_zh-cn_16k-common`) | Apache-2.0 | 身份一致性与剪枝 |
| FlashSR (`YatharthS/FlashSR`) | 见上游 | 带宽扩展（可选） |
| Opencpop (`wenet/opencpop`) | **non-commercial research and education only** | 训练数据来源 |

## 数据边界

- 本仓**不含任何音频、不含任何权重、不含任何参考音**。
- Opencpop 的条款会传递到用它训练出来的权重：**只能非商用**。
- 「大肥鱼」为各自权利人的名称/形象，本项目无关联、无背书。
