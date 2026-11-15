# Hunyuan 架构总览

腾讯混元文本线：Hunyuan-Large（2024）试水超大规模 MoE（GQA + CLA）→ Hunyuan-A13B（2025-06）轻量稀疏激活 + 中小 Dense 家族 → **Hy3**（2026-07）上一代范式的高完成度收官（GQA + sigmoid 路由 + MTP，295B/A21B）→ **Hy4-preview**（2026-08）直接切换到 Gated MLA + DSA 稀疏注意力 + iHC 残差的前沿组合，原生 1M 上下文——两个月内的单代际最大跳跃。均 Apache-2.0 开源。

## 系列与演进

| 系列 | 发布 | 说明 | 架构要点 |
| --- | --- | --- | --- |
| Tencent-Hunyuan-Large | 2024-10（HF）/ 2024-11 发布 | 389B A52B | 首个开源 MoE 旗舰之一；**GQA + CLA 跨层 KV 共享**；28K 上下文（RoPE 扩展 128K）；每 token 1 shared + 1 routed |
| Hunyuan-A13B | 2025-06 | 80B A13B | 细粒度 MoE，GQA，原生 256K；fast / slow thinking 双模 |
| Hunyuan 0.5B / 1.8B / 4B / 7B | 2025-07 | Dense 家族 | 标准 Dense Transformer（Instruct / Pretrain / 量化版） |
| Hunyuan-MT-7B | 2025-08 | 翻译特化 | Dense |
| [Hy3](/hunyuan/hy3/) | 2026-07（Preview 2026-04） | **295B A21B**（+MTP 3.8B） | **末代 GQA 旗舰**：64Q/8KV + QK-Norm，192+1 专家 Top-8（sigmoid + bias），256K，KV 320 KiB/token |
| [Hy4-preview](/hunyuan/hy4-preview/) | 2026-08（HF）/ 2026-09（官网） | 770B A49B（+MTP 10B） | **Gated MLA + DSA**（top-2048 + IndexCache 跨层索引复用）+ 256+1 专家 + iHC 残差 + 1M 上下文 |
| Hy 生态衍生 | 2026-07~10 | Hy-MT2 / Memory / SimulMT / Nomas（30B-A3B 级） | 翻译 / 长期记忆写入 / 同传等专用模型，独立产品线 |

> 视觉 / 视频 / 3D 线（HunyuanVideo、HunyuanImage、Hunyuan3D 等）不在本库收录范围。

## 架构特征速览

- **Attention**：GQA（Large 叠加 CLA → Hy3 仍是 64Q/8KV + QK-Norm，KV 320 KiB/token）→ Hy4 一步到位 `use_mla + use_dsa`：每 5 层一个 full indexer、其余层共享（`indexer_types`），Gated MLA（fp32 elementwise 门控 + learnable sink），KV 降到 ≈88–95 KiB/token（**3.4× 压缩**），细节见模型页。
- **MoE**：细粒度 + sigmoid + bias 路由贯穿（Hy3：192+1 Top-8 dim 1536，scaling 2.826；Hy4：256+1 Top-8 dim 2048，scaling 2.827——与 Kimi-K2 几乎同值）；config 无 `e_score_correction_bias`。
- **残差**：Hy3 常规加法；Hy4 用 iHC（4 条并行残差流）替代。
- **长上下文**：Hy3 原生 256K → Hy4 原生 1M。

## 已收录页面

- [Hy3](/hunyuan/hy3/)
- [Hy4-preview](/hunyuan/hy4-preview/)
