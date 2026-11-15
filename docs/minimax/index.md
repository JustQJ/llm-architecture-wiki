# MiniMax 架构总览

路线最「独」的一家：MiniMax-Text-01 / M1 以 **Lightning Attention（线性注意力）与 Full Attention 按层混合**著称（7:1）；M2 激进回归全 Full Attention；M3 在 GQA 底座上加 **MSA blockwise 稀疏注意力**（第三种稀疏切法），MoE 反转为「少专家 + 共享」。另有独立 H 系列产品线（H3，2026-07）。

## 系列与演进

| 系列 | 发布 | 参数 | 架构要点 |
| --- | --- | --- | --- |
| MiniMax-Text-01 | 2025-01 | 456B A45.9B | Lightning 与 softmax **7:1 按层混合**（80 层中 10 层线性）；32 专家 Top-2，softmax 路由 + aux loss；post-norm（残差 ×3.5565） |
| MiniMax-M1 | 2025-06 | 456B A45.9B | 与 Text-01 **完全同构**（config 仅元数据不同）+ RL，原生 1M 上下文，40k / 80k 变体 |
| [MiniMax-M2](/minimax/minimax-m2/) | 2025-10 | 230B A10B | **全 Full Attention**（GQA 48Q/8KV + per-layer QK-Norm + 半维 RoPE）；256 专家 Top-8，sigmoid + bias 路由；原生 FP8；训练含 3 个 MTP 模块（公开权重未含） |
| M2.1 / M2.5 / M2.7 | 2025-12 / 2026-02 / 04 | 同构迭代 | config 与 M2 基本一致 |
| [MiniMax-M3](/minimax/minimax-m3/) | 2026-06 | **428B A23B** | **MSA blockwise 稀疏**（4 组各选 Top-16×128-token block + local block，层 3-59）；MoE 反转为 **128 选 4 + 1 共享**（专家维度 3072）；SwiGLU-OAI（±7 限幅）、per-head QK-Norm、MTP 7 模块（权重未含），1M ctx 原生多模态 |
| MiniMax-H3 | 2026-07 | — | 独立 H 产品线，不在本次收录范围 |
| M3.1-Flash | 2026-10 | 428B A23B | 同构 + 全 60 层稀疏 + **QAT MXFP8**（router gate 不量化）+ DSpark draft head，对标 DeepSeek V4.1-Flash-1017 |

## 架构特征速览

- **Attention**：混合层模式（7:1）→ M2 全 softmax → M3 稀疏化。**MSA 按块选 token**（block=128，块内 max 池化打分、exp-free top-k），与 DSA（token 级、全层共享头）、Hy4 IndexCache（层间复用）并列三种稀疏切法，取舍分析见 [Attention 变体总览](/guide/attention/)。官方口径 vs M2 @1M：prefill 9× / decode 15×。
- **MoE**：M2 是 256 选 8 无共享；M3 反转为 **128 选 4 + 1 共享**——更少专家、更宽（1536→3072），路由始终 sigmoid + bias。
- **KV Cache**：M3 = 120 KiB/token（BF16，M2 的一半），1M token ≈114 GB。
- **量化**：M2 原生 FP8 → M3.1-Flash QAT MXFP8。

## 已收录页面

- [MiniMax-M2](/minimax/minimax-m2/)
- [MiniMax-M3](/minimax/minimax-m3/)
