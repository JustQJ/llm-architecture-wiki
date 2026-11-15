# GLM 架构总览

智谱（Z.ai）：从 GLM-130B（2022，Dense）起步，GLM-4.5 起转向**大比例 MoE**（GQA + 自有稳定性设计），GLM-5 完成**双跳换代**——Attention 切到 **MLA + DSA**（维度与腾讯 Hy4 的 gated MLA 一致），上下文冲到 1M；GLM-5.3-Flash 以 `glm5_next` 混合架构预演下一代。

## 系列与演进

| 系列 | 发布 | 说明 | 架构要点 |
| --- | --- | --- | --- |
| GLM-130B | 2022 | 双语开源 Dense 旗舰 | 标准 Transformer，MHA |
| GLM-4（0414 系列） | 2025-04 | 9B / 32B | Dense，GQA |
| [GLM-4.5](/glm/glm-4.5/) | 2025-07 | 357B A32B；Air 106B A12B | **转向 MoE**（160+1），Partial RoPE（半维）、QK-Norm、QKV bias，MTP，原生 128K |
| GLM-4.6 / 4.7 | 2025-09 / 2025-12 | 357B 迭代 | 200K → 上下文与后训练升级，架构不变 |
| GLM-5 / [GLM-5.1](/glm/glm-5.1/) | 2026-02 / 04 | **744B A40B** | **MLA + DSA 换代**（q_lora 2048 / latent 576，top-2048，每层独立 indexer），MoE 沿用 sigmoid+bias（256+1 Top-8），200K ctx；底座昇腾全程训练 |
| [GLM-5.2](/glm/glm-5.2/) | 2026-06 | 同底座 | **IndexShare**：每 4 层一个 full indexer 其余复用 top-k（1M ctx 下 per-token FLOPs ↓2.9×），RoPE theta 8e6，原生 1M |
| [GLM-5.3](/glm/glm-5.3/) | 2026-08 | 同底座 | 官方明说纯后训练升级（coding/agentic 强化），license 收紧；含 **5.3-Flash**（320B A18B，`glm5_next`） |
| GLM-5.3-Flash | 2026-08 | 320B A18B | **34 层 KDA 线性 + 11 层 DSA 混合**、无 RoPE 的 MLA（位置信息靠线性层递推）、MHC 4 残差流 + Sinkhorn、k-pool 索引压缩、288+1 Top-8，首个原生多模态（FP8） |

> 多模态线（GLM-4.5V / 4.6V 等）不在本库收录范围；5.3-Flash 作为 GLM-5.3 页内专节收录。

## 架构特征速览

- **Attention**：GQA + 半维 RoPE（4.5）→ MLA + DSA（5.x，latent 576，KV ≈88 KiB/token）→ 5.2 IndexShare 层间复用 → 5.3-Flash 去 RoPE 化混合架构。
- **MoE**：一路 DeepSeek 式 sigmoid + bias 路由（256+1 Top-8，前 3 层 Dense），5.2 起 router 计算 fp32。
- **Indexer 复用是 1M 上下文的关键杠杆**：GLM-5.2 IndexShare（每 4 层）与腾讯 Hy4 IndexCache（每 5 层）是同一思路的独立演化。

## 已收录页面

- [GLM-4.5](/glm/glm-4.5/)
- [GLM-5.1](/glm/glm-5.1/)
- [GLM-5.2](/glm/glm-5.2/)
- [GLM-5.3（含 5.3-Flash）](/glm/glm-5.3/)
