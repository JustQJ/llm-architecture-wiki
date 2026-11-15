# DeepSeek 架构总览

深度求索：架构创新的输出密度极高——**MLA（V2）、无辅助损失负载均衡 + FP8 训练（V3）、DSA 稀疏注意力（V3.2）、单流 KV + CSA/HCA/SWA 层型三分（V4）** 都在改写行业模板，Kimi-K2 / GLM-4.5 / Hunyuan Hy4 / GLM-5 等大量后续模型直接继承其架构与代码。

## 系列与演进

| 系列 | 发布 | 说明 | 架构要点 |
| --- | --- | --- | --- |
| DeepSeek LLM / Coder | 2024 起 | Dense 7B/67B | 标准 Llama-like |
| DeepSeek-V2 / V2.5 | 2024-05 / 2024-09 | 236B A21B | **首发 MLA**（KV 低秩压缩）+ 轻量 MoE |
| [DeepSeek-V3](/deepseek/deepseek-v3/) | 2024-12 | 671B A37B | 256+1 细粒度专家、无辅助损失负载均衡、MTP、FP8 训练 |
| DeepSeek-R1 | 2025-01 | 推理模型 | V3 底座 + RL |
| DeepSeek-V3.1 | 2025-08 | 权重更新 | 架构不变，长文本增强 |
| [DeepSeek-V3.2](/deepseek/deepseek-v3.2/) | 2025-12（Exp 2025-09） | 671B A37B | **DSA 稀疏注意力**：每层 lightning indexer 选 top-2048 token，注意力 O(S)→O(2048)；骨架与 V3 零差异 |
| [DeepSeek-V4](/deepseek/deepseek-v4/) | 2026-04 | **Pro 1.6T A49B / Flash 284B A13B** | **注意力与 MoE 双换代**：单一 512 维共享 KV 流（kv_heads=1）+ 层型三分 CSA(4)/HCA(128)/SWA；sqrtsoftplus 路由 + 前 3 层 hash_moe、专家 FP4、mHC 4 残差流、Muon、32T tokens，原生 1M |
| [DeepSeek-V4.1-Flash](/deepseek/deepseek-v4.1-flash/) | 2026-09（1017 刷新） | 552B（含 Engram 196B） | **CED 编码器-解码器 + CSA2 跨层 KV 共享 + FP4 主 KV（890 B/token）**，内置 DSpark 投机解码，prefill 8B / decode 16B 双激活口径 |

## 架构特征速览

- **Attention**：MLA（latent 576）→ V3.2 DSA（indexer top-k + 可学习 sink）→ V4 换轨为**单流 KV**（一份 512 维共享 KV，不再是 latent 压缩 + 吸收的 MLA 范式），indexer 移入压缩域且 topk 减半；V4.1 进一步跨层共享（CSA2 Full/Reindex/Reuse 三模式）。
- **MoE**：sigmoid + `e_score_correction_bias`（V3 学派）→ V4 **sqrtsoftplus（√ln(1+eˣ)）** + 前 3 层 hash_moe（冻结查表）+ 专家 FP4，是路由函数的第三个学派节点。
- **训练**：FP8 blockwise + MTP（V3）→ V4 Muon + mHC（Sinkhorn 双随机约束 4 残差流）+ 32T tokens；V4.1 45T。
- **投机解码**：V4.1 起内置 DSpark（3 draft 层 + 独立小 MoE 头），MiniMax M3.1-Flash 也采用 DSpark draft head。

## 已收录页面

- [DeepSeek-V3](/deepseek/deepseek-v3/)
- [DeepSeek-V3.2](/deepseek/deepseek-v3.2/)
- [DeepSeek-V4（Pro/Flash）](/deepseek/deepseek-v4/)
- [DeepSeek-V4.1-Flash](/deepseek/deepseek-v4.1-flash/)
