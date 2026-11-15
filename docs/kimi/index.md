# Kimi 架构总览

月之暗面：权重开源起步较晚但路线明确——Moonlight 先验证 **Muon 优化器**，K2 直接以 DeepSeek-V3 代码库冲到万亿参数，Kimi-Linear 验证混合线性注意力后，**K3 把 Kimi-Linear 架构直接推上 2.8T 旗舰**（首个开源 3T 级模型）。

## 系列与演进

| 系列 | 发布 | 说明 | 架构要点 |
| --- | --- | --- | --- |
| Moonlight-16B-A3B | 2025-02 | Muon 优化器验证模型 | MoE（细粒度），论文重点在训练效率 |
| Kimi-VL-A3B | 2025-04 | 视觉语言 | 不在收录范围 |
| [Kimi-K2](/kimi/kimi-k2/) | 2025-07 | 1T A32B | **DeepSeek-V3 代码库**：MLA + 384+1 专家，MuonClip 训练 |
| Kimi-Linear-48B-A3B | 2025-10 | **混合线性注意力**试验田 | KDA + full attention 混排 |
| Kimi-K2-Thinking | 2025-11 | 推理增强 | K2 架构 + iterative RL |
| K2.5 / K2.6 / K2.7-Code | 2026-01 / 04 / 06 | K2 架构迭代 | 后训练升级 |
| [Kimi-K3](/kimi/kimi-k3/) | 2026-06 | **2.8T A104B** | **Kimi-Linear 上旗舰**：69 层 KDA + 24 层 Gated MLA（不做权重吸收，缓存 per-head 展开结果）+ LatentMoE（896+2 Top-16）+ SiTU 激活 + AttnRes + 原生 MXFP4，1M ctx，自有代码库 |

## 架构特征速览

- **Attention**：K2 全盘 MLA（latent 576，68.6 KiB/token）→ K3 混合化：KDA 线性层（固定 state 221.6 MiB）+ Gated MLA，但 **MLA 分支放弃权重吸收**（cache 存 per-head 展开 K/V，1.41 MiB/token，为 K2 的 ~21×）——用 KV 换 kernel 简化，1M 部署依赖 KV 量化与 P/D 分离。
- **MoE**：K3 的 **LatentMoE** 是结构创新——共享 down/up 投影把路由专家放进 3584 维 latent 空间（每专家 33M，参数省半）；896+2 专家 Top-16 是业界最大专家池 + 最深 Top-k。
- **训练**：MuonClip（K2）延续；K3 原生 MXFP4 权重 + MXFP8 激活 QAT（group 32），部署口径 ≥8×GB300 / 8×MI355X。

## 已收录页面

- [Kimi-K2](/kimi/kimi-k2/)
- [Kimi-K3](/kimi/kimi-k3/)
