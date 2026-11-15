# Qwen 架构总览

阿里通义系列，开源谱系最完整的模型家族。架构主线清晰：**Dense + GQA（Qwen1-2.5）→ 全面 MoE（Qwen3）→ 混合线性注意力铺满全家族（Qwen3.5 起）**，Attention 侧始终未采用 MLA。Qwen3.5 起全系原生多模态，Qwen3.8-Flash-Next 以 `qwen4_exp` 之名预览下一代架构。

## 系列与演进

| 系列 | 发布 | 代表模型 | 架构要点 |
| --- | --- | --- | --- |
| Qwen / Qwen1.5 | 2023-08 起 | Qwen-14B | Dense，MHA |
| Qwen2 | 2024-06 | Qwen2-72B | Dense，**GQA 铺开**，QKV 带 bias |
| Qwen2.5 | 2024-09 | Qwen2.5-72B | Dense，GQA（末期加大长文本） |
| Qwen3 | 2025-04 | [Qwen3-30B-A3B](/qwen/qwen3-30b-a3b/) · [Qwen3-235B-A22B](/qwen/qwen3-235b-a22b/) | **全面 MoE**（128 选 8，无共享专家），GQA 4 KV head，移除 QKV bias，thinking / non-thinking 双模式 |
| Qwen3-Next | 2025-09 | Qwen3-Next-80B-A3B | **Gated DeltaNet（线性注意力）与 Full Attention 混合层** + MTP，混合路线试验田 |
| [Qwen3.5](/qwen/qwen3.5/) | 2026-02 | 397B-A17B 旗舰 / 122B-A10B / 35B-A3B / 27B / 9B 以下 dense | **混合层铺满全家族**（3:1，full 层为「Gated Attention」变体）+ 原生多模态，softmax 路由 + 门控共享专家 |
| [Qwen3.6](/qwen/qwen3.6/) | 2026-04 | 35B-A3B / 27B | config 与 3.5 逐字段相同，Agentic Coding 训练升级 |
| Qwen3.7 | 2026-05 | Max / Plus（API-only） | HF 无开源仓库 |
| [Qwen3.8](/qwen/qwen3.8/) | 2026-08 | **2.4T-A95B** / 27B | 文本旗舰同构放大（92 层 / hidden 8192），license 改为 qwen3.8-max（非 Apache） |
| [Qwen3.8-Flash-Next](/qwen/qwen3.8-flash-next/) | 2026-08 | 125B（+51B n-gram）/ A6B | `qwen4_exp`：**QSA 块级稀疏 + Gated Residual 4 分支 + n-gram embedding**，Qwen4 预览 |

> 表中日期来自 Hugging Face 官方仓库创建时间。

## 架构特征速览

- **Attention**：一路 GQA（KV head 压到 4 个）→ Qwen3-Next 起转向 Gated DeltaNet + Full Attention 混合，Qwen3.5 起全家族 3:1 混排（full 层在 `idx ≡ 3 mod 4`），full 层升级为 Gated Attention（head_dim 256 + partial RoPE 64 维 + 输出 sigmoid 门）。未用过 MLA。
- **KV Cache**：混合层把长上下文 KV 压到同级 GQA 的 ~1/6（397B-A17B ≈30 KiB/token + 90 MiB 固定 state；2.4T-A95B ≈92 KiB/token + 276 MiB state）。
- **MoE**：Qwen 学派「细粒度 + softmax 归一化路由」贯穿 3.5-3.8（512 选 10 + 1 共享）；与 DeepSeek 系（sigmoid + bias）形成对照。
- **License**：3.5/3.6 Apache-2.0；3.8-2.4T-A95B 与 Flash-Next 换为自有 license（qwen3.8-max / qwen-community）。

## 已收录页面

- [Qwen3-30B-A3B](/qwen/qwen3-30b-a3b/)
- [Qwen3-235B-A22B](/qwen/qwen3-235b-a22b/)
- [Qwen3.5 系列](/qwen/qwen3.5/)
- [Qwen3.6](/qwen/qwen3.6/)
- [Qwen3.8（2.4T-A95B / 27B）](/qwen/qwen3.8/)
- [Qwen3.8-Flash-Next](/qwen/qwen3.8-flash-next/)
