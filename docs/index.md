---
layout: home

hero:
  name: LLM Architecture Wiki
  text: 主流大模型架构知识库
  tagline: 六家厂商 · 结构图优先 · 每个数字可溯源到 Config 与技术报告
  actions:
    - theme: brand
      text: 从 Qwen3-30B-A3B 开始
      link: /qwen/qwen3-30b-a3b/
    - theme: alt
      text: Attention 变体总览
      link: /guide/attention/

features:
  - icon: 🧭
    title: Qwen（阿里）
    details: GQA → Qwen3 全面 MoE → Qwen3.5 起混合线性注意力铺满全家族，2.4T-A95B 旗舰
    link: /qwen/
  - icon: 🐋
    title: DeepSeek（深度求索）
    details: MLA → 无辅助损失 MoE → DSA 稀疏注意力 → V4 单流 KV + sqrtsoftplus + FP4 专家
    link: /deepseek/
  - icon: ⚡
    title: GLM（智谱）
    details: GLM-4.5 大比例 MoE（GQA）→ GLM-5 切换 MLA + DSA + IndexShare，1M 上下文
    link: /glm/
  - icon: 🌙
    title: Kimi（月之暗面）
    details: K2 万亿 MoE（V3 代码库）→ K3 以 Kimi-Linear 架构冲到 2.8T，LatentMoE + 原生 MXFP4
    link: /kimi/
  - icon: 🔆
    title: MiniMax
    details: Lightning 混合独行 → M2 全 Full Attention → M3 MSA 块级稀疏注意力
    link: /minimax/
  - icon: 🌊
    title: Hunyuan（腾讯）
    details: Hy3 末代 GQA 旗舰（295B/A21B）→ Hy4 一步切换 Gated MLA + DSA + iHC，1M 上下文
    link: /hunyuan/
  - icon: 🧩
    title: Attention 变体
    details: MHA → MQA → GQA → MLA → 稀疏三派（token/块/层间复用）→ 混合线性注意力成主力
    link: /guide/attention/
  - icon: 🎛️
    title: MoE
    details: Router、TopK、共享专家、负载均衡策略与专家并行的完整拆解
    link: /guide/moe/
---
