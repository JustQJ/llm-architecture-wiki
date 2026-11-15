# LLM Architecture Wiki

系统整理 Qwen / DeepSeek / GLM / Kimi / MiniMax / Hunyuan 六家厂商的大模型架构，以**结构图**为核心展示 Transformer、Attention、MoE 等内部模块，并记录各代模型的架构演进。

## 内容结构

- 一级：厂商页 —— 模型系列总览、架构演进时间线
- 二级：系列页 —— 同系列各型号列表与架构差异
- 三级：模型页 —— 完整模型结构、Attention / MoE 细节（含 Tensor Shape）、演进对比、原始资料

## 内容与数据规范（重要）

1. **所有数值必须可溯源**：以 `data/models/raw/` 下的原始 HF `config.json`、官方模型卡、技术报告为准，页面底部必须列出来源链接。
2. **架构与版本分离**：同一模型的 BF16 / FP8 / 量化版本共用架构页面；Attention / MoE 结构发生变化则独立成页。
3. **数据抓取脚本化**：`python3 scripts/fetch_hf_configs.py` 按 `scripts/models.json` 清单更新原始配置，人工校对后再进入正式页面，不做未验证的自动写入。

## 技术栈

VitePress + Mermaid（v1）→ Vue Flow 交互式架构图（v2 规划）

## 本地开发

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # 产物在 docs/.vitepress/dist
```

> v1 固定浅色主题（`appearance: false`），保证 Mermaid 架构图渲染一致。

## 目录结构

```
docs/                 # VitePress 站点（按厂商分目录）
data/models/raw/      # HF 原始 config.json（脚本产物，溯源依据）
scripts/              # fetch_hf_configs.py + models.json 清单
.github/workflows/    # GitHub Pages 部署
```

## 路线图

- [x] v1：厂商导航 + 代表模型详情页（Qwen3-30B-A3B / DeepSeek-V3 / GLM-4.5 / Kimi-K2 / MiniMax-M2 / Hunyuan Hy4-Preview）
- [ ] v2：模块点击展开（MLA 低秩压缩、MoE 路由内部计算图）、模型间对比视图
- [ ] v3：定时扫描六家 HF 官方仓库，新模型生成待审核 PR

## License

文档内容 CC BY 4.0；代码 MIT。
