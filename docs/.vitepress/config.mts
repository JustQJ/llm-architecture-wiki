import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const zhSearch = {
  translations: {
    button: { buttonText: '搜索', buttonAriaLabel: '搜索文档' },
    modal: {
      noResultsText: '无法找到相关结果',
      resetButtonTitle: '清除查询条件',
      displayDetails: '显示详细列表',
      footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
    }
  }
}

// 厂商导航（与 docs/ 下目录一一对应）
const navVendors = [
  { text: 'Qwen', link: '/qwen/' },
  { text: 'DeepSeek', link: '/deepseek/' },
  { text: 'GLM', link: '/glm/' },
  { text: 'Kimi', link: '/kimi/' },
  { text: 'MiniMax', link: '/minimax/' },
  { text: 'Hunyuan', link: '/hunyuan/' }
]

export default withMermaid(
  defineConfig({
    // GitHub Pages 部署在子路径下；仅 CI 构建时启用，本地 dev / 本地 build 保持根路径
    base: process.env.CI ? '/llm-architecture-wiki/' : '/',
    title: 'LLM Architecture Wiki',
    description:
      '系统整理主流大模型架构：Transformer / Attention / MoE 结构图与架构演进',
    appearance: false, // 架构图以浅色渲染，v1 固定浅色主题保证图形一致性
    lastUpdated: true,
    vite: {
      optimizeDeps: {
        // mermaid ≥11 新增的 CJS 依赖 fastdom（无 ESM 构建），插件内置 include
        // 清单未覆盖，不预打包会在 dev 下报 "does not provide an export named 'default'"
        // mermaid 共引两条：本体 + promised 扩展，都要预打包做 CJS 互操作
        include: ['fastdom', 'fastdom/extensions/fastdom-promised.js']
      }
    },
    mermaid: {
      // 冷启动配置，具体样式见各页面 frontmatter
      theme: 'default'
    },
    themeConfig: {
      nav: [
        { text: '首页', link: '/' },
        { text: '厂商', items: navVendors },
        {
          text: '核心模块',
          items: [
            { text: 'Attention 变体总览', link: '/guide/attention/' },
            { text: 'MoE 总览', link: '/guide/moe/' }
          ]
        }
      ],
      search: { provider: 'local', options: zhSearch },
      outline: { level: [2, 3], label: '本页目录' },
      docFooter: { prev: '上一篇', next: '下一篇' },
      lastUpdated: { text: '最后更新' },
      returnToTopLabel: '回到顶部',
      sidebarMenuLabel: '目录',
      darkModeSwitchLabel: '主题',
      sidebar: {
        '/qwen/': [
          {
            text: 'Qwen（阿里）',
            items: [
              { text: '架构总览', link: '/qwen/' },
              { text: 'Qwen3-30B-A3B', link: '/qwen/qwen3-30b-a3b/' },
              { text: 'Qwen3-235B-A22B', link: '/qwen/qwen3-235b-a22b/' },
              { text: 'Qwen3.5 系列', link: '/qwen/qwen3.5/' },
              { text: 'Qwen3.6', link: '/qwen/qwen3.6/' },
              { text: 'Qwen3.8（2.4T-A95B）', link: '/qwen/qwen3.8/' },
              { text: 'Qwen3.8-Flash-Next', link: '/qwen/qwen3.8-flash-next/' }
            ]
          }
        ],
        '/deepseek/': [
          {
            text: 'DeepSeek（深度求索）',
            items: [
              { text: '架构总览', link: '/deepseek/' },
              { text: 'DeepSeek-V3', link: '/deepseek/deepseek-v3/' },
              { text: 'DeepSeek-V3.2（DSA）', link: '/deepseek/deepseek-v3.2/' },
              { text: 'DeepSeek-V4（Pro/Flash）', link: '/deepseek/deepseek-v4/' },
              { text: 'DeepSeek-V4.1-Flash', link: '/deepseek/deepseek-v4.1-flash/' }
            ]
          }
        ],
        '/glm/': [
          {
            text: 'GLM（智谱）',
            items: [
              { text: '架构总览', link: '/glm/' },
              { text: 'GLM-4.5', link: '/glm/glm-4.5/' },
              { text: 'GLM-5.1', link: '/glm/glm-5.1/' },
              { text: 'GLM-5.2', link: '/glm/glm-5.2/' },
              { text: 'GLM-5.3（含 Flash）', link: '/glm/glm-5.3/' }
            ]
          }
        ],
        '/kimi/': [
          {
            text: 'Kimi（月之暗面）',
            items: [
              { text: '架构总览', link: '/kimi/' },
              { text: 'Kimi-K2', link: '/kimi/kimi-k2/' },
              { text: 'Kimi-K3（Kimi-Linear）', link: '/kimi/kimi-k3/' }
            ]
          }
        ],
        '/minimax/': [
          {
            text: 'MiniMax',
            items: [
              { text: '架构总览', link: '/minimax/' },
              { text: 'MiniMax-M2', link: '/minimax/minimax-m2/' },
              { text: 'MiniMax-M3', link: '/minimax/minimax-m3/' }
            ]
          }
        ],
        '/hunyuan/': [
          {
            text: 'Hunyuan（腾讯）',
            items: [
              { text: '架构总览', link: '/hunyuan/' },
              { text: 'Hunyuan Hy3', link: '/hunyuan/hy3/' },
              { text: 'Hunyuan Hy4-Preview', link: '/hunyuan/hy4-preview/' }
            ]
          }
        ],
        '/guide/': [
          {
            text: '核心模块',
            items: [
              { text: 'Attention 变体总览', link: '/guide/attention/' },
              { text: 'MoE 总览', link: '/guide/moe/' }
            ]
          }
        ]
      }
    }
  })
)
