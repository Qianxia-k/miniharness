# MiniHarness

一个以教学为导向的迷你编码代理框架 —— 持续更新中。

目前架构概览

用户输入 → 会话历史 → LLM 调用（流式）→ 工具执行 → 循环直到最终答案

主要特性

- 异步流式代理循环 — 实时令牌输出并可在流中拦截工具调用
- 多提供商支持 — DashScope（Qwen）、OpenAI 及兼容 API，自动检测提供商
- 内置工具 — `read_file`、`write_file`、`edit_file`、`grep`、`bash`（使用 Pydantic 校验输入）
- 可选的 Docker 沙箱，用于隔离执行 bash 命令
- 分层配置（默认 → 环境变量 → 自动检测 → CLI 覆盖）
- 写入/执行命令时的交互式权限提示
- LLM 调用的鲁棒重试（指数退避 + 抖动）

快速开始

1. 克隆并安装

```bash
git clone <repo-url> && cd miniharness
uv sync --extra dev
```

2. 准备凭证

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY 或 OPENAI_API_KEY
```

3. 运行代理

```bash
uv run mh "explain this project"
uv run mh --dry-run "test"          # 显示解析后的配置
uv run mh --sandbox "list files"    # 在 Docker 沙箱中运行命令
uv run mh -m gpt-4.1-mini "..."     # 覆盖模型
```

配置说明

设置按层次解析（低优先级 → 高优先级）：默认值、环境变量、提供商自动检测、CLI 参数。

常用环境变量

- `DASHSCOPE_API_KEY` — Qwen（DashScope）
- `OPENAI_API_KEY` — OpenAI
- `MINIHARNESS_PROFILE` — 强制指定提供商配置
- `MINIHARNESS_MODEL` — 覆盖模型名
- `MINIHARNESS_MAX_TURNS` — 最大循环回合（默认：8）
- `MINIHARNESS_SANDBOX_ENABLED` — 启用 Docker 沙箱
- `MINIHARNESS_SANDBOX_IMAGE` — 沙箱使用的 Docker 镜像

CLI 摘要

```text
uv run mh [PROMPT] [OPTIONS]

  --profile       Provider profile (qwen, openai, openai-compatible)
  --model, -m     Override model name
  --base-url      Override API base URL
  --max-turns     Maximum agent loop turns
  --sandbox       Enable Docker sandbox
  --sandbox-image Docker image for sandbox
  --dry-run       Show resolved settings and exit
  --cwd           Working directory (default: current directory)
```

项目结构（概览）

```
src/miniharness/     # 应用代码
tests/               # pytest 测试
docs/                # 架构文档
.env.example         # 环境变量模板
pyproject.toml       # 依赖与构建配置
```

运行测试

```bash
uv run pytest -v
```

设计要点

- 使用设置对象而非直接读取环境变量 —— 所有模块从共享的 `Settings` 实例读取配置。
- 使用 Pydantic 在工具边界验证输入并生成 schema。
- 所有文件操作受限于工作区边界；启用沙箱时进一步以容器隔离。

许可证

MIT

