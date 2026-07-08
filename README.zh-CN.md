# MiniHarness

MiniHarness 是一个 local-first 的 coding agent harness，用来学习、验证和迭代
接近真实工程的 agent 基础设施。它保持代码规模可读，但不是只靠 prompt 拼出来
的 demo：当前项目已经包含共享 runtime、结构化工具 schema、权限、hooks、MCP、
会话、记忆、上下文压缩、plugins、skills、后台任务，以及共用同一条后端流水线
的 CLI / TUI 前端。

English: [README.md](./README.md)

## 它做什么

MiniHarness 的运行流程：

```text
用户输入
  -> shared RuntimeController
  -> slash command 或 AgentLoop
  -> 动态 system prompt + 压缩后的会话上下文
  -> OpenAI-compatible 流式模型调用
  -> ToolRegistry 分发工具
  -> permission / hook 检查
  -> 工具结果 / 任务通知 / 记忆更新
  -> 保存会话 + 发出 UI state snapshot
```

CLI 和 TUI 只是前端。它们把用户输入交给同一个 runtime controller，并消费同一套
runtime events，所以 sessions、permissions、tools、memory、compaction、tasks、
plugin state 这些能力都属于后端 pipeline，而不是在 UI 里重复实现。

## 当前能力

- OpenAI-compatible 流式 chat completions。
- Qwen/DashScope、OpenAI 和兼容接口 provider profile。
- CLI/TUI 共享 runtime，支持结构化事件、state snapshots、task snapshots、
  permission prompts 和 user-question prompts。
- 内置工具使用 Pydantic 校验，并暴露为模型 tool schema。
- 文件、搜索、shell、LSP、web fetch、todo、task、memory、plan-mode、多 agent
  等工具。
- 支持 stdio / HTTP MCP server，包括按项目目录自适应 filesystem roots。
- Tool registry 层对未激活 plugin 工具和 MCP adapters 做 gating。
- 权限模式、敏感路径保护、hook presets 和审计日志。
- 会话保存、列表、恢复、tag 和隔离切换。
- 基于 `tiktoken` 的 token 估算、上下文预算展示和分层压缩。
- Core、semantic、episodic 和 session memory。
- 从 bundled、project、user、plugin 多来源发现 plugins 和 skills。
- 后台 shell task、delegated agent task，以及 coordinator result draining。
- 可选 Docker sandbox 执行 shell。

## 快速开始

```bash
git clone <repo-url>
cd miniharness
uv sync --extra dev
cp .env.example .env
```

设置一个 API 凭证：

```text
DASHSCOPE_API_KEY
OPENAI_API_KEY
MINIHARNESS_API_KEY
```

运行：

```bash
uv run mh "inspect this project"
uv run mh
uv run mh --tui
uv run mh --cwd /path/to/project "explain the codebase"
uv run mh --continue
uv run mh --resume <session-id-or-tag>
uv run mh --sessions
uv run mh --dry-run "check config"
```

## CLI 选项

```text
uv run mh [PROMPT] [OPTIONS]

--cwd                    工具工作目录和 `${cwd}` 展开根目录
--profile                provider profile
--model, -m              覆盖模型名
--base-url               覆盖 API base URL
--dry-run                打印解析后的配置并退出
--max-turns              最大 agent loop 轮数
--context-budget-ratio   触发压缩前的软上下文预算比例
--temperature            sampling temperature
--top-p                  nucleus sampling 阈值
--max-tokens             最大输出 token
--sandbox / --no-sandbox 启用或关闭 Docker sandbox
--sandbox-image          sandbox 使用的 Docker 镜像
--continue, -c           恢复最近会话
--resume                 按 ID 或 tag 恢复会话
--sessions               列出保存的会话并退出
--tui                    启动 Textual 前端
```

## REPL 命令

```text
/help                 显示命令
/exit, /quit, /q      退出
/clear                清空会话历史
/status               显示会话状态
/context              显示当前 runtime system prompt
/summary [n]          显示消息数量和最近历史
/compact [n]          压缩较早的会话历史
/project              显示项目指令
/branch [show|list]   显示 git branch 信息
/commit [message]     查看状态或创建 git commit
/diff [full|staged|head|path] 显示 git diff 输出
/model                显示或切换模型
/turns                显示或设置最大循环轮数
/tokens               显示上下文 token 预算
/permissions          切换或查看权限模式
/temperature          显示或设置 temperature
/top-p                显示或设置 top_p
/max-tokens           显示或设置最大输出 token
/memory               查看 memory
/hooks                查看 hook 配置
/skills               列出 skills
/plugins [name]       列出、查看或切换 plugins
/tools [name] [json]  列出、查看或执行 tools
/agents [name]        列出或查看 delegated agent definitions
/tasks                显示 task snapshots
/mcp                  查看 MCP server 状态
/sessions             列出会话
/resume [id|tag]      恢复会话
/tag <name>           给当前会话打 tag
```

模型或工具正在运行时，slash command 会等当前 turn 返回后处理。CLI 下可用
`Ctrl-C` 中断当前 turn。

## 内置工具

核心工具包括：

```text
read_file       读取文本文件，支持 offset/limit
ls              列出目录
grep            字面量文本搜索
glob            按 glob pattern 匹配文件
git_status      查看 git root、branch、dirty files 和 diff stats
git_diff        查看 unstaged、staged 或 HEAD diff 输出
enter_worktree  创建用于隔离开发的 git worktree
exit_worktree   按路径删除 git worktree
lsp             查看 Python symbols、definitions、references、hover
write_file      创建或覆盖文件
edit_file       精确字符串替换，并在权限确认时展示 diff preview
bash            在工作区或 sandbox 中执行 shell
web_fetch       抓取 URL 并把 HTML 转文本
todo_write      维护当前 todo list
task            维护当前会话的任务列表
task_*          创建、列出、查看、读取输出、停止、更新后台任务
agent           启动 delegated agent task
agent_list      列出 delegated agent definitions
send_message    给 delegated agent task 发送消息
team_*          创建、列出、删除 agent team
ask_user_question 向前端/用户询问信息
sleep           异步等待，不阻塞进程
enter_plan_mode / exit_plan_mode
memory_search / memory_add / memory_log
list_mcp_resources / read_mcp_resource / mcp_auth
```

连接后的 MCP 工具会暴露为：

```text
mcp__<server>__<tool>
```

## 配置

配置解析顺序：

```text
defaults
-> user MCP config
-> project MCP config
-> MINIHARNESS_MCP_SERVERS
-> environment variables
-> provider auto-detection
-> CLI overrides
```

常用环境变量：

```text
MINIHARNESS_PROFILE
MINIHARNESS_MODEL
MINIHARNESS_BASE_URL
MINIHARNESS_MAX_TURNS
MINIHARNESS_CONTEXT_BUDGET_RATIO
MINIHARNESS_TEMPERATURE
MINIHARNESS_TOP_P
MINIHARNESS_MAX_TOKENS
MINIHARNESS_SANDBOX_ENABLED
MINIHARNESS_SANDBOX_IMAGE
MINIHARNESS_ALLOW_PROJECT_PLUGINS
DASHSCOPE_API_KEY
OPENAI_API_KEY
MINIHARNESS_API_KEY
```

## MCP

MCP server 配置来源：

```text
~/.miniharness/mcp.json
<project>/.miniharness/mcp.json
MINIHARNESS_MCP_SERVERS
plugin mcp.json 文件
```

同名 server 下，项目配置覆盖用户配置。推荐工作流是先 `cd` 到目标项目再启动
`mh`；`${cwd}`、`${workspace}`、`${project}`、`${home}` 会在运行时展开。

示例：

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "allowed_directories": ["${cwd}"]
    }
  }
}
```

`allowed_directories` 和 `roots` 都表示 filesystem roots。MCP schema 来自外部
server，所以 MiniHarness 仍然会把 MCP 执行接入工具权限和 hooks。

## Skills 和 Plugins

Skill 是 markdown 指令文件。MiniHarness 会把紧凑的 skill index 注入上下文，
模型需要详细指令时再通过 `skill` 工具按需加载全文。

Skill 来源：

```text
bundled skills
project .miniharness/skills/<name>/SKILL.md
project .claude/skills/<name>/SKILL.md
user ~/.miniharness/skills/<name>/SKILL.md
plugin skills
```

Plugin 发现路径：

```text
~/.miniharness/plugins/<name>/
<project>/.miniharness/plugins/<name>/
```

Plugin 可以贡献：

```text
plugin.json      manifest
skills/          skill definitions
hooks.json       hook definitions
mcp.json         MCP server definitions
agents/          delegated agent definitions
```

使用 `/plugins` 查看、启用或禁用 plugin contributions。

Delegated agent definition 可以放在 `.miniharness/agents/<name>.md` 或 plugin
的 `agents/` 目录中。Frontmatter 可以控制 model、permissions、tools、最大
turn 数、hooks 和 isolation mode：

```markdown
---
name: worker
description: Implementation worker for scoped code changes.
permission_mode: accept-edits
maxTurns: 6
isolation: worktree
---

Make focused changes, run relevant checks, and report files changed.
```

设置 `isolation: worktree` 后，MiniHarness 会在 `.miniharness/worktrees/`
下创建 git worktree，并让 delegated agent 以该 worktree 作为工作目录启动。
后台 task metadata 会记录 `isolation`、`worktree_path` 和 `worktree_branch`。

## 会话、记忆和上下文

会话保存在：

```text
~/.miniharness/sessions/<project-slug>/
```

会话切换会为目标 conversation 创建新的 `AgentLoop`，避免 session id、history、
tool metadata 和保存目标互相污染。

上下文系统每轮都会重建 prompt，内容包括静态指令、运行时信息、项目指令、MCP
状态、已启用 skill、memory、conversation history 和 carryover attachments。
Token 使用量通过 `tiktoken` 估算；超过软预算时，MiniHarness 会执行分层压缩并
向前端发出 compact progress events。

## 权限和 Hooks

权限模式：

```text
default       写文件、shell、未知 mutating 操作需要确认
accept-edits 文件编辑自动允许，shell 仍需确认
bypass        除 hard-denied critical paths 外基本允许
plan          只读模式
```

Hooks 是第二层安全控制，用于危险命令、敏感路径、人类审批和审计日志。审计记录
默认写入 `~/.miniharness/audit/`。

## 项目结构

```text
src/miniharness/cli.py              CLI 入口
src/miniharness/ui/                 TUI protocol、backend host、shared runtime
src/miniharness/loop.py             AgentLoop 编排
src/miniharness/runtime/            runtime events
src/miniharness/state/              observable app state snapshots
src/miniharness/tool_registry.py    tool schemas、gating、execution
src/miniharness/tools/              内置工具
src/miniharness/context/            token budget、carryover、compaction
src/miniharness/sessions/           session persistence 和切换
src/miniharness/services/           LSP、memory extraction、session memory
src/miniharness/mcp/                MCP config、clients、adapters、resources
src/miniharness/skills/             skill discovery 和 loading
src/miniharness/plugins/            plugin discovery 和 contributions
src/miniharness/hooks/              hook events、presets、executor
src/miniharness/swarm/              delegated-agent coordination
src/miniharness/tasks/              background task runtime
src/miniharness/config/             settings 和 path helpers
```

## 验证

```bash
uv run pytest
python3 -m compileall src/miniharness
uv run ruff check .
```

当前测试覆盖 permissions、MCP security、hooks、sessions、memory、token
estimation、compaction events、runtime events、TUI runtime、state snapshots、task
snapshots、background tasks、delegated-agent coordination、tool registry、skills、
plugins、sandbox path validation 和 provider defaults。

## 已知限制

- MiniHarness 面向工程化使用，但仍然是紧凑实现，基础设施还在持续加固。
- Git workflow tools、patch apply、更完整的 LSP backend、provider 热切换、发布
  打包、端到端 dogfood 测试仍是重要下一步。
- Direct MCP tools 连接后会暴露给模型。Plugin-contributed MCP tools 已按 plugin
  激活状态 gating；大型 direct tool set 后续应加入语义级 per-turn tool selection。
- `edit_file` 当前是精确字符串替换，不是完整 patch apply。
- Docker sandbox 需要本机安装 Docker 并可在 `PATH` 里访问。

## 许可证

MIT
