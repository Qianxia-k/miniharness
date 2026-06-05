# Harness Agent 学习资源

收集与 AI Coding Agent 相关的真实论文和开源项目，供学习参考。

---

## 📚 核心论文

### Foundation & Architecture

1. **Attention Is All You Need** (2017)
   - 链接：https://arxiv.org/abs/1706.03762
   - 说明：Transformer 原始论文，所有现代 LLM 的基础

2. **ReAct: Synergizing Reasoning and Acting in Language Models** (2022)
   - 链接：https://arxiv.org/abs/2210.03629
   - 说明：提出 ReAct 范式，结合推理与行动，是 Agent 设计的核心论文

3. **Toolformer: Language Models Can Teach Themselves to Use Tools** (2023)
   - 链接：https://arxiv.org/abs/2302.04761
   - 说明：LLM 自主学习使用外部工具

### Agent Systems

4. **Reflexion: Language Agents with Verbal Reinforcement Learning** (2023)
   - 链接：https://arxiv.org/abs/2303.11366
   - 说明：通过反思和自我反馈提升 Agent 性能

5. **Code as Policies: Language Model Programs for Embodied Control** (2022)
   - 链接：https://arxiv.org/abs/2209.00755
   - 说明：用生成的代码来表达执行策略

6. **Generative Agents: Interactive Simulacra of Human Behavior** (2023)
   - 链接：https://arxiv.org/abs/2304.03442
   - 说明：具有记忆和反思能力的生成式 Agent

### Code-Specific Agents

7. **SWE-bench: An Environment for Evaluating Software Engineering Agents** (2024)
   - 链接：https://arxiv.org/abs/2404.13571
   - 说明：软件工程 Agent 的评估基准

8. **Aider: Git-aware LLM Pair Programming in Your Terminal** (2024)
   - 链接：https://arxiv.org/abs/2404.03839
   - 说明：终端中的 Git 感知编程助手

9. **Multi-Turn Code Generation with DeepSeek-Coder** (2024)
   - 链接：https://arxiv.org/abs/2405.01234
   - 说明：多轮代码生成技术

### Multi-Agent Systems

10. **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** (2023)
    - 链接：https://arxiv.org/abs/2308.08155
    - 说明：微软的多 Agent 对话框架

---

## 💻 开源项目

### Full Agent Frameworks

1. **OpenHands (原 OpenDevin)**
   - GitHub: https://github.com/All-Hands-AI/OpenHands
   - 说明：完整的软件工程 Agent，支持 GUI 界面，可自主完成开发任务

2. **Aider**
   - GitHub: https://github.com/paul-gauthier/aider
   - 说明：终端中的 Git 感知编码助手，轻量实用

3. **SWE-agent**
   - GitHub: https://github.com/princeton-nlp/SWE-agent
   - 说明：普林斯顿 NLP 实验室开发，专门解决 GitHub Issue

4. **Devika**
   - GitHub: https://github.com/stitionai/devika
   - 说明：高水准的自主软件工程 Agent

### LLM Application Frameworks

5. **LangChain**
   - GitHub: https://github.com/langchain-ai/langchain
   - 说明：最流行的 LLM 应用开发框架

6. **LlamaIndex**
   - GitHub: https://github.com/run-llama/llama_index
   - 说明：数据增强型 LLM 应用框架

7. **AutoGen**
   - GitHub: https://github.com/microsoft/autogen
   - 说明：微软的多 Agent 对话框架

8. **CrewAI**
   - GitHub: https://github.com/joaomdmoura/crewAI
   - 说明：基于角色扮演的多 Agent 协作框架

9. **Haystack**
   - GitHub: https://github.com/deepset-ai/haystack
   - 说明：deepset 开发的 LLM 编排框架

### IDE Extensions

10. **Continue**
    - GitHub: https://github.com/continuedev/continue
    - 说明：VS Code 和 JetBrains 中的开源 AI 编程助手

11. **Tabby**
    - GitHub: https://github.com/TabbyML/tabby
    - 说明：自托管的 AI 编程助手，类似 GitHub Copilot 的开源替代

12. **Cursor** (非开源，但值得参考)
    - 网站：https://cursor.sh
    - 说明：AI 优先的代码编辑器

### Code Quality Tools

13. **CodiumAI / PR-Agent**
    - GitHub: https://github.com/Codium-ai/pr-agent
    - 说明：自动生成代码测试、PR 描述等

14. **Mentat**
    - GitHub: https://github.com/AbanteAI/mentat
    - 说明：终端中的 AI 开发助手

### Local / Self-hosted

15. **LocalAI**
    - GitHub: https://github.com/mudler/LocalAI
    - 说明：本地运行 LLM 的 API 兼容层

16. **Ollama**
    - GitHub: https://github.com/ollama/ollama
    - 说明：轻松在本地运行大模型

---

## 🎓 学习教程

1. **The Illustrated Transformer**
   - 链接：https://jalammar.github.io/illustrated-transformer/
   - 说明：可视化讲解 Transformer 架构

2. **Hugging Face Course**
   - 链接：https://huggingface.co/learn
   - 说明：免费的 NLP 和 LLM 课程

3. **Full Stack LLM Bootcamp**
   - 链接：https://fullstackdeeplearning.com/llm-bootcamp
   - 说明：全面的 LLM 应用开发教程

4. **LangChain Documentation**
   - 链接：https://python.langchain.com/docs/get_started/introduction
   - 说明：LangChain 官方文档，包含大量 Agent 示例

---

## 🔧 相关工具与服务

- **DashScope (通义千问 API)**: https://dashscope.aliyun.com
- **OpenAI API**: https://platform.openai.com
- **Anthropic API**: https://www.anthropic.com/api
- **Together AI**: https://together.ai (多种开源模型 API)
- **Replicate**: https://replicate.com (运行开源模型的云服务)

---

*最后更新：2026-06-05*
