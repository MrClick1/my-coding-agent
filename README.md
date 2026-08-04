# SafePatch Agent

一个用于学习编程 Agent 底层机制的轻量项目。项目会从只读 Agent 开始，逐步加入代码搜索、精确修改和测试验证，而不是一开始堆叠复杂框架。

代码参考 `agent-tutorial` 等项目的学习思路，但采用独立结构重新实现。

## 当前阶段：只读最小闭环

当前版本已经打通：

```text
自然语言任务
    -> OpenAI 兼容模型
    -> 工具调用（Tool Calling）
    -> list_files / read_file
    -> 工具结果回传模型
    -> 最终回答
```

运行时还会强制执行三项约束：

- 工具只能访问指定工作区，拒绝绝对路径和 `../` 路径越界；
- `.env`、`.git`、私钥和常见凭据文件会被隐藏并拒绝读取；
- 模型调用轮数和工具调用总数都有上限，避免 Agent 无限循环或单轮滥用工具。

这一阶段故意不提供文件写入和任意 Shell，因此 Agent 只能检查、解释代码，不能修改项目。

这些规则是教学项目中的防护层，不等同于操作系统级沙箱。敏感文件名拒绝表也不可能覆盖所有凭据格式，因此不要把包含真实生产密钥的不可信目录直接交给模型。

## 项目结构

```text
src/safe_patch_agent/
├── agent.py          # 工具调用主循环
├── cli.py            # 命令行入口
├── config.py         # 环境变量与 .env 配置
├── llm_client.py     # OpenAI 兼容 HTTP 客户端
├── messages.py       # 消息和工具调用数据结构
├── tooling.py        # 工具定义、注册和执行
└── workspace.py      # 工作区安全与只读工具
tests/                # 不需要真实模型的离线测试
```

## 准备环境

项目使用 uv 管理 Python 3.12、虚拟环境和依赖。首次进入项目后执行：

```powershell
uv sync
```

uv 会根据 `.python-version` 和 `uv.lock` 创建 `.venv`，并安装项目及开发依赖。

复制配置模板并填写你所使用的 OpenAI 兼容服务：

```powershell
Copy-Item .env.example .env
```

```dotenv
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model
LLM_TIMEOUT_SECONDS=60
```

`.env` 不会被 Git 提交；同名系统环境变量的优先级高于 `.env`。
远程 API 地址必须使用 HTTPS；只有 `localhost` 和回环 IP 地址可以使用 HTTP，方便连接本地模型服务。

## 运行

先让 Agent 阅读当前项目：

```powershell
uv run safe-patch-agent --workspace . "请总结这个项目的结构和入口"
```

也可以使用模块入口：

```powershell
uv run python -m safe_patch_agent --workspace . "README 说明了什么？"
```

## 测试

测试使用假的模型客户端，不会发起网络请求，也不需要 API 密钥：

```powershell
uv run pytest
uv run ruff check .
```

## 逐步实现路线

1. **已完成：只读工具调用闭环**——模型可列目录、读文件并回答；
2. 加入 `search_code`，学习更高效地定位代码；
3. 加入 `AgentState` 和“先读后写”约束；
4. 加入 `replace_text`，校验精确替换次数；
5. 加入固定的 `run_tests`，修改后必须验证；
6. 准备小型基准测试，记录真实成功率并完善简历描述。

第一版暂不加入 RAG、MCP、多智能体、网页、长期记忆或自动提交 Git。
