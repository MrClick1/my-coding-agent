# SafePatch Agent

一个用于学习编程 Agent 底层机制的轻量项目。当前已经完成文件浏览、代码搜索、单任务状态约束、精确文本修改、修改后测试验证和可复现基准测试，而不是一开始堆叠复杂框架。

代码参考 `agent-tutorial` 等项目的学习思路，但采用独立结构重新实现。

## 当前阶段：可复现的小型基准测试

当前版本已经打通：

```text
自然语言任务
    -> OpenAI 兼容模型
    -> 工具调用（Tool Calling）
    -> list_files / search_code / read_file / replace_text / run_tests
    -> AgentState 记录读取、修改与测试状态
    -> 未验证修改会阻止最终回答
    -> 工具结果回传模型
    -> 最终回答
    -> 隔离 fixture 自动评分与 JSON 报告
```

运行时还会强制执行五项约束：

- 工具只能访问指定工作区，拒绝绝对路径和 `../` 路径越界；
- `.env`、`.git`、私钥和常见凭据文件会被隐藏并拒绝读取；
- 模型调用轮数和工具调用总数都有上限，避免 Agent 无限循环或单轮滥用工具；
- 写入型工具只能修改当前任务已经通过 `read_file` 成功读取的文件；
- 最新一批修改必须调用 `run_tests`，否则 Agent 不能给出最终回答。

当前只提供 `replace_text` 这一种受控写入方式：它不能创建或删除文件，也不会在缺少旧内容校验时直接覆盖整个文件。旧文本的实际出现次数必须与预期完全一致，修改后的文件不能超过 1 MB，并通过同目录临时文件进行原子替换。项目仍不提供任意 Shell。

这些规则是教学项目中的防护层，不等同于操作系统级沙箱。敏感文件名拒绝表也不可能覆盖所有凭据格式，因此不要把包含真实生产密钥的不可信目录直接交给模型。

## 当前可用工具

- `list_files`：列出安全相对目录下的文件和目录；
- `search_code`：在安全相对路径下递归搜索字面量，返回文件路径、行号、列号和命中内容；
- `read_file`：按行读取工作区内的 UTF-8 文本文件；
- `replace_text`：精确替换已读取文件中的字面量，并校验预期替换次数；
- `run_tests`：在工作区执行固定的 `python -m pytest -q`，不接受模型提供的命令参数。

`search_code` 默认不区分大小写，可以限定单个文件或目录；默认最多返回 50 条结果，可在 1 到 100 之间设置。文件工具都会遵守工作区边界与敏感文件过滤，并分别设置结果数、扫描量、读取量或输出大小上限。

`run_tests` 使用当前 Python 环境，固定 120 秒超时和 4 万字符输出上限，并移除常见密钥变量、`PYTHONPATH` 和 pytest 注入变量。测试通过或失败都会被记录；如果测试失败，Agent 可以如实报告失败，但不能声称任务验证成功。测试本身会执行项目中的 Python 代码，这不是操作系统级沙箱，因此只应对你信任的工作区启用。

## 项目结构

```text
src/safe_patch_agent/
├── agent.py          # 工具调用主循环
├── benchmark.py      # 隔离案例运行、自动评分和报告命令行入口
├── cli.py            # 命令行入口
├── config.py         # 环境变量与 .env 配置
├── llm_client.py     # OpenAI 兼容 HTTP 客户端
├── messages.py       # 消息和工具调用数据结构
├── state.py          # 单任务状态、先读后写与测试验证约束
├── tooling.py        # 工具定义、注册和执行
└── workspace.py      # 工作区安全与目录、搜索、读取、替换、测试工具
tests/                # 不需要真实模型的离线测试
benchmarks/           # 版本化案例清单与微型项目 fixture
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

也可以让 Agent 先搜索符号，再阅读相关文件：

```powershell
uv run safe-patch-agent --workspace . "请搜索 AgentError 的定义和引用，并说明错误处理流程"
```

修改任务会强制执行“先读后写”，例如：

```powershell
uv run safe-patch-agent --workspace . "把 README 中的当前阶段标题改为更准确的描述"
```

Agent 必须先调用 `read_file`，随后才能调用 `replace_text`。如果旧文本出现次数与预期不一致，工具会拒绝修改并要求模型重新检查文件。修改成功后还必须调用 `run_tests`，否则主循环会要求模型继续验证，不能直接结束任务。

## 基准测试

仓库自带三个小型案例，分别检查文件读取、符号搜索和“修复代码后运行测试”。每个案例的 fixture 都会先复制到新的临时目录，因此模型的修改不会污染版本化样本。评分器会同时检查：

- Agent 是否正常结束；
- 最终回答是否包含必要信息；
- 模型是否成功调用了指定工具；
- 修改后的目标文件是否与预期精确一致；
- 需要测试的案例是否实际运行且通过测试。

使用已配置的真实模型运行全部案例：

```powershell
uv run safe-patch-benchmark
```

这条命令会产生真实模型 API 调用，可能产生费用，所以普通离线测试不会自动执行它。运行结果会显示逐案例成败和整体成功率，完整 JSON 默认写入 `benchmark-results/latest.json`；该目录已被 Git 忽略。案例格式位于 `benchmarks/cases.json`，可以通过 `--manifest` 和 `--output` 指定其他清单与报告路径。

## 测试

测试使用假的模型客户端，不会发起网络请求，也不需要 API 密钥：

```powershell
uv run pytest
uv run ruff check .
```

## 逐步实现路线

1. **已完成：只读工具调用闭环**——模型可列目录、读文件并回答；
2. **已完成：代码搜索**——加入 `search_code`，模型可按关键词在工作区内定位代码；
3. **已完成：运行状态约束**——加入 `AgentState`，写入型工具执行前必须先读取目标文件；
4. **已完成：精确文本修改**——加入 `replace_text`，校验替换次数并使用原子写入；
5. **已完成：修改后测试**——加入固定的 `run_tests`，未验证修改会阻止最终回答；
6. **已完成：小型基准测试**——隔离运行版本化 fixture，自动检查回答、工具轨迹、文件结果和测试状态，并输出成功率报告；
7. 下一步可增加补丁预览与用户确认，让高风险修改在写入前可审阅。

第一版暂不加入 RAG、MCP、多智能体、网页、长期记忆或自动提交 Git。
