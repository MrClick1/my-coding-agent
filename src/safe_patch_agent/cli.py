"""支持持续聊天和人工确认修改的 Agent 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from safe_patch_agent.agent import AgentError, CodingAgent, CodingSession
from safe_patch_agent.config import ConfigurationError, LLMConfig
from safe_patch_agent.llm_client import LLMError, OpenAICompatibleClient
from safe_patch_agent.workspace import (
    ReplacementPreview,
    SafeWorkspace,
    WorkspaceError,
    build_agent_registry,
)

CHAT_HELP = """可用会话命令：
  /help   显示这份帮助
  /clear  清除对话历史；待验证修改不会被清除
  /exit   退出持续会话

直接输入自然语言任务并回车即可发送。每轮工具状态独立，修改仍需逐次确认并运行测试。"""


class ChineseArgumentParser(argparse.ArgumentParser):
    """把 argparse 的固定帮助标题和错误前缀转换为中文。"""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法：", 1)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 参数错误：{message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(
        prog="safe-patch-agent",
        description=(
            "运行具备持续聊天、代码搜索、人工确认精确替换和固定测试能力的编程 Agent。"
        ),
        add_help=False,
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"
    parser.add_argument(
        "goal",
        nargs="?",
        help="可选的一次性任务；省略时进入持续聊天",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="允许 Agent 检查的项目目录（默认：当前目录）",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="简化的 KEY=VALUE 配置文件（默认：.env）",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=8,
        help="停止前允许的最大模型调用轮数（默认：8）",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=32,
        help="停止前允许的工具调用总数（默认：32）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="完成可选的初始任务后继续保持聊天会话",
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    return parser


def request_replacement_approval(
    preview: ReplacementPreview,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """在交互式终端完整展示 diff，并读取用户的明确批准。"""

    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stderr if output_stream is None else output_stream
    if not input_stream.isatty():
        print("无法确认修改：当前输入不是交互式终端。", file=output_stream)
        return False

    print("\n=== SafePatch 修改预览 ===", file=output_stream)
    print(
        f"文件：{preview.path}；替换：{preview.replacements} 处；"
        f"大小：{preview.original_bytes} -> {preview.updated_bytes} 字节",
        file=output_stream,
    )
    print(preview.diff, end="", file=output_stream)
    print("应用以上修改？[y/N]：", end="", file=output_stream, flush=True)
    try:
        answer = input_stream.readline()
    except (OSError, KeyboardInterrupt):
        print("\n未获得确认，修改已取消。", file=output_stream)
        return False
    approved = answer.strip().casefold() in {"y", "yes", "是"}
    if not approved:
        print("修改未获批准，已取消。", file=output_stream)
    return approved


def run_chat(
    session: CodingSession,
    *,
    initial_goal: str | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    """运行单行输入的持续聊天循环，直到用户退出或输入结束。"""

    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    if not input_stream.isatty():
        print(
            "错误：持续聊天需要交互式终端；也可以提供 goal 运行一次性任务。",
            file=output_stream,
        )
        return 1

    print("SafePatch Agent 持续会话已启动。输入 /help 查看命令。", file=output_stream)
    pending_goal = initial_goal.strip() if initial_goal is not None else None
    while True:
        if pending_goal is None:
            print("\n你> ", end="", file=output_stream, flush=True)
            try:
                raw_goal = input_stream.readline()
            except (OSError, KeyboardInterrupt):
                print("\n会话已退出。", file=output_stream)
                return 0
            if raw_goal == "":
                print("\n会话已退出。", file=output_stream)
                return 0
            goal = raw_goal.strip()
        else:
            goal = pending_goal
            pending_goal = None

        if not goal:
            continue
        command = goal.casefold()
        if command in {"/exit", "/quit"}:
            print("会话已退出。", file=output_stream)
            return 0
        if command == "/help":
            print(CHAT_HELP, file=output_stream)
            continue
        if command == "/clear":
            session.clear()
            print("会话历史已清除；待验证修改（如有）仍需测试。", file=output_stream)
            continue
        if goal.startswith("/"):
            print(f"未知会话命令：{goal}。输入 /help 查看命令。", file=output_stream)
            continue

        try:
            result = session.run(goal)
        except (AgentError, LLMError, ValueError) as exc:
            print(f"\n本轮错误：{exc}", file=output_stream)
            continue
        except KeyboardInterrupt:
            print("\n本轮已中断，可以继续输入任务。", file=output_stream)
            continue
        print(f"\nAgent> {result.answer}", file=output_stream)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = LLMConfig.load(args.env_file)
        workspace = SafeWorkspace(
            args.workspace,
            replacement_approval=request_replacement_approval,
        )
        registry = build_agent_registry(workspace)
        client = OpenAICompatibleClient(config)
        agent = CodingAgent(
            client=client,
            registry=registry,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
        )
        if args.chat or args.goal is None:
            return run_chat(CodingSession(agent), initial_goal=args.goal)
        result = agent.run(args.goal)
    except (ConfigurationError, WorkspaceError, LLMError, AgentError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
