"""支持人工确认修改的 Agent 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from safe_patch_agent.agent import AgentError, CodingAgent
from safe_patch_agent.config import ConfigurationError, LLMConfig
from safe_patch_agent.llm_client import LLMError, OpenAICompatibleClient
from safe_patch_agent.workspace import (
    ReplacementPreview,
    SafeWorkspace,
    WorkspaceError,
    build_agent_registry,
)


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
        description="运行具备代码搜索、人工确认精确替换和固定测试能力的编程 Agent。",
        add_help=False,
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"
    parser.add_argument("goal", help="交给 Agent 的自然语言任务")
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
        result = CodingAgent(
            client=client,
            registry=registry,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
        ).run(args.goal)
    except (ConfigurationError, WorkspaceError, LLMError, AgentError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
