"""只读 Agent 的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from safe_patch_agent.agent import AgentError, CodingAgent
from safe_patch_agent.config import ConfigurationError, LLMConfig
from safe_patch_agent.llm_client import LLMError, OpenAICompatibleClient
from safe_patch_agent.workspace import SafeWorkspace, WorkspaceError, build_read_only_registry


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
        description="运行具备代码搜索能力的只读编程 Agent。",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = LLMConfig.load(args.env_file)
        workspace = SafeWorkspace(args.workspace)
        registry = build_read_only_registry(workspace)
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
