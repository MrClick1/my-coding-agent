"""支持持续聊天和人工确认修改的 Agent 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from safe_patch_agent.agent import (
    AgentError,
    AgentEvent,
    AgentEventKind,
    CodingAgent,
    CodingSession,
)
from safe_patch_agent.changes import ChangeJournal, ChangeKind
from safe_patch_agent.config import ConfigurationError, LLMConfig
from safe_patch_agent.llm_client import LLMError, OpenAICompatibleClient
from safe_patch_agent.workspace import (
    CreationPreview,
    DeletionPreview,
    ReplacementPreview,
    RollbackPreview,
    SafeWorkspace,
    WorkspaceError,
    build_agent_registry,
)

CHAT_HELP = """可用会话命令：
  /help            显示这份帮助
  /changes         查看当前会话的修改日志与测试状态
  /rollback        回滚最近一次修改（先展示反向差异并确认）
  /rollback <编号> 回滚指定修改；同一文件有更新修改时需先回滚更新项
  /rollback all    回滚当前会话的全部活动修改
  /clear           清除对话历史；修改日志和待验证修改不会被清除
  /exit            退出持续会话

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
            "运行具备持续聊天、代码搜索、人工确认文件创建/精确替换/删除和固定测试能力的编程 Agent。"
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
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用模型 SSE 流式响应；仍显示工具执行进度",
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


def request_creation_approval(
    preview: CreationPreview,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """完整展示新文件 diff，并读取用户的明确批准。"""

    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stderr if output_stream is None else output_stream
    if not input_stream.isatty():
        print("无法确认创建：当前输入不是交互式终端。", file=output_stream)
        return False

    print("\n=== SafePatch 新文件预览 ===", file=output_stream)
    print(
        f"文件：{preview.path}；大小：{preview.updated_bytes} 字节",
        file=output_stream,
    )
    print(preview.diff, end="", file=output_stream)
    print("创建以上文件？[y/N]：", end="", file=output_stream, flush=True)
    try:
        answer = input_stream.readline()
    except (OSError, KeyboardInterrupt):
        print("\n未获得确认，创建已取消。", file=output_stream)
        return False
    approved = answer.strip().casefold() in {"y", "yes", "是"}
    if not approved:
        print("创建未获批准，已取消。", file=output_stream)
    return approved


def request_deletion_approval(
    preview: DeletionPreview,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """完整展示删除 diff，并读取用户的明确批准。"""

    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stderr if output_stream is None else output_stream
    if not input_stream.isatty():
        print("无法确认删除：当前输入不是交互式终端。", file=output_stream)
        return False

    print("\n=== SafePatch 文件删除预览 ===", file=output_stream)
    print(
        f"文件：{preview.path}；大小：{preview.original_bytes} -> 0 字节",
        file=output_stream,
    )
    print(preview.diff, end="", file=output_stream)
    print("删除以上文件？[y/N]：", end="", file=output_stream, flush=True)
    try:
        answer = input_stream.readline()
    except (OSError, KeyboardInterrupt):
        print("\n未获得确认，删除已取消。", file=output_stream)
        return False
    approved = answer.strip().casefold() in {"y", "yes", "是"}
    if not approved:
        print("删除未获批准，已取消。", file=output_stream)
    return approved


def request_rollback_approval(
    preview: RollbackPreview,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """展示完整反向 diff，并要求用户明确批准回滚。"""

    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stderr if output_stream is None else output_stream
    if not input_stream.isatty():
        print("无法确认回滚：当前输入不是交互式终端。", file=output_stream)
        return False

    ids = ", ".join(f"#{change_id}" for change_id in preview.change_ids)
    print("\n=== SafePatch 回滚预览 ===", file=output_stream)
    print(
        f"修改：{ids}；文件：{', '.join(preview.paths)}；"
        f"总大小：{preview.original_bytes} -> {preview.updated_bytes} 字节",
        file=output_stream,
    )
    if preview.deleted_paths:
        print(
            f"将删除本会话创建的文件：{', '.join(preview.deleted_paths)}",
            file=output_stream,
        )
    if preview.created_paths:
        print(
            f"将恢复此前删除的文件：{', '.join(preview.created_paths)}",
            file=output_stream,
        )
    print(preview.diff, end="", file=output_stream)
    print("应用以上回滚？[y/N]：", end="", file=output_stream, flush=True)
    try:
        answer = input_stream.readline()
    except (OSError, KeyboardInterrupt):
        print("\n未获得确认，回滚已取消。", file=output_stream)
        return False
    approved = answer.strip().casefold() in {"y", "yes", "是"}
    if not approved:
        print("回滚未获批准，已取消。", file=output_stream)
    return approved


def format_change_log(journal: ChangeJournal) -> str:
    """把修改日志整理为紧凑的中文列表。"""

    summaries = journal.summaries()
    if not summaries:
        return "当前会话还没有成功写入的修改。"
    lines = ["当前会话修改日志："]
    for item in summaries:
        if item.rolled_back:
            status = "已回滚"
        elif item.test_passed is True:
            status = "测试通过"
        elif item.test_passed is False:
            status = "测试失败"
        else:
            status = "待测试"
        before_hash = (
            item.before_sha256[:10]
            if item.before_sha256 is not None
            else "（不存在）"
        )
        after_hash = (
            item.after_sha256[:10]
            if item.after_sha256 is not None
            else "（不存在）"
        )
        kind = {
            ChangeKind.CREATE: "创建",
            ChangeKind.REPLACE: "替换",
            ChangeKind.DELETE: "删除",
        }[item.kind]
        detail = {
            ChangeKind.CREATE: "创建文件",
            ChangeKind.REPLACE: f"替换 {item.replacements} 处",
            ChangeKind.DELETE: "删除文件",
        }[item.kind]
        lines.append(
            f"  #{item.change_id} [{status}][{kind}] {item.path}；"
            f"SHA-256 {before_hash} -> {after_hash}；{detail}"
        )
    pending_paths = journal.pending_rollback_paths
    if pending_paths:
        lines.append(f"回滚结果待测试：{', '.join(pending_paths)}")
    return "\n".join(lines)


def parse_rollback_target(command: str) -> int | str | None:
    """解析 `/rollback` 的可选编号或 all 参数。"""

    parts = command.split()
    if len(parts) == 1:
        return None
    if len(parts) != 2:
        raise ValueError("用法：/rollback [修改编号|all]")
    if parts[1].casefold() == "all":
        return "all"
    try:
        change_id = int(parts[1])
    except ValueError as exc:
        raise ValueError("回滚目标必须是修改编号或 all") from exc
    if change_id < 1:
        raise ValueError("修改编号必须是正整数")
    return change_id


class CLIProgressRenderer:
    """把 Agent 进度事件渲染为不泄露工具结果正文的终端输出。"""

    def __init__(self, output_stream: TextIO) -> None:
        self.output_stream = output_stream
        self.final_answer_streamed = False
        self._text_rounds: set[int] = set()
        self._line_open = False

    def __call__(self, event: AgentEvent) -> None:
        if event.kind is AgentEventKind.MODEL_START:
            self.final_answer_streamed = False
            self._ensure_newline()
            print(
                f"\n[模型] 第 {event.round_number} 轮生成中...",
                file=self.output_stream,
                flush=True,
            )
            return
        if event.kind is AgentEventKind.TEXT_DELTA:
            text = event.text or ""
            if event.round_number not in self._text_rounds:
                print("\nAgent> ", end="", file=self.output_stream, flush=True)
                self._text_rounds.add(event.round_number)
                self._line_open = True
            print(text, end="", file=self.output_stream, flush=True)
            if text:
                self._line_open = not text.endswith(("\n", "\r"))
            return
        if event.kind is AgentEventKind.MODEL_COMPLETE:
            self._ensure_newline()
            self.final_answer_streamed = (
                event.has_tool_calls is False
                and event.round_number in self._text_rounds
            )
            return
        if event.kind is AgentEventKind.TOOL_START:
            self._ensure_newline()
            print(
                f"[工具] 开始 {event.tool_name}",
                file=self.output_stream,
                flush=True,
            )
            return
        if event.kind is AgentEventKind.TOOL_COMPLETE:
            self._ensure_newline()
            if event.succeeded is True:
                status = "成功"
            elif event.succeeded is False:
                status = "失败"
            else:
                status = "状态未知"
            duration = (
                f"，{event.duration_seconds:.3f} 秒"
                if event.duration_seconds is not None
                else ""
            )
            print(
                f"[工具] 完成 {event.tool_name}（{status}{duration}）",
                file=self.output_stream,
                flush=True,
            )
            return
        if event.kind is AgentEventKind.VERIFICATION_REQUIRED:
            self.final_answer_streamed = False
            self._ensure_newline()
            print(
                "[验证] 检测到尚未测试的修改，继续请求模型运行固定测试。",
                file=self.output_stream,
                flush=True,
            )

    def _ensure_newline(self) -> None:
        if self._line_open:
            print(file=self.output_stream, flush=True)
            self._line_open = False


def run_chat(
    session: CodingSession,
    *,
    change_journal: ChangeJournal | None = None,
    workspace: SafeWorkspace | None = None,
    stream: bool = True,
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
        if command == "/changes":
            if change_journal is None:
                print("当前会话未启用修改日志。", file=output_stream)
            else:
                print(format_change_log(change_journal), file=output_stream)
            continue
        if command.split(maxsplit=1)[0] == "/rollback":
            if change_journal is None or workspace is None:
                print("当前会话未启用安全回滚。", file=output_stream)
                continue
            try:
                target = parse_rollback_target(goal)
                rollback = workspace.rollback_changes(target)
                session.mark_external_modifications(rollback["paths"])
            except (WorkspaceError, ValueError) as exc:
                print(f"回滚失败：{exc}", file=output_stream)
                continue
            ids = ", ".join(f"#{item}" for item in rollback["change_ids"])
            print(
                f"已回滚修改 {ids}；回滚后的文件尚未测试。",
                file=output_stream,
            )
            continue
        if goal.startswith("/"):
            print(f"未知会话命令：{goal}。输入 /help 查看命令。", file=output_stream)
            continue

        try:
            progress = CLIProgressRenderer(output_stream)
            result = session.run(
                goal,
                event_handler=progress,
                stream=stream,
            )
        except (AgentError, LLMError, ValueError) as exc:
            print(f"\n本轮错误：{exc}", file=output_stream)
            continue
        except KeyboardInterrupt:
            print("\n本轮已中断，可以继续输入任务。", file=output_stream)
            continue
        if not progress.final_answer_streamed:
            print(f"\nAgent> {result.answer}", file=output_stream)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = LLMConfig.load(args.env_file)
        interactive = args.chat or args.goal is None
        change_journal = ChangeJournal() if interactive else None
        workspace = SafeWorkspace(
            args.workspace,
            replacement_approval=request_replacement_approval,
            creation_approval=request_creation_approval,
            deletion_approval=request_deletion_approval,
            change_journal=change_journal,
            rollback_approval=request_rollback_approval,
        )
        registry = build_agent_registry(workspace)
        client = OpenAICompatibleClient(config)
        agent = CodingAgent(
            client=client,
            registry=registry,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
        )
        if interactive:
            return run_chat(
                CodingSession(agent),
                change_journal=change_journal,
                workspace=workspace,
                stream=not args.no_stream,
                initial_goal=args.goal,
            )
        progress = CLIProgressRenderer(sys.stdout)
        result = agent.run(
            args.goal,
            event_handler=progress,
            stream=not args.no_stream,
        )
    except (ConfigurationError, WorkspaceError, LLMError, AgentError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if not progress.final_answer_streamed:
        print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
