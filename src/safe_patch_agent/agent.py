"""SafePatch Coding Agent 的 Tool Calling 主循环。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from safe_patch_agent.llm_client import LLMClient
from safe_patch_agent.messages import ChatMessage
from safe_patch_agent.state import AgentStateSnapshot
from safe_patch_agent.tooling import ToolRegistry

SYSTEM_PROMPT = """你是 SafePatch Agent，一个具备受控文件创建、精确修改、删除和批量变更能力的
编程助手。

工作区工具是你了解项目内容的唯一可靠来源。当任务依赖项目内容时，必须先检查工作区再回答。
所有路径都必须是相对于已配置工作区的相对路径。如果工具返回错误，请修正调用参数，或者明确
说明当前限制。修改文件前必须先使用 read_file 读取目标文件，再使用 replace_text 做精确替换。
只有需要新增且目标尚不存在的 UTF-8 文件时，才使用 create_file 提交完整非空内容；它不要求先读
不存在的文件。只有明确需要移除现有文件时，才在读取完整文件后使用 delete_file。replace_text、
create_file 和 delete_file 都会向用户展示完整差异并要求确认；用户拒绝后不得反复请求相同操作。
每次成功变更后必须运行所有必选验证任务：使用 run_validation 按名称调用，tests 任务也可使用
兼容入口 run_tests。即使验证失败，也要根据真实结果说明。不得声称自己执行了其他任意命令；
当前没有任意 Shell 能力。

需要同时变更多个不同文件时，优先使用 apply_change_set 一次提交创建、精确替换和删除操作，让
用户统一预览和确认。批次中的 replace 和 delete 目标仍必须分别先用 read_file 读取；create 目标
必须不存在；同一路径在一个批次中只能出现一次。批量变更成功后同样必须运行全部必选验证。

需要定位符号、定义或引用时，优先使用 search_code 搜索，再使用 read_file 阅读命中位置的上下文。

请根据你实际检查过的文件，给出简洁、准确的最终回答。
"""

VERIFICATION_REMINDER = """你已经修改了文件，但尚未运行全部必选验证任务。
待运行：{pending}。请调用 run_validation；tests 也可调用兼容入口 run_tests。
在给出最终回答前必须实际执行这些工具；不要仅用文字声称验证已经运行。"""


class AgentError(RuntimeError):
    """Agent 运行时异常的基类。"""


class AgentLoopLimitError(AgentError):
    """模型在循环上限内始终没有给出最终答案。"""


class AgentToolLimitError(AgentError):
    """模型请求的工具调用总数超过上限。"""


class AgentVerificationError(AgentError):
    """Agent 在模型轮次耗尽前没有验证最新修改。"""


class AgentEventKind(StrEnum):
    """供命令行等观察者消费的 Agent 进度事件类型。"""

    MODEL_START = "model_start"
    TEXT_DELTA = "text_delta"
    MODEL_COMPLETE = "model_complete"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    VERIFICATION_REQUIRED = "verification_required"


@dataclass(frozen=True)
class AgentEvent:
    """不包含工具结果正文的轻量运行进度事件。"""

    kind: AgentEventKind
    round_number: int
    text: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    succeeded: bool | None = None
    duration_seconds: float | None = None
    has_tool_calls: bool | None = None


AgentEventHandler = Callable[[AgentEvent], None]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    model_rounds: int
    tool_calls: int
    messages: tuple[ChatMessage, ...]
    state: AgentStateSnapshot


class CodingAgent:
    """在模型与已注册工具之间循环，直到获得最终答案。"""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        *,
        max_rounds: int = 8,
        max_tool_calls: int = 32,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须至少为 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须至少为 1")
        self.client = client
        self.registry = registry
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.system_prompt = system_prompt

    def run(
        self,
        goal: str,
        *,
        event_handler: AgentEventHandler | None = None,
        stream: bool = False,
    ) -> AgentResult:
        """执行一个不继承历史的独立任务。"""

        return self._run(
            goal,
            history=(),
            event_handler=event_handler,
            stream=stream,
        )

    def _run(
        self,
        goal: str,
        *,
        history: tuple[ChatMessage, ...],
        event_handler: AgentEventHandler | None = None,
        stream: bool = False,
    ) -> AgentResult:
        """执行一轮任务，并在系统提示词后附加已压缩的会话历史。"""

        goal = goal.strip()
        if not goal:
            raise ValueError("任务目标不能为空")
        if event_handler is not None and not callable(event_handler):
            raise ValueError("event_handler 必须是可调用对象")
        if not isinstance(stream, bool):
            raise ValueError("stream 必须是布尔值")

        self.registry.state.start_turn()
        messages = [
            ChatMessage.system(self.system_prompt),
            *history,
            ChatMessage.user(goal),
        ]
        tool_call_count = 0

        for round_number in range(1, self.max_rounds + 1):
            self._emit(
                event_handler,
                AgentEvent(AgentEventKind.MODEL_START, round_number),
            )
            stream_complete = getattr(self.client, "stream_complete", None)
            if stream and callable(stream_complete):
                completion = stream_complete(
                    messages,
                    self.registry.schemas(),
                    lambda text, current_round=round_number: self._emit(
                        event_handler,
                        AgentEvent(
                            AgentEventKind.TEXT_DELTA,
                            current_round,
                            text=text,
                        ),
                    ),
                )
            else:
                completion = self.client.complete(messages, self.registry.schemas())
            assistant_message = completion.message
            messages.append(assistant_message)
            self._emit(
                event_handler,
                AgentEvent(
                    AgentEventKind.MODEL_COMPLETE,
                    round_number,
                    has_tool_calls=bool(assistant_message.tool_calls),
                ),
            )

            if completion.finish_reason == "length":
                raise AgentError("模型输出在完整轮次结束前被截断")
            if completion.finish_reason == "content_filter":
                raise AgentError("模型输出被服务提供方的内容过滤器中止")

            if not assistant_message.tool_calls:
                if completion.finish_reason == "tool_calls":
                    raise AgentError("模型报告需要调用工具，但没有返回任何工具调用")
                answer = (assistant_message.content or "").strip()
                if not answer:
                    raise AgentError("模型既没有返回工具调用，也没有给出最终答案")
                if self.registry.state.has_unverified_changes:
                    self._emit(
                        event_handler,
                        AgentEvent(
                            AgentEventKind.VERIFICATION_REQUIRED,
                            round_number,
                        ),
                    )
                    if round_number >= self.max_rounds:
                        raise AgentVerificationError(
                            "Agent 修改了文件，但在模型调用上限内没有运行全部必选验证"
                        )
                    pending = ", ".join(
                        self.registry.state.snapshot().pending_validations
                    )
                    messages.append(
                        ChatMessage.system(
                            VERIFICATION_REMINDER.format(pending=pending)
                        )
                    )
                    continue
                return AgentResult(
                    answer=answer,
                    model_rounds=round_number,
                    tool_calls=tool_call_count,
                    messages=tuple(messages),
                    state=self.registry.state.snapshot(),
                )

            requested_call_count = len(assistant_message.tool_calls)
            if tool_call_count + requested_call_count > self.max_tool_calls:
                raise AgentToolLimitError(
                    f"Agent 已达到 {self.max_tool_calls} 次工具调用上限"
                )
            for call in assistant_message.tool_calls:
                tool_call_count += 1
                self._emit(
                    event_handler,
                    AgentEvent(
                        AgentEventKind.TOOL_START,
                        round_number,
                        tool_name=call.name,
                        tool_call_id=call.id,
                    ),
                )
                started_at = time.monotonic()
                result = self.registry.execute(call)
                self._emit(
                    event_handler,
                    AgentEvent(
                        AgentEventKind.TOOL_COMPLETE,
                        round_number,
                        tool_name=call.name,
                        tool_call_id=call.id,
                        succeeded=self._tool_result_succeeded(result),
                        duration_seconds=round(time.monotonic() - started_at, 3),
                    ),
                )
                messages.append(ChatMessage.tool(call, result))

        raise AgentLoopLimitError(
            f"Agent 已达到 {self.max_rounds} 轮模型调用上限，但仍未获得最终答案"
        )

    @staticmethod
    def _emit(
        event_handler: AgentEventHandler | None,
        event: AgentEvent,
    ) -> None:
        """隔离普通观察者异常，但允许 Ctrl+C 继续中断 Agent。"""

        if event_handler is None:
            return
        try:
            event_handler(event)
        except Exception:
            return

    @staticmethod
    def _tool_result_succeeded(result: str) -> bool | None:
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, Mapping):
            return None
        succeeded = payload.get("ok")
        return succeeded if isinstance(succeeded, bool) else None


@dataclass(frozen=True)
class ConversationTurn:
    """持续会话中保留的一轮用户目标与最终回答。"""

    goal: str
    answer: str


class CodingSession:
    """保留有界对话上下文，同时让每轮工具状态相互隔离。"""

    _MAX_STORED_MESSAGE_CHARS = 20_000
    _TRUNCATION_MARKER = "\n... [会话历史已截断]"

    def __init__(
        self,
        agent: CodingAgent,
        *,
        max_history_turns: int = 20,
        max_history_chars: int = 80_000,
    ) -> None:
        if not isinstance(max_history_turns, int) or isinstance(
            max_history_turns, bool
        ) or max_history_turns < 1:
            raise ValueError("max_history_turns 必须至少为 1")
        if not isinstance(max_history_chars, int) or isinstance(
            max_history_chars, bool
        ) or max_history_chars < 1_000:
            raise ValueError("max_history_chars 必须至少为 1000")
        self.agent = agent
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars
        self._turns: list[ConversationTurn] = []

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        """清除对话历史，但保留真实存在的待验证修改。"""

        self._turns.clear()
        self.agent.registry.state.start_turn()

    def mark_external_modifications(self, paths: tuple[str, ...]) -> None:
        """把聊天命令直接产生的文件变化纳入下一轮测试门禁。"""

        for path in paths:
            self.agent.registry.state.mark_file_modified(path)

    def run(
        self,
        goal: str,
        *,
        event_handler: AgentEventHandler | None = None,
        stream: bool = False,
    ) -> AgentResult:
        """在最近的有界问答历史之后执行新一轮任务。"""

        normalized_goal = goal.strip()
        if not normalized_goal:
            raise ValueError("任务目标不能为空")
        result = self.agent._run(
            normalized_goal,
            history=self._history_messages(),
            event_handler=event_handler,
            stream=stream,
        )
        self._turns.append(
            ConversationTurn(
                goal=self._truncate_for_history(normalized_goal),
                answer=self._truncate_for_history(result.answer),
            )
        )
        if len(self._turns) > self.max_history_turns:
            del self._turns[: -self.max_history_turns]
        return result

    def _history_messages(self) -> tuple[ChatMessage, ...]:
        selected_turns: list[ConversationTurn] = []
        used_chars = 0
        for turn in reversed(self._turns[-self.max_history_turns :]):
            turn_chars = len(turn.goal) + len(turn.answer)
            if used_chars + turn_chars > self.max_history_chars:
                break
            selected_turns.append(turn)
            used_chars += turn_chars

        messages: list[ChatMessage] = []
        for turn in reversed(selected_turns):
            messages.append(ChatMessage.user(turn.goal))
            messages.append(ChatMessage.assistant(turn.answer))
        return tuple(messages)

    def _truncate_for_history(self, text: str) -> str:
        per_message_limit = min(
            self._MAX_STORED_MESSAGE_CHARS,
            self.max_history_chars // 2,
        )
        if len(text) <= per_message_limit:
            return text
        retained_chars = per_message_limit - len(self._TRUNCATION_MARKER)
        return f"{text[:retained_chars]}{self._TRUNCATION_MARKER}"
