"""模型客户端与 Agent 运行时共享的数据结构。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MessageFormatError(ValueError):
    """模型响应不符合预期的消息格式。"""


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """模型请求的一次函数工具调用。"""

    id: str
    name: str
    arguments: Mapping[str, Any]

    @classmethod
    def from_api_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        call_id = data.get("id")
        call_type = data.get("type")
        function = data.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise MessageFormatError("工具调用缺少非空 id")
        if call_type != "function":
            raise MessageFormatError("工具调用的 type 必须是 'function'")
        if not isinstance(function, Mapping):
            raise MessageFormatError("工具调用缺少 function 对象")

        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise MessageFormatError("工具调用缺少函数名称")
        if not isinstance(raw_arguments, str):
            raise MessageFormatError("工具调用参数必须是 JSON 字符串")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise MessageFormatError(f"工具调用 {name!r} 返回了无效的 JSON 参数") from exc
        if not isinstance(arguments, dict):
            raise MessageFormatError("工具调用参数必须能解析为 JSON 对象")
        return cls(id=call_id, name=name, arguments=arguments)

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ChatMessage:
    """一条聊天消息，可以包含助手的工具调用或工具执行结果。"""

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.role, str) and not isinstance(self.role, MessageRole):
            try:
                object.__setattr__(self, "role", MessageRole(self.role))
            except ValueError as exc:
                raise MessageFormatError(f"不支持的消息角色：{self.role!r}") from exc

        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise MessageFormatError("只有 assistant 消息可以包含工具调用")
        call_ids = [call.id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise MessageFormatError("同一轮 assistant 消息中的工具调用 id 必须唯一")
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise MessageFormatError("tool 消息必须包含 tool_call_id")
        if self.role is not MessageRole.TOOL and self.tool_call_id is not None:
            raise MessageFormatError("tool_call_id 只能用于 tool 消息")
        if self.content is None and not self.tool_calls:
            raise MessageFormatError("消息必须包含 content 或至少一个工具调用")

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> ChatMessage:
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, call: ToolCall, content: str) -> ChatMessage:
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=call.id,
        )

    def to_api_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_api_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data
