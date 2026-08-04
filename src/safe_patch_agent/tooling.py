"""Agent 运行时使用的最小工具定义与注册表。"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from safe_patch_agent.messages import ToolCall

ToolHandler = Callable[..., Any]


class ToolRegistrationError(ValueError):
    """工具定义无法注册。"""


@dataclass(frozen=True)
class ToolDefinition:
    """可调用函数，以及展示给模型的 JSON Schema。"""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name:
            raise ToolRegistrationError("工具名称不能为空")
        if not callable(self.handler):
            raise ToolRegistrationError(f"工具 {self.name!r} 的 handler 必须可调用")

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


class ToolRegistry:
    """注册工具、公开工具 Schema，并安全执行模型请求。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ToolRegistrationError(f"工具 {definition.name!r} 已经注册")
        self._tools[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.to_api_dict() for definition in self._tools.values()]

    def execute(self, call: ToolCall) -> str:
        """执行模型发起的工具调用，并始终返回 JSON 工具消息。

        可预期的用户或工具错误会返回给模型，而不会让运行时崩溃。未预期的
        异常也会被隔离，避免向模型泄露调用栈或本地敏感信息。
        """

        definition = self._tools.get(call.name)
        if definition is None:
            return _json_result(
                {
                    "ok": False,
                    "error": {
                        "type": "unknown_tool",
                        "message": f"未知工具：{call.name}",
                    },
                }
            )

        arguments = dict(call.arguments)
        try:
            handler_signature = inspect.signature(definition.handler)
        except (TypeError, ValueError):
            handler_signature = None
        if handler_signature is not None:
            try:
                handler_signature.bind(**arguments)
            except TypeError:
                return _json_result(
                    {
                        "ok": False,
                        "error": {
                            "type": "invalid_arguments",
                            "message": "工具调用参数与函数签名不匹配",
                        },
                    }
                )

        try:
            result = definition.handler(**arguments)
        except ValueError as exc:
            return _json_result(
                {
                    "ok": False,
                    "error": {"type": "tool_error", "message": str(exc)},
                }
            )
        except Exception:
            return _json_result(
                {
                    "ok": False,
                    "error": {
                        "type": "internal_tool_error",
                        "message": "工具执行时发生了未预期的内部错误。",
                    },
                }
            )

        if isinstance(result, Mapping):
            payload: Any = dict(result)
        else:
            payload = {"ok": True, "result": result}
        return _json_result(payload)


def _json_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
