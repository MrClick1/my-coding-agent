"""Agent 运行时使用的最小工具定义与注册表。"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from safe_patch_agent.messages import ToolCall
from safe_patch_agent.state import AgentState

ToolHandler = Callable[..., Any]
PathNormalizer = Callable[[str], str]


class ToolRegistrationError(ValueError):
    """工具定义无法注册。"""


class ToolFileAccess(StrEnum):
    """工具对路径参数所指文件执行的访问类型。"""

    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"


@dataclass(frozen=True)
class ToolDefinition:
    """可调用函数，以及展示给模型的 JSON Schema。"""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    file_access: ToolFileAccess | None = None
    path_argument: str | None = None
    path_normalizer: PathNormalizer | None = None
    records_test_result: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ToolRegistrationError("工具名称不能为空")
        if not callable(self.handler):
            raise ToolRegistrationError(f"工具 {self.name!r} 的 handler 必须可调用")
        if self.file_access is not None and not isinstance(self.file_access, ToolFileAccess):
            try:
                object.__setattr__(self, "file_access", ToolFileAccess(self.file_access))
            except (TypeError, ValueError) as exc:
                raise ToolRegistrationError(
                    f"工具 {self.name!r} 的 file_access 无效"
                ) from exc
        if (self.file_access is None) != (self.path_argument is None):
            raise ToolRegistrationError(
                f"工具 {self.name!r} 必须同时配置 file_access 和 path_argument"
            )
        if self.path_argument is not None and (
            not isinstance(self.path_argument, str) or not self.path_argument
        ):
            raise ToolRegistrationError(f"工具 {self.name!r} 的 path_argument 不能为空")
        if self.path_normalizer is not None and self.file_access is None:
            raise ToolRegistrationError(
                f"工具 {self.name!r} 只有配置文件访问类型后才能使用 path_normalizer"
            )
        if self.path_normalizer is not None and not callable(self.path_normalizer):
            raise ToolRegistrationError(f"工具 {self.name!r} 的 path_normalizer 必须可调用")
        if not isinstance(self.records_test_result, bool):
            raise ToolRegistrationError(
                f"工具 {self.name!r} 的 records_test_result 必须是布尔值"
            )

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

    def __init__(self, state: AgentState | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.state = state or AgentState()

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
            accessed_path = self._prepare_file_access(definition, arguments)
            result = definition.handler(**arguments)
            if isinstance(result, Mapping):
                payload: Any = dict(result)
            else:
                payload = {"ok": True, "result": result}
            if payload.get("ok", True):
                self._record_file_access(definition, accessed_path, payload)
                self._record_test_result(definition, payload)
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

        return _json_result(payload)

    def _prepare_file_access(
        self,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> str | None:
        """在处理器运行前执行写入授权检查。"""

        if definition.file_access is None or definition.path_argument is None:
            return None
        path = arguments.get(definition.path_argument)
        if not isinstance(path, str):
            raise ValueError(f"工具参数 {definition.path_argument!r} 必须是字符串")
        if definition.path_normalizer is not None:
            path = definition.path_normalizer(path)
            if not isinstance(path, str) or not path:
                raise ValueError("工具的路径规范化器必须返回非空字符串")
        if definition.file_access in {ToolFileAccess.WRITE, ToolFileAccess.DELETE}:
            self.state.require_file_read(path)
        return path

    def _record_file_access(
        self,
        definition: ToolDefinition,
        requested_path: str | None,
        payload: Mapping[str, Any],
    ) -> None:
        """只在工具成功后记录实际访问的规范化路径。"""

        if definition.file_access is None or requested_path is None:
            return
        result_path = (
            requested_path
            if definition.file_access in {ToolFileAccess.WRITE, ToolFileAccess.DELETE}
            or definition.path_normalizer is not None
            else payload.get("path", requested_path)
        )
        if not isinstance(result_path, str):
            result_path = requested_path
        if definition.file_access is ToolFileAccess.READ:
            self.state.mark_file_read(result_path)
        elif definition.file_access in {
            ToolFileAccess.WRITE,
            ToolFileAccess.CREATE,
            ToolFileAccess.DELETE,
        }:
            self.state.mark_file_modified(result_path)

    def _record_test_result(
        self,
        definition: ToolDefinition,
        payload: Mapping[str, Any],
    ) -> None:
        """记录固定测试工具报告的通过或失败状态。"""

        if not definition.records_test_result:
            return
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise ValueError("测试工具必须返回布尔类型的 passed 字段")
        self.state.record_test_result(passed)


def _json_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
