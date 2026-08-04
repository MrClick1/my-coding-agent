"""基于标准库实现的轻量 OpenAI-compatible Chat Completions 客户端。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from safe_patch_agent.config import LLMConfig
from safe_patch_agent.messages import ChatMessage, MessageFormatError, ToolCall


class LLMError(RuntimeError):
    """模型请求与响应异常的基类。"""


class LLMRequestError(LLMError):
    """远程模型接口无法完成请求。"""


class LLMResponseError(LLMError):
    """模型接口返回了不受支持的响应结构。"""


@dataclass(frozen=True)
class ChatCompletion:
    message: ChatMessage
    finish_reason: str | None = None


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatCompletion:
        """请求模型生成下一轮 assistant 消息。"""


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """发送 JSON POST 请求，并返回解析后的响应对象。"""


class _NoRedirectHandler(HTTPRedirectHandler):
    """禁止通过 HTTP 重定向转发 Bearer 凭据。"""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class UrllibJsonTransport:
    """零依赖的同步 JSON 传输层。"""

    _MAX_RESPONSE_BYTES = 5_000_000
    _MAX_ERROR_BYTES = 500

    def __init__(self) -> None:
        self._opener = build_opener(_NoRedirectHandler())

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=body, headers=dict(headers), method="POST")

        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                raw_body = response.read(self._MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            if 300 <= exc.code < 400:
                raise LLMRequestError("模型接口不允许重定向") from exc
            error_body = exc.read(self._MAX_ERROR_BYTES).decode("utf-8", errors="replace")
            detail = f": {error_body}" if error_body else ""
            raise LLMRequestError(f"模型接口返回 HTTP {exc.code}{detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise LLMRequestError(f"无法连接模型接口：{exc}") from exc

        if len(raw_body) > self._MAX_RESPONSE_BYTES:
            raise LLMResponseError(
                f"模型响应超过 {self._MAX_RESPONSE_BYTES} 字节上限"
            )

        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError("模型接口没有返回有效的 UTF-8 JSON") from exc
        if not isinstance(decoded, Mapping):
            raise LLMResponseError("模型接口返回的 JSON 不是对象")
        return decoded


class OpenAICompatibleClient:
    """向 Chat Completions 接口发送消息和函数工具 Schema。"""

    def __init__(
        self,
        config: LLMConfig,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_api_dict() for message in messages],
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        raw = self.transport.post_json(
            url=self.config.chat_completions_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        return self._parse_completion(raw)

    @staticmethod
    def _parse_completion(raw: Mapping[str, Any]) -> ChatCompletion:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("模型响应中没有 choices")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMResponseError("模型响应中的 choice 不是对象")

        message_data = first_choice.get("message")
        if not isinstance(message_data, Mapping):
            raise LLMResponseError("模型响应的 choice 中没有 message 对象")
        if message_data.get("role") != "assistant":
            raise LLMResponseError("模型响应消息的 role 必须是 'assistant'")
        content = message_data.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMResponseError("目前只支持文本形式的 assistant content")

        raw_tool_calls = message_data.get("tool_calls", [])
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            raise LLMResponseError("assistant 的 tool_calls 必须是列表")
        try:
            tool_calls = tuple(
                ToolCall.from_api_dict(item)
                for item in raw_tool_calls
                if isinstance(item, Mapping)
            )
        except MessageFormatError as exc:
            raise LLMResponseError(str(exc)) from exc
        if len(tool_calls) != len(raw_tool_calls):
            raise LLMResponseError("assistant 的 tool_calls 中包含非对象元素")

        try:
            message = ChatMessage.assistant(content=content, tool_calls=tool_calls)
        except MessageFormatError as exc:
            raise LLMResponseError(str(exc)) from exc

        finish_reason = first_choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)
        return ChatCompletion(message=message, finish_reason=finish_reason)
