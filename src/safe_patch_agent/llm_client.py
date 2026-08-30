"""基于标准库实现的轻量 OpenAI-compatible Chat Completions 客户端。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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


class SseTransport(Protocol):
    def post_sse(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]:
        """发送 JSON POST 请求，并逐个返回 SSE data JSON 对象。"""


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
    """零依赖的同步 JSON 与有界 SSE 传输层。"""

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

    def post_sse(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]:
        """逐行解析 Server-Sent Events，并限制累计响应大小。"""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=body, headers=dict(headers), method="POST")

        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                yield from self._iter_sse_events(response)
        except HTTPError as exc:
            if 300 <= exc.code < 400:
                raise LLMRequestError("模型接口不允许重定向") from exc
            error_body = exc.read(self._MAX_ERROR_BYTES).decode(
                "utf-8", errors="replace"
            )
            detail = f": {error_body}" if error_body else ""
            raise LLMRequestError(f"模型接口返回 HTTP {exc.code}{detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise LLMRequestError(f"无法连接模型接口：{exc}") from exc

    def _iter_sse_events(self, response: Any) -> Iterator[Mapping[str, Any]]:
        """从打开的响应中解析 UTF-8 SSE data 字段。"""

        total_bytes = 0
        data_lines: list[str] = []
        while True:
            remaining = self._MAX_RESPONSE_BYTES - total_bytes
            raw_line = response.readline(remaining + 1)
            if raw_line == b"":
                break
            total_bytes += len(raw_line)
            if total_bytes > self._MAX_RESPONSE_BYTES:
                raise LLMResponseError(
                    f"模型流式响应超过 {self._MAX_RESPONSE_BYTES} 字节上限"
                )
            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise LLMResponseError("模型接口返回了无效的 UTF-8 SSE") from exc

            if line == "":
                if data_lines:
                    event = self._decode_sse_data("\n".join(data_lines))
                    data_lines.clear()
                    if event is None:
                        return
                    yield event
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)

        if data_lines:
            event = self._decode_sse_data("\n".join(data_lines))
            if event is not None:
                yield event

    @staticmethod
    def _decode_sse_data(data: str) -> Mapping[str, Any] | None:
        if data == "[DONE]":
            return None
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("模型接口返回了无效的 SSE JSON") from exc
        if not isinstance(decoded, Mapping):
            raise LLMResponseError("模型接口返回的 SSE data JSON 不是对象")
        return decoded


@dataclass
class _StreamedToolCall:
    """按工具调用 index 累积 Chat Completions 的增量字段。"""

    call_id: str = ""
    call_type: str = ""
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)


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
        payload = self._build_payload(messages, tools)

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

    def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ChatCompletion:
        """请求 SSE 流式响应，并重组文本与函数工具调用增量。"""

        post_sse = getattr(self.transport, "post_sse", None)
        if not callable(post_sse):
            completion = self.complete(messages, tools)
            content = completion.message.content
            if content and on_text_delta is not None:
                on_text_delta(content)
            return completion

        payload = self._build_payload(messages, tools)
        payload["stream"] = True
        chunks = post_sse(
            url=self.config.chat_completions_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        return self._parse_stream(chunks, on_text_delta)

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_api_dict() for message in messages],
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _parse_stream(
        chunks: Iterator[Mapping[str, Any]],
        on_text_delta: Callable[[str], None] | None,
    ) -> ChatCompletion:
        content_parts: list[str] = []
        saw_content = False
        tool_calls: dict[int, _StreamedToolCall] = {}
        finish_reason: str | None = None

        for chunk in chunks:
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                raise LLMResponseError("模型流式响应中没有 choices 列表")
            if not choices:
                continue
            first_choice = choices[0]
            if not isinstance(first_choice, Mapping):
                raise LLMResponseError("模型流式响应中的 choice 不是对象")
            choice_index = first_choice.get("index", 0)
            if choice_index != 0:
                continue

            delta = first_choice.get("delta")
            if not isinstance(delta, Mapping):
                raise LLMResponseError("模型流式响应的 choice 中没有 delta 对象")
            role = delta.get("role")
            if role is not None and role != "assistant":
                raise LLMResponseError("模型流式响应消息的 role 必须是 'assistant'")

            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise LLMResponseError("模型流式响应目前只支持文本 content")
                saw_content = True
                content_parts.append(content)
                if content and on_text_delta is not None:
                    on_text_delta(content)

            raw_tool_calls = delta.get("tool_calls")
            if raw_tool_calls is not None:
                if not isinstance(raw_tool_calls, list):
                    raise LLMResponseError("模型流式响应的 tool_calls 必须是列表")
                for raw_call in raw_tool_calls:
                    OpenAICompatibleClient._accumulate_tool_call(
                        tool_calls,
                        raw_call,
                    )

            raw_finish_reason = first_choice.get("finish_reason")
            if raw_finish_reason is not None:
                finish_reason = (
                    raw_finish_reason
                    if isinstance(raw_finish_reason, str)
                    else str(raw_finish_reason)
                )

        if finish_reason is None:
            raise LLMResponseError("模型流式响应在结束前没有 finish_reason")

        ordered_indices = sorted(tool_calls)
        if ordered_indices != list(range(len(ordered_indices))):
            raise LLMResponseError("模型流式响应的工具调用 index 不连续")
        parsed_calls: list[ToolCall] = []
        try:
            for index in ordered_indices:
                call = tool_calls[index]
                parsed_calls.append(
                    ToolCall.from_api_dict(
                        {
                            "id": call.call_id,
                            "type": call.call_type,
                            "function": {
                                "name": "".join(call.name_parts),
                                "arguments": "".join(call.argument_parts),
                            },
                        }
                    )
                )
            message = ChatMessage.assistant(
                content="".join(content_parts) if saw_content else None,
                tool_calls=tuple(parsed_calls),
            )
        except MessageFormatError as exc:
            raise LLMResponseError(str(exc)) from exc
        return ChatCompletion(message=message, finish_reason=finish_reason)

    @staticmethod
    def _accumulate_tool_call(
        accumulated: dict[int, _StreamedToolCall],
        raw_call: Any,
    ) -> None:
        if not isinstance(raw_call, Mapping):
            raise LLMResponseError("模型流式响应的工具调用增量不是对象")
        index = raw_call.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise LLMResponseError("模型流式响应的工具调用缺少有效 index")
        call = accumulated.setdefault(index, _StreamedToolCall())

        call_id = raw_call.get("id")
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id:
                raise LLMResponseError("模型流式响应的工具调用 id 无效")
            if call.call_id and call.call_id != call_id:
                raise LLMResponseError("同一工具调用 index 返回了不同 id")
            call.call_id = call_id

        call_type = raw_call.get("type")
        if call_type is not None:
            if call_type != "function":
                raise LLMResponseError("模型流式响应只支持 function 工具调用")
            call.call_type = call_type

        function = raw_call.get("function")
        if function is None:
            return
        if not isinstance(function, Mapping):
            raise LLMResponseError("模型流式响应的工具调用 function 不是对象")
        name = function.get("name")
        if name is not None:
            if not isinstance(name, str):
                raise LLMResponseError("模型流式响应的工具函数名增量不是字符串")
            call.name_parts.append(name)
        arguments = function.get("arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise LLMResponseError("模型流式响应的工具参数增量不是字符串")
            call.argument_parts.append(arguments)

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
