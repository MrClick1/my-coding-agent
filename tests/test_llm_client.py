import unittest
from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import patch

from safe_patch_agent.config import LLMConfig
from safe_patch_agent.llm_client import (
    LLMResponseError,
    OpenAICompatibleClient,
    UrllibJsonTransport,
)
from safe_patch_agent.messages import ChatMessage, ToolCall


class FakeTransport:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.request = {
            "url": url,
            "headers": dict(headers),
            "payload": dict(payload),
            "timeout_seconds": timeout_seconds,
        }
        return self.response


class FakeStreamingTransport(FakeTransport):
    def __init__(self, events: list[Mapping[str, Any]]) -> None:
        super().__init__({})
        self.events = events

    def post_sse(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]:
        self.request = {
            "url": url,
            "headers": dict(headers),
            "payload": dict(payload),
            "timeout_seconds": timeout_seconds,
        }
        yield from self.events


class OversizedResponse:
    def __enter__(self) -> "OversizedResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return b"x" * amount


class OversizedResponseOpener:
    def open(self, request: Any, timeout: float) -> OversizedResponse:
        return OversizedResponse()


class SseResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = iter(lines)

    def __enter__(self) -> "SseResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def readline(self, amount: int) -> bytes:
        line = next(self.lines, b"")
        return line[:amount]


class SseResponseOpener:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def open(self, request: Any, timeout: float) -> SseResponse:
        return SseResponse(self.lines)


class LLMClientTests(unittest.TestCase):
    def make_config(self) -> LLMConfig:
        return LLMConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            timeout_seconds=15,
        )

    def test_sends_tool_schemas_and_parses_tool_call(self) -> None:
        transport = FakeTransport(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": '{"path":"."}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.make_config(), transport)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        completion = client.complete([ChatMessage.user("inspect")], tools)

        self.assertEqual(completion.message.tool_calls[0].name, "list_files")
        self.assertEqual(completion.message.tool_calls[0].arguments, {"path": "."})
        self.assertEqual(transport.request["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(transport.request["payload"]["tool_choice"], "auto")
        self.assertEqual(transport.request["payload"]["tools"], tools)
        self.assertEqual(transport.request["headers"]["Authorization"], "Bearer test-key")

    def test_rejects_response_without_choices(self) -> None:
        client = OpenAICompatibleClient(self.make_config(), FakeTransport({}))

        with self.assertRaisesRegex(LLMResponseError, "没有 choices"):
            client.complete([ChatMessage.user("inspect")], [])

    def test_rejects_non_assistant_response_message(self) -> None:
        transport = FakeTransport(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "user", "content": "unexpected"},
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.make_config(), transport)

        with self.assertRaisesRegex(LLMResponseError, "role 必须是 'assistant'"):
            client.complete([ChatMessage.user("inspect")], [])

    def test_follow_up_wire_payload_links_tool_result_without_name_field(self) -> None:
        transport = FakeTransport(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.make_config(), transport)
        call = ToolCall(id="call-1", name="list_files", arguments={"path": "."})
        messages = [
            ChatMessage.user("inspect"),
            ChatMessage.assistant(None, (call,)),
            ChatMessage.tool(call, '{"ok":true}'),
        ]

        client.complete(messages, [])

        wire_messages = transport.request["payload"]["messages"]
        self.assertEqual(
            wire_messages[-1],
            {
                "role": "tool",
                "content": '{"ok":true}',
                "tool_call_id": "call-1",
            },
        )

    def test_streaming_completion_emits_text_deltas_and_final_message(self) -> None:
        transport = FakeStreamingTransport(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "你"},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "好"},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ]
                },
            ]
        )
        client = OpenAICompatibleClient(self.make_config(), transport)
        deltas: list[str] = []

        completion = client.stream_complete(
            [ChatMessage.user("问候")],
            [],
            deltas.append,
        )

        self.assertEqual(deltas, ["你", "好"])
        self.assertEqual(completion.message.content, "你好")
        self.assertEqual(completion.finish_reason, "stop")
        self.assertTrue(transport.request["payload"]["stream"])
        self.assertEqual(
            transport.request["headers"]["Accept"],
            "text/event-stream",
        )

    def test_streaming_completion_reassembles_multiple_tool_calls(self) -> None:
        transport = FakeStreamingTransport(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_",
                                            "arguments": '{"query":',
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "id": "call-2",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"',
                                        },
                                    },
                                ],
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "code",
                                            "arguments": '"Agent"}',
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "function": {
                                            "arguments": 'README.md"}',
                                        },
                                    },
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        )
        client = OpenAICompatibleClient(self.make_config(), transport)

        completion = client.stream_complete([ChatMessage.user("检查")], [])

        self.assertEqual(
            [call.name for call in completion.message.tool_calls],
            ["search_code", "read_file"],
        )
        self.assertEqual(
            completion.message.tool_calls[0].arguments,
            {"query": "Agent"},
        )
        self.assertEqual(
            completion.message.tool_calls[1].arguments,
            {"path": "README.md"},
        )
        self.assertEqual(completion.finish_reason, "tool_calls")

    def test_streaming_completion_requires_finish_reason(self) -> None:
        transport = FakeStreamingTransport(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "partial"},
                            "finish_reason": None,
                        }
                    ]
                }
            ]
        )
        client = OpenAICompatibleClient(self.make_config(), transport)

        with self.assertRaisesRegex(LLMResponseError, "finish_reason"):
            client.stream_complete([ChatMessage.user("inspect")], [])

    def test_streaming_completion_rejects_non_contiguous_tool_indices(self) -> None:
        transport = FakeStreamingTransport(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "id": "call-2",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"README.md"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ]
        )
        client = OpenAICompatibleClient(self.make_config(), transport)

        with self.assertRaisesRegex(LLMResponseError, "index 不连续"):
            client.stream_complete([ChatMessage.user("inspect")], [])

    def test_streaming_completion_rejects_incomplete_tool_json(self) -> None:
        transport = FakeStreamingTransport(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ]
        )
        client = OpenAICompatibleClient(self.make_config(), transport)

        with self.assertRaisesRegex(LLMResponseError, "无效的 JSON 参数"):
            client.stream_complete([ChatMessage.user("inspect")], [])

    def test_streaming_falls_back_for_json_only_custom_transport(self) -> None:
        transport = FakeTransport(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "fallback"},
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.make_config(), transport)
        deltas: list[str] = []

        completion = client.stream_complete(
            [ChatMessage.user("inspect")],
            [],
            deltas.append,
        )

        self.assertEqual(completion.message.content, "fallback")
        self.assertEqual(deltas, ["fallback"])

    def test_standard_transport_parses_sse_data_and_done_marker(self) -> None:
        transport = UrllibJsonTransport()
        transport._opener = SseResponseOpener(
            [
                b": keepalive\n",
                b"data: {\"choices\": []}\n",
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ]
        )

        events = list(
            transport.post_sse(
                "https://example.test/v1/chat/completions",
                {"Authorization": "Bearer test"},
                {"model": "test", "messages": [], "stream": True},
                10,
            )
        )

        self.assertEqual(events, [{"choices": []}])

    def test_standard_transport_rejects_oversized_sse_response(self) -> None:
        transport = UrllibJsonTransport()
        transport._opener = SseResponseOpener([b"data: " + b"x" * 20])

        with (
            patch.object(UrllibJsonTransport, "_MAX_RESPONSE_BYTES", 10),
            self.assertRaisesRegex(LLMResponseError, "流式响应超过"),
        ):
            list(
                transport.post_sse(
                    "https://example.test/v1/chat/completions",
                    {"Authorization": "Bearer test"},
                    {"model": "test", "messages": [], "stream": True},
                    10,
                )
            )

    def test_standard_transport_rejects_oversized_response(self) -> None:
        transport = UrllibJsonTransport()
        transport._opener = OversizedResponseOpener()

        with self.assertRaisesRegex(LLMResponseError, "超过"):
            transport.post_json(
                "https://example.test/v1/chat/completions",
                {"Authorization": "Bearer test"},
                {"model": "test", "messages": []},
                10,
            )


if __name__ == "__main__":
    unittest.main()
