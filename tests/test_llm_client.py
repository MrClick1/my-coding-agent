import unittest
from collections.abc import Mapping
from typing import Any

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
