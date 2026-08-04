import unittest

from safe_patch_agent.messages import ChatMessage, MessageFormatError, ToolCall


class MessageTests(unittest.TestCase):
    def test_tool_call_round_trip_uses_json_argument_string(self) -> None:
        call = ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})

        encoded = call.to_api_dict()
        decoded = ToolCall.from_api_dict(encoded)

        self.assertEqual(decoded, call)
        self.assertIsInstance(encoded["function"]["arguments"], str)

    def test_tool_message_contains_matching_call_id(self) -> None:
        call = ToolCall(id="call-1", name="list_files", arguments={})

        message = ChatMessage.tool(call, '{"ok": true}')

        self.assertEqual(
            message.to_api_dict(),
            {
                "role": "tool",
                "content": '{"ok": true}',
                "tool_call_id": "call-1",
            },
        )

    def test_invalid_tool_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(MessageFormatError, "无效的 JSON"):
            ToolCall.from_api_dict(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{"},
                }
            )

    def test_non_function_tool_call_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(MessageFormatError, "type 必须是 'function'"):
            ToolCall.from_api_dict(
                {
                    "id": "call-1",
                    "type": "custom",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            )

    def test_duplicate_tool_call_ids_are_rejected(self) -> None:
        first = ToolCall(id="duplicate", name="list_files", arguments={})
        second = ToolCall(id="duplicate", name="read_file", arguments={"path": "README.md"})

        with self.assertRaisesRegex(MessageFormatError, "必须唯一"):
            ChatMessage.assistant(None, (first, second))

    def test_missing_tool_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(MessageFormatError, "JSON 字符串"):
            ToolCall.from_api_dict(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file"},
                }
            )


if __name__ == "__main__":
    unittest.main()
