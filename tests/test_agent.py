import json
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from safe_patch_agent.agent import AgentLoopLimitError, AgentToolLimitError, CodingAgent
from safe_patch_agent.llm_client import ChatCompletion
from safe_patch_agent.messages import ChatMessage, ToolCall
from safe_patch_agent.tooling import ToolRegistry
from safe_patch_agent.workspace import SafeWorkspace, build_agent_registry, build_read_only_registry


class ScriptedClient:
    def __init__(self, completions: Sequence[ChatCompletion]) -> None:
        self.completions = list(completions)
        self.requests: list[tuple[list[ChatMessage], list[Mapping[str, Any]]]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatCompletion:
        self.requests.append((list(messages), list(tools)))
        return self.completions.pop(0)


class CountingRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.execution_count = 0

    def schemas(self) -> list[dict[str, Any]]:
        return []

    def execute(self, call: ToolCall) -> str:
        self.execution_count += 1
        return '{"ok":true}'


class AgentTests(unittest.TestCase):
    def test_tool_result_is_sent_back_before_final_answer(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            registry = build_agent_registry(SafeWorkspace(root))
            call = ToolCall(id="call-1", name="list_files", arguments={"path": "."})
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (call,)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant("项目包含 README.md。"), "stop"),
                ]
            )

            result = CodingAgent(client, registry).run("项目中有什么？")

        self.assertEqual(result.answer, "项目包含 README.md。")
        self.assertEqual(result.model_rounds, 2)
        self.assertEqual(result.tool_calls, 1)
        second_request_messages = client.requests[1][0]
        tool_message = second_request_messages[-1]
        self.assertEqual(tool_message.role.value, "tool")
        self.assertEqual(tool_message.tool_call_id, "call-1")
        self.assertTrue(json.loads(tool_message.content)["ok"])
        self.assertEqual(
            {tool["function"]["name"] for tool in client.requests[0][1]},
            {"list_files", "read_file", "replace_text", "search_code"},
        )

    def test_loop_limit_stops_repeated_tool_calls(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = build_read_only_registry(SafeWorkspace(Path(temporary_directory)))
            first = ToolCall(id="call-1", name="list_files", arguments={})
            second = ToolCall(id="call-2", name="list_files", arguments={})
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (first,))),
                    ChatCompletion(ChatMessage.assistant(None, (second,))),
                ]
            )

            with self.assertRaisesRegex(AgentLoopLimitError, "2 轮模型调用上限"):
                CodingAgent(client, registry, max_rounds=2).run("一直查看")

    def test_tool_call_limit_applies_within_one_model_round(self) -> None:
        registry = CountingRegistry()
        calls = tuple(
            ToolCall(id=f"call-{index}", name="list_files", arguments={})
            for index in range(3)
        )
        client = ScriptedClient(
            [ChatCompletion(ChatMessage.assistant(None, calls), "tool_calls")]
        )

        with self.assertRaisesRegex(AgentToolLimitError, "2 次工具调用上限"):
            CodingAgent(client, registry, max_tool_calls=2).run("查看")
        self.assertEqual(registry.execution_count, 0)

    def test_truncated_model_output_is_not_accepted_as_final_answer(self) -> None:
        registry = CountingRegistry()
        client = ScriptedClient(
            [ChatCompletion(ChatMessage.assistant("partial answer"), "length")]
        )

        with self.assertRaisesRegex(RuntimeError, "被截断"):
            CodingAgent(client, registry).run("查看")

    def test_result_contains_read_state_and_new_run_resets_it(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            registry = build_agent_registry(SafeWorkspace(root))
            read_call = ToolCall(
                id="read",
                name="read_file",
                arguments={"path": "README.md"},
            )
            first_client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (read_call,)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant("已读取。"), "stop"),
                ]
            )

            first_result = CodingAgent(first_client, registry).run("读取 README")
            second_client = ScriptedClient(
                [ChatCompletion(ChatMessage.assistant("无需工具。"), "stop")]
            )
            second_result = CodingAgent(second_client, registry).run("直接回答")

        self.assertEqual(first_result.state.read_files, ("README.md",))
        self.assertEqual(second_result.state.read_files, ())

    def test_agent_can_replace_text_only_after_reading_target(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "demo.py"
            target.write_text("value = 1\n", encoding="utf-8")
            registry = build_agent_registry(SafeWorkspace(root))
            read_call = ToolCall(
                id="read",
                name="read_file",
                arguments={"path": "demo.py"},
            )
            replace_call = ToolCall(
                id="replace",
                name="replace_text",
                arguments={
                    "path": "demo.py",
                    "old_text": "value = 1",
                    "new_text": "value = 2",
                },
            )
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (read_call,)), "tool_calls"),
                    ChatCompletion(
                        ChatMessage.assistant(None, (replace_call,)),
                        "tool_calls",
                    ),
                    ChatCompletion(ChatMessage.assistant("修改完成。"), "stop"),
                ]
            )

            result = CodingAgent(client, registry).run("把 value 改为 2")
            updated_content = target.read_text(encoding="utf-8")

        self.assertEqual(updated_content, "value = 2\n")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.state.read_files, ("demo.py",))
        self.assertEqual(result.state.modified_files, ("demo.py",))


if __name__ == "__main__":
    unittest.main()
