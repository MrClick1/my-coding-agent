import json
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from safe_patch_agent.agent import (
    AgentLoopLimitError,
    AgentToolLimitError,
    AgentVerificationError,
    CodingAgent,
    CodingSession,
)
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


def build_approved_registry(root: Path) -> ToolRegistry:
    """为离线 Agent 测试显式批准临时工作区中的替换。"""

    workspace = SafeWorkspace(root, replacement_approval=lambda _preview: True)
    return build_agent_registry(workspace)


class AgentTests(unittest.TestCase):
    def test_tool_result_is_sent_back_before_final_answer(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            registry = build_approved_registry(root)
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
            {"list_files", "read_file", "replace_text", "run_tests", "search_code"},
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
            registry = build_approved_registry(root)
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
            registry = build_approved_registry(root)
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
            test_call = ToolCall(id="tests", name="run_tests", arguments={})
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (read_call,)), "tool_calls"),
                    ChatCompletion(
                        ChatMessage.assistant(None, (replace_call,)),
                        "tool_calls",
                    ),
                    ChatCompletion(
                        ChatMessage.assistant(None, (test_call,)),
                        "tool_calls",
                    ),
                    ChatCompletion(ChatMessage.assistant("修改完成。"), "stop"),
                ]
            )

            with patch("safe_patch_agent.workspace.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "1 passed\n"
                result = CodingAgent(client, registry).run("把 value 改为 2")
            updated_content = target.read_text(encoding="utf-8")

        self.assertEqual(updated_content, "value = 2\n")
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(result.state.read_files, ("demo.py",))
        self.assertEqual(result.state.modified_files, ("demo.py",))

    def test_agent_can_report_a_user_rejected_modification(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "demo.py"
            target.write_text("value = 1\n", encoding="utf-8")
            workspace = SafeWorkspace(
                root,
                replacement_approval=lambda _preview: False,
            )
            registry = build_agent_registry(workspace)
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
                    ChatCompletion(ChatMessage.assistant("用户拒绝了修改。"), "stop"),
                ]
            )

            result = CodingAgent(client, registry).run("把 value 改为 2")
            updated_content = target.read_text(encoding="utf-8")

        self.assertEqual(updated_content, "value = 1\n")
        self.assertEqual(result.answer, "用户拒绝了修改。")
        self.assertEqual(result.state.modified_files, ())
        self.assertEqual(result.state.test_runs, 0)
        tool_result = json.loads(client.requests[2][0][-1].content)
        self.assertFalse(tool_result["ok"])
        self.assertIn("用户拒绝", tool_result["error"]["message"])

    def test_agent_requires_test_run_after_latest_modification(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "demo.py"
            target.write_text("value = 1\n", encoding="utf-8")
            registry = build_approved_registry(root)
            calls = [
                ToolCall(id="read", name="read_file", arguments={"path": "demo.py"}),
                ToolCall(
                    id="replace",
                    name="replace_text",
                    arguments={
                        "path": "demo.py",
                        "old_text": "value = 1",
                        "new_text": "value = 2",
                    },
                ),
                ToolCall(id="tests", name="run_tests", arguments={}),
            ]
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (calls[0],)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant(None, (calls[1],)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant("修改完成。"), "stop"),
                    ChatCompletion(ChatMessage.assistant(None, (calls[2],)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant("修改并验证完成。"), "stop"),
                ]
            )

            with patch("safe_patch_agent.workspace.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "1 passed\n"
                result = CodingAgent(client, registry).run("修改并测试")

        self.assertEqual(result.answer, "修改并验证完成。")
        self.assertEqual(result.model_rounds, 5)
        self.assertEqual(result.state.test_runs, 1)
        self.assertTrue(result.state.last_test_passed)
        self.assertFalse(result.state.has_unverified_changes)
        reminder = client.requests[3][0][-1]
        self.assertEqual(reminder.role.value, "system")
        self.assertIn("必须调用 run_tests", reminder.content)

    def test_agent_errors_when_round_limit_prevents_required_test(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "demo.py").write_text("old\n", encoding="utf-8")
            registry = build_approved_registry(root)
            read_call = ToolCall(
                id="read",
                name="read_file",
                arguments={"path": "demo.py"},
            )
            replace_call = ToolCall(
                id="replace",
                name="replace_text",
                arguments={"path": "demo.py", "old_text": "old", "new_text": "new"},
            )
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (read_call,)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant(None, (replace_call,)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant("修改完成。"), "stop"),
                ]
            )

            with self.assertRaisesRegex(AgentVerificationError, "没有运行测试"):
                CodingAgent(client, registry, max_rounds=3).run("修改")

    def test_unverified_change_survives_into_next_agent_turn(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "demo.py").write_text("old\n", encoding="utf-8")
            registry = build_approved_registry(root)
            first_client = ScriptedClient(
                [
                    ChatCompletion(
                        ChatMessage.assistant(
                            None,
                            (
                                ToolCall(
                                    id="read",
                                    name="read_file",
                                    arguments={"path": "demo.py"},
                                ),
                            ),
                        ),
                        "tool_calls",
                    ),
                    ChatCompletion(
                        ChatMessage.assistant(
                            None,
                            (
                                ToolCall(
                                    id="replace",
                                    name="replace_text",
                                    arguments={
                                        "path": "demo.py",
                                        "old_text": "old",
                                        "new_text": "new",
                                    },
                                ),
                            ),
                        ),
                        "tool_calls",
                    ),
                    ChatCompletion(ChatMessage.assistant("已修改。"), "stop"),
                ]
            )
            with self.assertRaises(AgentVerificationError):
                CodingAgent(first_client, registry, max_rounds=3).run("修改")

            test_call = ToolCall(id="tests", name="run_tests", arguments={})
            second_client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant("无需继续。"), "stop"),
                    ChatCompletion(
                        ChatMessage.assistant(None, (test_call,)),
                        "tool_calls",
                    ),
                    ChatCompletion(ChatMessage.assistant("测试完成。"), "stop"),
                ]
            )
            with patch("safe_patch_agent.workspace.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "1 passed\n"
                result = CodingAgent(second_client, registry).run("继续")

        self.assertEqual(result.answer, "测试完成。")
        self.assertEqual(result.state.modified_files, ("demo.py",))
        self.assertEqual(result.state.test_runs, 1)
        self.assertTrue(result.state.last_test_passed)
        reminder = second_client.requests[1][0][-1]
        self.assertEqual(reminder.role.value, "system")
        self.assertIn("必须调用 run_tests", reminder.content)

    def test_session_keeps_compact_answers_but_not_old_tool_messages(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            registry = build_approved_registry(root)
            read_call = ToolCall(
                id="read",
                name="read_file",
                arguments={"path": "README.md"},
            )
            client = ScriptedClient(
                [
                    ChatCompletion(ChatMessage.assistant(None, (read_call,)), "tool_calls"),
                    ChatCompletion(ChatMessage.assistant("标题是 Demo。"), "stop"),
                    ChatCompletion(ChatMessage.assistant("它来自上一轮回答。"), "stop"),
                ]
            )
            session = CodingSession(CodingAgent(client, registry))

            first_result = session.run("README 的标题是什么？")
            second_result = session.run("你上一轮发现了什么？")

        self.assertEqual(first_result.state.read_files, ("README.md",))
        self.assertEqual(second_result.state.read_files, ())
        self.assertEqual(session.turn_count, 2)
        second_turn_messages = client.requests[2][0]
        self.assertEqual(
            [message.role.value for message in second_turn_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(second_turn_messages[1].content, "README 的标题是什么？")
        self.assertEqual(second_turn_messages[2].content, "标题是 Demo。")
        self.assertEqual(second_turn_messages[3].content, "你上一轮发现了什么？")

    def test_session_clear_removes_history_and_tool_state(self) -> None:
        registry = CountingRegistry()
        client = ScriptedClient(
            [
                ChatCompletion(ChatMessage.assistant("第一次回答。"), "stop"),
                ChatCompletion(ChatMessage.assistant("第二次回答。"), "stop"),
            ]
        )
        session = CodingSession(CodingAgent(client, registry))

        session.run("第一次问题")
        registry.state.mark_file_read("demo.py")
        session.clear()
        session.run("第二次问题")

        self.assertEqual(session.turn_count, 1)
        self.assertEqual(registry.state.snapshot().read_files, ())
        self.assertEqual(
            [message.content for message in client.requests[1][0]],
            [CodingAgent(client, registry).system_prompt, "第二次问题"],
        )

    def test_session_history_keeps_only_configured_recent_turns(self) -> None:
        registry = CountingRegistry()
        client = ScriptedClient(
            [
                ChatCompletion(ChatMessage.assistant("回答一"), "stop"),
                ChatCompletion(ChatMessage.assistant("回答二"), "stop"),
                ChatCompletion(ChatMessage.assistant("回答三"), "stop"),
            ]
        )
        session = CodingSession(
            CodingAgent(client, registry),
            max_history_turns=1,
        )

        session.run("问题一")
        session.run("问题二")
        session.run("问题三")

        self.assertEqual(session.turn_count, 1)
        third_request_contents = [message.content for message in client.requests[2][0]]
        self.assertNotIn("问题一", third_request_contents)
        self.assertIn("问题二", third_request_contents)
        self.assertIn("回答二", third_request_contents)
        self.assertIn("问题三", third_request_contents)

    def test_session_truncates_large_messages_before_reusing_them(self) -> None:
        registry = CountingRegistry()
        client = ScriptedClient(
            [
                ChatCompletion(ChatMessage.assistant("a" * 700), "stop"),
                ChatCompletion(ChatMessage.assistant("完成"), "stop"),
            ]
        )
        session = CodingSession(
            CodingAgent(client, registry),
            max_history_chars=1_000,
        )

        session.run("q" * 700)
        session.run("继续")

        reused_history = client.requests[1][0][1:3]
        self.assertTrue(all(len(message.content or "") <= 500 for message in reused_history))
        self.assertTrue(
            all("会话历史已截断" in (message.content or "") for message in reused_history)
        )

    def test_session_rejects_invalid_history_limits(self) -> None:
        agent = CodingAgent(ScriptedClient([]), CountingRegistry())

        with self.assertRaisesRegex(ValueError, "max_history_turns"):
            CodingSession(agent, max_history_turns=0)
        with self.assertRaisesRegex(ValueError, "max_history_turns"):
            CodingSession(agent, max_history_turns=True)
        with self.assertRaisesRegex(ValueError, "max_history_chars"):
            CodingSession(agent, max_history_chars=999)
        with self.assertRaisesRegex(ValueError, "max_history_chars"):
            CodingSession(agent, max_history_chars=1_000.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
