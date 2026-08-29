import json
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from safe_patch_agent.benchmark import (
    BenchmarkCase,
    BenchmarkError,
    BenchmarkRunner,
    ExpectedFile,
    build_parser,
    load_benchmark_cases,
)
from safe_patch_agent.llm_client import ChatCompletion
from safe_patch_agent.messages import ChatMessage, ToolCall


class ScriptedClient:
    def __init__(self, completions: Sequence[ChatCompletion]) -> None:
        self.completions = list(completions)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatCompletion:
        return self.completions.pop(0)


class BenchmarkTests(unittest.TestCase):
    def test_runner_scores_edit_and_keeps_original_fixture_unchanged(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fixture = Path(temporary_directory) / "fixture"
            fixture.mkdir()
            target = fixture / "demo.py"
            target.write_text("value = 1\n", encoding="utf-8")
            case = BenchmarkCase(
                id="edit_value",
                description="修改值并测试",
                fixture=fixture,
                goal="把值改成 2",
                expected_answer_contains=("完成",),
                required_tools=("read_file", "replace_text", "run_tests"),
                expected_files=(ExpectedFile("demo.py", "value = 2\n"),),
                require_tests_passed=True,
            )
            client = ScriptedClient(
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
                                        "old_text": "value = 1",
                                        "new_text": "value = 2",
                                    },
                                ),
                            ),
                        ),
                        "tool_calls",
                    ),
                    ChatCompletion(
                        ChatMessage.assistant(
                            None,
                            (ToolCall(id="tests", name="run_tests", arguments={}),),
                        ),
                        "tool_calls",
                    ),
                    ChatCompletion(ChatMessage.assistant("修改完成。"), "stop"),
                ]
            )

            with patch("safe_patch_agent.workspace.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "1 passed\n"
                report = BenchmarkRunner(
                    lambda _case: client,
                    model_name="fake-model",
                ).run((case,))

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")

        self.assertEqual(report.passed, 1)
        self.assertEqual(report.success_rate, 1.0)
        self.assertTrue(report.results[0].passed)
        self.assertEqual(
            report.results[0].observed_tools,
            ("read_file", "replace_text", "run_tests"),
        )
        self.assertEqual(
            report.results[0].successful_tools,
            ("read_file", "replace_text", "run_tests"),
        )
        self.assertTrue(report.results[0].state.last_test_passed)
        json.dumps(report.to_dict(), ensure_ascii=False)

    def test_failed_expectations_produce_explainable_checks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fixture = Path(temporary_directory) / "fixture"
            fixture.mkdir()
            case = BenchmarkCase(
                id="missing_evidence",
                description="缺少依据",
                fixture=fixture,
                goal="回答",
                expected_answer_contains=("目标文本",),
                required_tools=("read_file",),
            )
            client = ScriptedClient(
                [
                    ChatCompletion(
                        ChatMessage.assistant(
                            None,
                            (
                                ToolCall(
                                    id="bad-read",
                                    name="read_file",
                                    arguments={"path": "missing.txt"},
                                ),
                            ),
                        ),
                        "tool_calls",
                    ),
                    ChatCompletion(ChatMessage.assistant("目标文本"), "stop"),
                ]
            )
            report = BenchmarkRunner(lambda _case: client).run((case,))

        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertEqual(report.success_rate, 0.0)
        self.assertEqual(result.observed_tools, ("read_file",))
        self.assertEqual(result.successful_tools, ())
        failed_names = {check.name for check in result.checks if not check.passed}
        self.assertIn("调用工具：read_file", failed_names)

    def test_manifest_loads_relative_fixture_and_expected_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = root / "fixtures" / "demo"
            fixture.mkdir(parents=True)
            manifest = root / "cases.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "id": "demo",
                                "description": "演示案例",
                                "fixture": "fixtures/demo",
                                "goal": "完成演示",
                                "expected_files": {"result.txt": "完成\n"},
                                "require_tests_passed": True,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cases = load_benchmark_cases(manifest)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].fixture, fixture.resolve())
        self.assertEqual(cases[0].expected_files[0].path, "result.txt")
        self.assertTrue(cases[0].require_tests_run)
        self.assertTrue(cases[0].require_tests_passed)

    def test_manifest_rejects_fixture_path_traversal(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "cases.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "id": "escape",
                                "description": "越界",
                                "fixture": "../outside",
                                "goal": "查看",
                                "required_tools": ["read_file"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BenchmarkError, "相对路径"):
                load_benchmark_cases(manifest)

    def test_benchmark_help_warns_about_model_api_calls(self) -> None:
        help_text = build_parser().format_help()

        self.assertTrue(help_text.startswith("用法："))
        self.assertIn("模型 API 调用", help_text)
        self.assertIn("--manifest", help_text)


if __name__ == "__main__":
    unittest.main()
