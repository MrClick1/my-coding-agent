import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from safe_patch_agent.cli import (
    build_parser,
    main,
    request_replacement_approval,
    run_chat,
)
from safe_patch_agent.workspace import ReplacementPreview


class InteractiveStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class FakeSession:
    def __init__(self) -> None:
        self.goals: list[str] = []
        self.clear_count = 0

    def run(self, goal: str) -> SimpleNamespace:
        self.goals.append(goal)
        return SimpleNamespace(answer=f"已处理：{goal}")

    def clear(self) -> None:
        self.clear_count += 1


class CLITests(unittest.TestCase):
    def test_help_text_uses_chinese_headings(self) -> None:
        help_text = build_parser().format_help()

        self.assertTrue(help_text.startswith("用法："))
        self.assertIn("位置参数:", help_text)
        self.assertIn("选项:", help_text)

    def test_goal_is_optional_and_chat_flag_is_available(self) -> None:
        parser = build_parser()

        chat_args = parser.parse_args([])
        initial_chat_args = parser.parse_args(["初始任务", "--chat"])

        self.assertIsNone(chat_args.goal)
        self.assertFalse(chat_args.chat)
        self.assertEqual(initial_chat_args.goal, "初始任务")
        self.assertTrue(initial_chat_args.chat)

    def test_missing_model_configuration_returns_clear_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_env_file = root / "missing.env"
            stderr = StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "inspect this project",
                        "--workspace",
                        str(root),
                        "--env-file",
                        str(missing_env_file),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("缺少必需配置", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_replacement_approval_shows_diff_and_accepts_explicit_yes(self) -> None:
        preview = ReplacementPreview(
            path="src/app.py",
            diff="--- a/src/app.py\n+++ b/src/app.py\n-old\n+new\n",
            replacements=1,
            original_bytes=4,
            updated_bytes=4,
        )
        output = StringIO()

        approved = request_replacement_approval(
            preview,
            input_stream=InteractiveStringIO("是\n"),
            output_stream=output,
        )

        self.assertTrue(approved)
        self.assertIn("SafePatch 修改预览", output.getvalue())
        self.assertIn("src/app.py", output.getvalue())
        self.assertIn("-old\n+new", output.getvalue())

    def test_replacement_approval_defaults_to_rejection(self) -> None:
        preview = ReplacementPreview("demo.py", "diff\n", 1, 3, 3)
        output = StringIO()

        approved = request_replacement_approval(
            preview,
            input_stream=InteractiveStringIO("\n"),
            output_stream=output,
        )

        self.assertFalse(approved)
        self.assertIn("已取消", output.getvalue())

    def test_non_interactive_input_cannot_approve_replacement(self) -> None:
        preview = ReplacementPreview("demo.py", "diff\n", 1, 3, 3)
        output = StringIO()

        approved = request_replacement_approval(
            preview,
            input_stream=StringIO("yes\n"),
            output_stream=output,
        )

        self.assertFalse(approved)
        self.assertIn("不是交互式终端", output.getvalue())

    def test_chat_runs_multiple_inputs_and_session_commands(self) -> None:
        session = FakeSession()
        output = StringIO()
        input_stream = InteractiveStringIO(
            "第一个任务\n/help\n/clear\n/unknown\n第二个任务\n/exit\n"
        )

        exit_code = run_chat(  # type: ignore[arg-type]
            session,
            input_stream=input_stream,
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(session.goals, ["第一个任务", "第二个任务"])
        self.assertEqual(session.clear_count, 1)
        self.assertIn("持续会话已启动", output.getvalue())
        self.assertIn("已处理：第一个任务", output.getvalue())
        self.assertIn("可用会话命令", output.getvalue())
        self.assertIn("未知会话命令", output.getvalue())
        self.assertIn("会话已退出", output.getvalue())

    def test_chat_can_run_initial_goal_before_prompting(self) -> None:
        session = FakeSession()
        output = StringIO()

        exit_code = run_chat(  # type: ignore[arg-type]
            session,
            initial_goal="先介绍项目",
            input_stream=InteractiveStringIO("/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(session.goals, ["先介绍项目"])
        self.assertIn("已处理：先介绍项目", output.getvalue())

    def test_chat_rejects_non_interactive_input(self) -> None:
        session = FakeSession()
        output = StringIO()

        exit_code = run_chat(  # type: ignore[arg-type]
            session,
            input_stream=StringIO("任务\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(session.goals, [])
        self.assertIn("需要交互式终端", output.getvalue())


if __name__ == "__main__":
    unittest.main()
