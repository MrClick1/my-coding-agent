import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from safe_patch_agent.agent import AgentEvent, AgentEventKind
from safe_patch_agent.changes import ChangeJournal
from safe_patch_agent.cli import (
    CLIProgressRenderer,
    build_parser,
    format_change_log,
    main,
    parse_rollback_target,
    request_batch_change_approval,
    request_creation_approval,
    request_deletion_approval,
    request_replacement_approval,
    request_rollback_approval,
    run_chat,
)
from safe_patch_agent.workspace import (
    BatchChangePreview,
    CreationPreview,
    DeletionPreview,
    ReplacementPreview,
    RollbackPreview,
    SafeWorkspace,
)


class InteractiveStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class FakeSession:
    def __init__(self) -> None:
        self.goals: list[str] = []
        self.clear_count = 0
        self.external_modifications: list[tuple[str, ...]] = []

    def run(self, goal: str, **_kwargs: Any) -> SimpleNamespace:
        self.goals.append(goal)
        return SimpleNamespace(answer=f"已处理：{goal}")

    def clear(self) -> None:
        self.clear_count += 1

    def mark_external_modifications(self, paths: tuple[str, ...]) -> None:
        self.external_modifications.append(paths)


class StreamingFakeSession(FakeSession):
    def run(self, goal: str, **kwargs: Any) -> SimpleNamespace:
        self.goals.append(goal)
        event_handler = kwargs["event_handler"]
        self.stream_enabled = kwargs["stream"]
        event_handler(AgentEvent(AgentEventKind.MODEL_START, 1))
        event_handler(AgentEvent(AgentEventKind.TEXT_DELTA, 1, text="已处理："))
        event_handler(AgentEvent(AgentEventKind.TEXT_DELTA, 1, text=goal))
        event_handler(
            AgentEvent(
                AgentEventKind.MODEL_COMPLETE,
                1,
                has_tool_calls=False,
            )
        )
        return SimpleNamespace(answer=f"已处理：{goal}")


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
        no_stream_args = parser.parse_args(["任务", "--no-stream"])
        validation_args = parser.parse_args(
            ["任务", "--validation-config", "checks.toml"]
        )

        self.assertIsNone(chat_args.goal)
        self.assertFalse(chat_args.chat)
        self.assertEqual(initial_chat_args.goal, "初始任务")
        self.assertTrue(initial_chat_args.chat)
        self.assertTrue(no_stream_args.no_stream)
        self.assertEqual(validation_args.validation_config, Path("checks.toml"))

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

    def test_invalid_named_validation_config_returns_clear_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            validation_path = root / "checks.toml"
            validation_path.write_text(
                """[validation.tasks.tests]
description = "bad"
command = "pytest -q"
required = true
""",
                encoding="utf-8",
            )
            stderr = StringIO()
            environment = {
                "LLM_API_KEY": "key",
                "LLM_BASE_URL": "https://example.test/v1",
                "LLM_MODEL": "model",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "inspect",
                        "--workspace",
                        str(root),
                        "--env-file",
                        str(root / "missing.env"),
                        "--validation-config",
                        "checks.toml",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("command 必须是字符串数组", stderr.getvalue())
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

    def test_creation_approval_shows_full_diff_and_requires_explicit_yes(self) -> None:
        preview = CreationPreview(
            path="src/new.py",
            diff="--- a/src/new.py\n+++ b/src/new.py\n+value = 1\n",
            updated_bytes=10,
        )
        output = StringIO()

        approved = request_creation_approval(
            preview,
            input_stream=InteractiveStringIO("yes\n"),
            output_stream=output,
        )

        self.assertTrue(approved)
        self.assertIn("SafePatch 新文件预览", output.getvalue())
        self.assertIn("src/new.py", output.getvalue())
        self.assertIn("+value = 1", output.getvalue())

    def test_non_interactive_input_cannot_approve_creation(self) -> None:
        preview = CreationPreview("new.py", "+new\n", 4)
        output = StringIO()

        approved = request_creation_approval(
            preview,
            input_stream=StringIO("yes\n"),
            output_stream=output,
        )

        self.assertFalse(approved)
        self.assertIn("不是交互式终端", output.getvalue())

    def test_deletion_approval_shows_full_diff_and_requires_explicit_yes(self) -> None:
        preview = DeletionPreview(
            path="src/old.py",
            diff="--- a/src/old.py\n+++ /dev/null\n-value = 1\n",
            original_bytes=10,
        )
        output = StringIO()

        approved = request_deletion_approval(
            preview,
            input_stream=InteractiveStringIO("是\n"),
            output_stream=output,
        )

        self.assertTrue(approved)
        self.assertIn("SafePatch 文件删除预览", output.getvalue())
        self.assertIn("src/old.py", output.getvalue())
        self.assertIn("+++ /dev/null", output.getvalue())

    def test_non_interactive_input_cannot_approve_deletion(self) -> None:
        preview = DeletionPreview("old.py", "-old\n", 4)
        output = StringIO()

        approved = request_deletion_approval(
            preview,
            input_stream=StringIO("yes\n"),
            output_stream=output,
        )

        self.assertFalse(approved)
        self.assertIn("不是交互式终端", output.getvalue())

    def test_batch_approval_shows_summary_full_diff_and_one_prompt(self) -> None:
        preview = BatchChangePreview(
            paths=("new.py", "old.py"),
            diff="--- /dev/null\n+++ b/new.py\n+new\n--- a/old.py\n+++ /dev/null\n-old\n",
            creations=1,
            replacements=0,
            deletions=1,
            original_bytes=4,
            updated_bytes=4,
        )
        output = StringIO()

        approved = request_batch_change_approval(
            preview,
            input_stream=InteractiveStringIO("yes\n"),
            output_stream=output,
        )

        rendered = output.getvalue()
        self.assertTrue(approved)
        self.assertIn("SafePatch 批量变更预览", rendered)
        self.assertIn("共 2 个文件；创建 1；替换 0；删除 1", rendered)
        self.assertIn("new.py, old.py", rendered)
        self.assertIn("+++ /dev/null", rendered)
        self.assertEqual(rendered.count("应用以上批量变更？"), 1)

    def test_non_interactive_input_cannot_approve_batch_change(self) -> None:
        preview = BatchChangePreview(("new.py",), "+new\n", 1, 0, 0, 0, 4)
        output = StringIO()

        approved = request_batch_change_approval(
            preview,
            input_stream=StringIO("yes\n"),
            output_stream=output,
        )

        self.assertFalse(approved)
        self.assertIn("不是交互式终端", output.getvalue())

    def test_rollback_approval_shows_ids_paths_and_reverse_diff(self) -> None:
        preview = RollbackPreview(
            change_ids=(2, 1),
            paths=("demo.py",),
            diff="--- a/demo.py\n+++ b/demo.py\n-new\n+old\n",
            original_bytes=4,
            updated_bytes=4,
            deleted_paths=("demo.py",),
            created_paths=("old.py",),
        )
        output = StringIO()

        approved = request_rollback_approval(
            preview,
            input_stream=InteractiveStringIO("y\n"),
            output_stream=output,
        )

        self.assertTrue(approved)
        self.assertIn("SafePatch 回滚预览", output.getvalue())
        self.assertIn("#2, #1", output.getvalue())
        self.assertIn("-new\n+old", output.getvalue())
        self.assertIn("将删除本会话创建的文件：demo.py", output.getvalue())
        self.assertIn("将恢复此前删除的文件：old.py", output.getvalue())

    def test_change_log_and_rollback_target_formatting(self) -> None:
        journal = ChangeJournal()
        record = journal.record(
            path="demo.py",
            before_text="old",
            after_text="new",
            diff="diff",
            replacements=1,
        )

        pending = format_change_log(journal)
        journal.record_test_result(True)
        tested = format_change_log(journal)
        journal.mark_rolled_back((record,))
        rolled_back = format_change_log(journal)

        created = ChangeJournal()
        created.record_creation(
            path="new.py",
            after_text="value = 1\n",
            diff="diff",
        )
        deleted = ChangeJournal()
        deleted.record_deletion(
            path="old.py",
            before_text="value = 1\n",
            diff="diff",
        )
        validated = ChangeJournal()
        validated.record(
            path="checked.py",
            before_text="old",
            after_text="new",
            diff="diff",
            replacements=1,
        )
        validated.record_validation_result("lint", True)
        validated.record_validation_result("tests", False)

        self.assertIn("#1 [待测试][替换] demo.py", pending)
        self.assertIn("[测试通过]", tested)
        self.assertIn("[已回滚]", rolled_back)
        self.assertIn("回滚结果待测试：demo.py", rolled_back)
        self.assertIn("[创建] new.py", format_change_log(created))
        self.assertIn("SHA-256 （不存在）", format_change_log(created))
        self.assertIn("[删除] old.py", format_change_log(deleted))
        self.assertIn("-> （不存在）", format_change_log(deleted))
        self.assertIn("[验证失败]", format_change_log(validated))
        self.assertIn("验证 lint=通过, tests=失败", format_change_log(validated))
        self.assertIsNone(parse_rollback_target("/rollback"))
        self.assertEqual(parse_rollback_target("/rollback 3"), 3)
        self.assertEqual(parse_rollback_target("/rollback ALL"), "all")
        self.assertEqual(parse_rollback_target("/rollback\tall"), "all")
        with self.assertRaisesRegex(ValueError, "修改编号"):
            parse_rollback_target("/rollback zero")

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

    def test_chat_can_show_and_rollback_session_changes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "demo.py"
            target.write_text("old\n", encoding="utf-8")
            journal = ChangeJournal()
            workspace = SafeWorkspace(
                root,
                replacement_approval=lambda _preview: True,
                change_journal=journal,
                rollback_approval=lambda _preview: True,
            )
            workspace.replace_text("demo.py", "old", "new")
            session = FakeSession()
            output = StringIO()

            exit_code = run_chat(  # type: ignore[arg-type]
                session,
                change_journal=journal,
                workspace=workspace,
                input_stream=InteractiveStringIO(
                    "/changes\n/rollback 1\n/changes\n/exit\n"
                ),
                output_stream=output,
            )

            restored = target.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(restored, "old\n")
        self.assertEqual(session.external_modifications, [("demo.py",)])
        self.assertIn("#1 [待测试]", output.getvalue())
        self.assertIn("已回滚修改 #1", output.getvalue())
        self.assertIn("#1 [已回滚]", output.getvalue())

    def test_chat_streams_final_answer_without_printing_it_twice(self) -> None:
        session = StreamingFakeSession()
        output = StringIO()

        exit_code = run_chat(  # type: ignore[arg-type]
            session,
            input_stream=InteractiveStringIO("任务\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(session.stream_enabled)
        self.assertEqual(output.getvalue().count("Agent> 已处理：任务"), 1)
        self.assertIn("[模型] 第 1 轮生成中", output.getvalue())

    def test_progress_renderer_shows_tool_status_and_verification_notice(self) -> None:
        output = StringIO()
        renderer = CLIProgressRenderer(output)

        renderer(AgentEvent(AgentEventKind.MODEL_START, 1))
        renderer(AgentEvent(AgentEventKind.TEXT_DELTA, 1, text="准备修改"))
        renderer(
            AgentEvent(
                AgentEventKind.MODEL_COMPLETE,
                1,
                has_tool_calls=True,
            )
        )
        renderer(
            AgentEvent(
                AgentEventKind.TOOL_START,
                1,
                tool_name="replace_text",
                tool_call_id="call-1",
            )
        )
        renderer(
            AgentEvent(
                AgentEventKind.TOOL_COMPLETE,
                1,
                tool_name="replace_text",
                tool_call_id="call-1",
                succeeded=True,
                duration_seconds=0.125,
            )
        )
        renderer(AgentEvent(AgentEventKind.VERIFICATION_REQUIRED, 1))

        rendered = output.getvalue()
        self.assertIn("Agent> 准备修改", rendered)
        self.assertIn("[工具] 开始 replace_text", rendered)
        self.assertIn("[工具] 完成 replace_text（成功，0.125 秒）", rendered)
        self.assertIn("尚未测试", rendered)
        self.assertFalse(renderer.final_answer_streamed)

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
