import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from safe_patch_agent.cli import build_parser, main, request_replacement_approval
from safe_patch_agent.workspace import ReplacementPreview


class InteractiveStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class CLITests(unittest.TestCase):
    def test_help_text_uses_chinese_headings(self) -> None:
        help_text = build_parser().format_help()

        self.assertTrue(help_text.startswith("用法："))
        self.assertIn("位置参数:", help_text)
        self.assertIn("选项:", help_text)

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


if __name__ == "__main__":
    unittest.main()
