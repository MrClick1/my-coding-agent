import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from safe_patch_agent.cli import build_parser, main


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


if __name__ == "__main__":
    unittest.main()
