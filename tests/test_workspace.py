import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from safe_patch_agent.messages import ToolCall
from safe_patch_agent.workspace import SafeWorkspace, WorkspaceError, build_read_only_registry


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "def greet(name):\n    return f'Hello, {name}'\n",
            encoding="utf-8",
        )
        (self.root / "README.md").write_text("# Demo\n", encoding="utf-8")
        self.workspace = SafeWorkspace(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_list_files_returns_relative_paths(self) -> None:
        result = self.workspace.list_files()
        paths = {entry["path"] for entry in result["entries"]}

        self.assertTrue(result["ok"])
        self.assertIn("README.md", paths)
        self.assertIn("src/app.py", paths)
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_read_file_adds_line_numbers(self) -> None:
        result = self.workspace.read_file("src/app.py", start_line=2, end_line=2)

        self.assertEqual(result["content"], "2:     return f'Hello, {name}'")
        self.assertEqual(result["start_line"], 2)
        self.assertEqual(result["end_line"], 2)

    def test_empty_file_has_an_explicit_zero_line_range(self) -> None:
        (self.root / "empty.txt").write_text("", encoding="utf-8")

        result = self.workspace.read_file("empty.txt")

        self.assertEqual(result["content"], "")
        self.assertEqual(result["start_line"], 0)
        self.assertEqual(result["end_line"], 0)
        self.assertEqual(result["total_lines"], 0)

    def test_parent_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkspaceError, "超出了"):
            self.workspace.read_file("../outside.txt")

    def test_absolute_path_is_rejected_even_when_it_is_inside_workspace(self) -> None:
        absolute = str(self.root / "README.md")
        with self.assertRaisesRegex(WorkspaceError, "绝对路径"):
            self.workspace.read_file(absolute)

    def test_tool_errors_are_returned_as_json(self) -> None:
        registry = build_read_only_registry(self.workspace)
        call = ToolCall(id="call-1", name="read_file", arguments={"path": "missing.py"})

        result = json.loads(registry.execute(call))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "tool_error")

    def test_large_file_is_rejected_before_reading(self) -> None:
        large_file = self.root / "large.txt"
        large_file.write_text("x" * 1_000_001, encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "读取上限"):
            self.workspace.read_file("large.txt")

    def test_sensitive_files_are_hidden_and_cannot_be_read_directly(self) -> None:
        (self.root / ".env").write_text("API_KEY=secret\n", encoding="utf-8")

        listed_paths = {entry["path"] for entry in self.workspace.list_files()["entries"]}

        self.assertNotIn(".env", listed_paths)
        with self.assertRaisesRegex(WorkspaceError, "敏感文件"):
            self.workspace.read_file(".env")

    def test_sensitive_directory_cannot_be_read_directly(self) -> None:
        git_directory = self.root / ".git"
        git_directory.mkdir()
        (git_directory / "config").write_text("token=secret\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "敏感目录"):
            self.workspace.read_file(".git/config")

    def test_env_example_remains_readable(self) -> None:
        (self.root / ".env.example").write_text("API_KEY=replace-me\n", encoding="utf-8")

        result = self.workspace.read_file(".env.example")

        self.assertTrue(result["ok"])

    def test_symlink_cannot_disguise_a_sensitive_file(self) -> None:
        secret = self.root / ".env"
        secret.write_text("API_KEY=secret\n", encoding="utf-8")
        link = self.root / "harmless.txt"
        try:
            link.symlink_to(secret)
        except OSError:
            self.skipTest("当前 Windows 环境不允许创建符号链接")

        with self.assertRaisesRegex(WorkspaceError, "敏感文件"):
            self.workspace.read_file("harmless.txt")

    def test_list_files_does_not_traverse_directory_symlink_outside_workspace(self) -> None:
        with TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory)
            (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
            link = self.root / "external"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("当前 Windows 环境不允许创建符号链接")

            listed_paths = {entry["path"] for entry in self.workspace.list_files()["entries"]}

        self.assertNotIn("external", listed_paths)
        self.assertNotIn("external/secret.txt", listed_paths)


if __name__ == "__main__":
    unittest.main()
