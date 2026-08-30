import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from safe_patch_agent.changes import ChangeJournal, ChangeKind
from safe_patch_agent.config import ValidationConfig, ValidationTask
from safe_patch_agent.messages import ToolCall
from safe_patch_agent.workspace import (
    BatchChangePreview,
    CreationPreview,
    DeletionPreview,
    ReplacementPreview,
    RollbackPreview,
    SafeWorkspace,
    WorkspaceError,
    build_agent_registry,
    build_read_only_registry,
)


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
        self.workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
        )

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

    def test_search_code_returns_stable_relative_matches(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "guide.md").write_text("Needle in docs\n", encoding="utf-8")
        (self.root / "src" / "app.py").write_text(
            "def greet(name):\n    return name\n# needle in source\n",
            encoding="utf-8",
        )

        result = self.workspace.search_code("needle")

        self.assertTrue(result["ok"])
        self.assertEqual(
            [(match["path"], match["line_number"], match["column"]) for match in result["matches"]],
            [("docs/guide.md", 1, 1), ("src/app.py", 3, 3)],
        )
        self.assertEqual(result["matched_files"], 2)
        self.assertFalse(result["truncated"])
        self.assertTrue(result["complete"])
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_search_code_supports_case_sensitive_literal_search(self) -> None:
        target = self.root / "literal.txt"
        target.write_text("Needle needle\nvalue = '.*'\n", encoding="utf-8")

        insensitive = self.workspace.search_code("needle", path="literal.txt")
        sensitive = self.workspace.search_code(
            "needle",
            path="literal.txt",
            case_sensitive=True,
        )
        literal = self.workspace.search_code(".*", path="literal.txt")

        self.assertEqual(insensitive["matches"][0]["column"], 1)
        self.assertEqual(len(insensitive["matches"]), 1)
        self.assertEqual(sensitive["matches"][0]["column"], 8)
        self.assertEqual(len(sensitive["matches"]), 1)
        self.assertEqual(literal["matches"][0]["line_number"], 2)

    def test_search_code_limits_search_to_requested_path(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "guide.md").write_text("scope-needle\n", encoding="utf-8")
        (self.root / "src" / "app.py").write_text("scope-needle\n", encoding="utf-8")

        directory_result = self.workspace.search_code("scope-needle", path="src")
        file_result = self.workspace.search_code("scope-needle", path="docs/guide.md")

        self.assertEqual(
            [match["path"] for match in directory_result["matches"]],
            ["src/app.py"],
        )
        self.assertEqual(
            [match["path"] for match in file_result["matches"]],
            ["docs/guide.md"],
        )

    def test_search_code_skips_sensitive_binary_and_large_files(self) -> None:
        (self.root / ".env").write_text("needle-secret\n", encoding="utf-8")
        (self.root / "secrets.py").write_text("needle-secret\n", encoding="utf-8")
        (self.root / ".env.example").write_text("needle-example\n", encoding="utf-8")
        git_directory = self.root / ".git"
        git_directory.mkdir()
        (git_directory / "config").write_text("needle-git\n", encoding="utf-8")
        uv_cache = self.root / ".uv-cache"
        uv_cache.mkdir()
        (uv_cache / "cached.txt").write_text("needle-cache\n", encoding="utf-8")
        (self.root / "bad.bin").write_bytes(b"\xffneedle-bad")
        (self.root / "null.bin").write_bytes(b"needle-null\x00")
        (self.root / "large.txt").write_text(
            "needle-large" + "x" * 1_000_000,
            encoding="utf-8",
        )

        result = self.workspace.search_code("needle")
        serialized = json.dumps(result, ensure_ascii=False)

        self.assertEqual(
            [match["path"] for match in result["matches"]],
            [".env.example"],
        )
        self.assertEqual(result["skipped_files"], 3)
        self.assertFalse(result["truncated"])
        self.assertFalse(result["complete"])
        self.assertNotIn("needle-secret", serialized)
        self.assertNotIn("needle-git", serialized)
        self.assertNotIn("needle-cache", serialized)
        self.assertNotIn("needle-bad", serialized)
        self.assertNotIn("needle-null", serialized)
        self.assertNotIn("needle-large", serialized)

    def test_search_code_reports_result_truncation_accurately(self) -> None:
        (self.root / "matches.txt").write_text(
            "limit-needle one\nlimit-needle two\nlimit-needle three\n",
            encoding="utf-8",
        )

        limited = self.workspace.search_code("limit-needle", max_results=2)
        exact = self.workspace.search_code("limit-needle", max_results=3)

        self.assertEqual(len(limited["matches"]), 2)
        self.assertTrue(limited["truncated"])
        self.assertEqual(len(exact["matches"]), 3)
        self.assertFalse(exact["truncated"])

    def test_search_code_bounds_long_match_content(self) -> None:
        (self.root / "long.txt").write_text(
            "x" * 800 + "needle" + "y" * 800 + "\n",
            encoding="utf-8",
        )

        result = self.workspace.search_code("needle", path="long.txt")
        match = result["matches"][0]

        self.assertEqual(match["column"], 801)
        self.assertTrue(match["content_truncated"])
        self.assertIn("needle", match["content"])
        self.assertLessEqual(len(match["content"]), 502)

    def test_search_code_counts_skipped_files_toward_byte_budget(self) -> None:
        budget_directory = self.root / "budget"
        budget_directory.mkdir()
        (budget_directory / "a-bad.bin").write_bytes(b"\xff" * 8)
        (budget_directory / "b-match.txt").write_text("needle\n", encoding="utf-8")

        with patch.object(SafeWorkspace, "_MAX_SEARCH_BYTES", 10):
            result = self.workspace.search_code("needle", path="budget")

        self.assertEqual(result["scanned_bytes"], 10)
        self.assertEqual(result["skipped_files"], 1)
        self.assertEqual(result["matches"], [])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["complete"])

    def test_search_code_limits_serialized_output_size(self) -> None:
        escaped_content = "needle" + "\x01" * 494
        (self.root / "escaped.txt").write_text(
            "\n".join([escaped_content] * 3),
            encoding="utf-8",
        )

        with patch.object(SafeWorkspace, "_MAX_SEARCH_OUTPUT_CHARS", 4_000):
            result = self.workspace.search_code("needle", path="escaped.txt")

        self.assertEqual(len(result["matches"]), 1)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 4_000)

    def test_search_code_limits_directory_traversal_work(self) -> None:
        traversal_directory = self.root / "traversal"
        traversal_directory.mkdir()
        for name in ("a.txt", "b.txt", "c.txt"):
            (traversal_directory / name).write_text("needle\n", encoding="utf-8")

        with patch.object(SafeWorkspace, "_MAX_SEARCH_ENTRIES", 2):
            result = self.workspace.search_code("needle", path="traversal")

        self.assertLessEqual(result["scanned_files"], 2)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["complete"])

    def test_search_code_limits_candidate_file_count(self) -> None:
        file_limit_directory = self.root / "file-limit"
        file_limit_directory.mkdir()
        for name in ("a.txt", "b.txt"):
            (file_limit_directory / name).write_text("needle\n", encoding="utf-8")

        with patch.object(SafeWorkspace, "_MAX_SEARCH_FILES", 1):
            result = self.workspace.search_code("needle", path="file-limit")

        self.assertEqual(result["scanned_files"], 1)
        self.assertEqual(len(result["matches"]), 1)
        self.assertTrue(result["truncated"])

    def test_search_code_limits_visited_directory_count(self) -> None:
        directory_limit_root = self.root / "directory-limit"
        directory_limit_root.mkdir()
        for name in ("a", "b"):
            child = directory_limit_root / name
            child.mkdir()
            (child / "match.txt").write_text("needle\n", encoding="utf-8")

        with patch.object(SafeWorkspace, "_MAX_SEARCH_DIRECTORIES", 1):
            result = self.workspace.search_code("needle", path="directory-limit")

        self.assertEqual(result["scanned_files"], 0)
        self.assertEqual(result["matches"], [])
        self.assertTrue(result["truncated"])

    def test_search_code_rejects_invalid_parameters(self) -> None:
        invalid_queries = [
            "",
            "   ",
            "line one\nline two",
            "line one\vline two",
            "nul\x00value",
            "x" * 201,
            123,
        ]
        for query in invalid_queries:
            with self.subTest(query=query), self.assertRaises(WorkspaceError):
                self.workspace.search_code(query)  # type: ignore[arg-type]

        with self.assertRaisesRegex(WorkspaceError, "case_sensitive"):
            self.workspace.search_code("needle", case_sensitive="yes")  # type: ignore[arg-type]

        for max_results in [0, 101, True, "2"]:
            with self.subTest(max_results=max_results), self.assertRaises(WorkspaceError):
                self.workspace.search_code("needle", max_results=max_results)  # type: ignore[arg-type]

        with self.assertRaisesRegex(WorkspaceError, "路径不存在"):
            self.workspace.search_code("needle", path="missing")
        with self.assertRaisesRegex(WorkspaceError, "绝对路径"):
            self.workspace.search_code("needle", path=str(self.root / "README.md"))
        with self.assertRaisesRegex(WorkspaceError, "超出了"):
            self.workspace.search_code("needle", path="../outside")

    def test_search_code_is_available_through_registry(self) -> None:
        registry = build_read_only_registry(self.workspace)
        call = ToolCall(
            id="call-search",
            name="search_code",
            arguments={"query": "greet", "path": "src"},
        )

        result = json.loads(registry.execute(call))
        missing_query = json.loads(
            registry.execute(ToolCall(id="call-invalid", name="search_code", arguments={}))
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["matches"][0]["path"], "src/app.py")
        self.assertFalse(missing_query["ok"])
        self.assertEqual(missing_query["error"]["type"], "invalid_arguments")
        search_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "search_code"
        )
        parameters = search_schema["function"]["parameters"]
        self.assertEqual(parameters["required"], ["query"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(parameters["properties"]["max_results"]["default"], 50)

    def test_replace_text_changes_exact_expected_occurrences(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old = 1\nold = 1\n", encoding="utf-8")

        result = self.workspace.replace_text(
            "replace.py",
            "old = 1",
            "new = 2",
            expected_replacements=2,
        )

        self.assertEqual(target.read_text(encoding="utf-8"), "new = 2\nnew = 2\n")
        self.assertEqual(result["replacements"], 2)
        self.assertEqual(result["path"], "replace.py")
        self.assertTrue(result["approved"])
        self.assertIn("--- a/replace.py", result["diff"])
        self.assertIn("+++ b/replace.py", result["diff"])

    def test_replace_text_sends_complete_diff_to_approval_callback(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        previews: list[ReplacementPreview] = []

        def approve(preview: ReplacementPreview) -> bool:
            previews.append(preview)
            return True

        workspace = SafeWorkspace(
            self.root,
            replacement_approval=approve,
        )

        workspace.replace_text("replace.py", "old", "new")

        self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].path, "replace.py")
        self.assertEqual(previews[0].replacements, 1)
        self.assertIn("-old", previews[0].diff)
        self.assertIn("+new", previews[0].diff)

    def test_replace_text_is_denied_without_approval_callback(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        workspace = SafeWorkspace(self.root)

        with self.assertRaisesRegex(WorkspaceError, "未配置用户确认"):
            workspace.replace_text("replace.py", "old", "new")

        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_successful_replace_is_recorded_in_session_journal(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
        )

        result = workspace.replace_text("replace.py", "old", "new")

        self.assertEqual(result["change_id"], 1)
        self.assertEqual(journal.record_count, 1)
        self.assertEqual(journal.summaries()[0].path, "replace.py")

    def test_journal_capacity_is_checked_before_approval_and_write(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        approval_called = False

        def approve(_preview: ReplacementPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(
            self.root,
            replacement_approval=approve,
            change_journal=ChangeJournal(max_stored_bytes=1),
        )

        with self.assertRaisesRegex(WorkspaceError, "可回滚内容"):
            workspace.replace_text("replace.py", "old", "new")

        self.assertFalse(approval_called)
        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_create_file_previews_exclusively_writes_and_records_creation(self) -> None:
        previews: list[CreationPreview] = []
        journal = ChangeJournal()

        def approve(preview: CreationPreview) -> bool:
            previews.append(preview)
            return True

        workspace = SafeWorkspace(
            self.root,
            creation_approval=approve,
            change_journal=journal,
        )

        result = workspace.create_file("src/new.py", "value = 1\n")

        target = self.root / "src" / "new.py"
        self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
        self.assertTrue(result["created"])
        self.assertEqual(result["change_id"], 1)
        self.assertEqual(previews[0].path, "src/new.py")
        self.assertIn("--- /dev/null", previews[0].diff)
        self.assertIn("+++ b/src/new.py", previews[0].diff)
        self.assertIn("+value = 1", previews[0].diff)
        summary = journal.summaries()[0]
        self.assertIs(summary.kind, ChangeKind.CREATE)
        self.assertIsNone(summary.before_sha256)

    def test_create_file_requires_approval_and_honors_rejection(self) -> None:
        target = self.root / "new.py"

        with self.assertRaisesRegex(WorkspaceError, "未配置用户确认"):
            self.workspace.create_file("new.py", "value = 1\n")

        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: False,
        )
        with self.assertRaisesRegex(WorkspaceError, "用户拒绝"):
            workspace.create_file("new.py", "value = 1\n")

        self.assertFalse(target.exists())

    def test_create_file_rejects_existing_sensitive_and_missing_parent_paths(
        self,
    ) -> None:
        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
        )

        with self.assertRaisesRegex(WorkspaceError, "已经存在"):
            workspace.create_file("README.md", "new\n")
        with self.assertRaisesRegex(WorkspaceError, "敏感文件"):
            workspace.create_file(".env", "TOKEN=value\n")
        with self.assertRaisesRegex(WorkspaceError, "父目录不存在"):
            workspace.create_file("missing/new.py", "value = 1\n")
        with self.assertRaisesRegex(WorkspaceError, "超出了"):
            workspace.create_file("../outside.py", "value = 1\n")

    def test_create_file_rejects_invalid_or_unpreviewable_content(self) -> None:
        approval_called = False

        def approve(_preview: CreationPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(self.root, creation_approval=approve)
        for content in ("", "bad\x00content"):
            with self.subTest(content=content), self.assertRaises(WorkspaceError):
                workspace.create_file("new.py", content)

        with self.assertRaisesRegex(WorkspaceError, "新文件超过"):
            workspace.create_file("new.py", "x" * 1_000_001)
        with (
            patch.object(SafeWorkspace, "_MAX_PATCH_PREVIEW_CHARS", 10),
            self.assertRaisesRegex(WorkspaceError, "无法完整展示"),
        ):
            workspace.create_file("new.py", "value = 1\n")

        self.assertFalse(approval_called)
        self.assertFalse((self.root / "new.py").exists())

    def test_create_file_checks_journal_capacity_before_approval(self) -> None:
        approval_called = False

        def approve(_preview: CreationPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(
            self.root,
            creation_approval=approve,
            change_journal=ChangeJournal(max_stored_bytes=1),
        )

        with self.assertRaisesRegex(WorkspaceError, "可回滚内容"):
            workspace.create_file("new.py", "value = 1\n")

        self.assertFalse(approval_called)
        self.assertFalse((self.root / "new.py").exists())

    def test_create_file_rejects_target_created_during_approval(self) -> None:
        target = self.root / "new.py"

        def create_during_approval(_preview: CreationPreview) -> bool:
            target.write_text("external\n", encoding="utf-8")
            return True

        workspace = SafeWorkspace(
            self.root,
            creation_approval=create_during_approval,
        )

        with self.assertRaisesRegex(WorkspaceError, "确认期间"):
            workspace.create_file("new.py", "agent\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")

    def test_exclusive_create_preserves_file_that_wins_final_race(self) -> None:
        target = self.root / "new.py"
        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
        )

        def competing_link(_source: Path, destination: Path) -> None:
            Path(destination).write_text("external\n", encoding="utf-8")
            raise FileExistsError

        with (
            patch("safe_patch_agent.workspace.os.link", side_effect=competing_link),
            self.assertRaisesRegex(WorkspaceError, "拒绝覆盖"),
        ):
            workspace.create_file("new.py", "agent\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")

    def test_delete_file_previews_checks_and_records_deletion(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("value = 1\n", encoding="utf-8")
        expected_bytes = len(target.read_bytes())
        previews: list[DeletionPreview] = []
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            deletion_approval=lambda preview: previews.append(preview) is None,
            change_journal=journal,
        )

        result = workspace.delete_file("obsolete.py")

        self.assertFalse(target.exists())
        self.assertTrue(result["deleted"])
        self.assertEqual(result["change_id"], 1)
        self.assertEqual(previews[0].path, "obsolete.py")
        self.assertEqual(previews[0].original_bytes, expected_bytes)
        self.assertIn("--- a/obsolete.py", previews[0].diff)
        self.assertIn("+++ /dev/null", previews[0].diff)
        self.assertIn("-value = 1", previews[0].diff)
        summary = journal.summaries()[0]
        self.assertIs(summary.kind, ChangeKind.DELETE)
        self.assertIsNone(summary.after_sha256)

    def test_delete_file_requires_approval_and_honors_rejection(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("value = 1\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "未配置用户确认"):
            self.workspace.delete_file("obsolete.py")

        workspace = SafeWorkspace(
            self.root,
            deletion_approval=lambda _preview: False,
        )
        with self.assertRaisesRegex(WorkspaceError, "用户拒绝"):
            workspace.delete_file("obsolete.py")

        self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")

    def test_delete_file_rejects_invalid_or_unpreviewable_text(self) -> None:
        binary = self.root / "binary.dat"
        binary.write_bytes(b"bad\x00content")
        large = self.root / "large.txt"
        large.write_bytes(b"x" * 1_000_001)
        long_diff = self.root / "long.txt"
        long_diff.write_text("value = 1\n", encoding="utf-8")
        approval_called = False

        def approve(_preview: DeletionPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(self.root, deletion_approval=approve)

        with self.assertRaisesRegex(WorkspaceError, "NUL"):
            workspace.delete_file("binary.dat")
        with self.assertRaisesRegex(WorkspaceError, "读取上限"):
            workspace.delete_file("large.txt")
        with (
            patch.object(SafeWorkspace, "_MAX_PATCH_PREVIEW_CHARS", 10),
            self.assertRaisesRegex(WorkspaceError, "无法完整展示"),
        ):
            workspace.delete_file("long.txt")

        self.assertFalse(approval_called)
        self.assertTrue(binary.exists())
        self.assertTrue(large.exists())
        self.assertTrue(long_diff.exists())

    def test_delete_file_checks_journal_capacity_before_approval(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("value = 1\n", encoding="utf-8")
        approval_called = False

        def approve(_preview: DeletionPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(
            self.root,
            deletion_approval=approve,
            change_journal=ChangeJournal(max_stored_bytes=1),
        )

        with self.assertRaisesRegex(WorkspaceError, "可回滚内容"):
            workspace.delete_file("obsolete.py")

        self.assertFalse(approval_called)
        self.assertTrue(target.exists())

    def test_delete_file_rejects_content_changed_during_approval(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("agent\n", encoding="utf-8")

        def change_during_approval(_preview: DeletionPreview) -> bool:
            target.write_text("external\n", encoding="utf-8")
            return True

        workspace = SafeWorkspace(
            self.root,
            deletion_approval=change_during_approval,
        )

        with self.assertRaisesRegex(WorkspaceError, "确认期间发生变化"):
            workspace.delete_file("obsolete.py")

        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")

    def test_checked_delete_restores_file_that_changes_in_final_race(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("agent\n", encoding="utf-8")
        workspace = SafeWorkspace(
            self.root,
            deletion_approval=lambda _preview: True,
        )
        real_replace = os.replace

        def competing_replace(source: Path, destination: Path) -> None:
            target.write_text("external\n", encoding="utf-8")
            real_replace(source, destination)

        with (
            patch("safe_patch_agent.workspace.os.replace", side_effect=competing_replace),
            self.assertRaisesRegex(WorkspaceError, "删除提交时发生变化"),
        ):
            workspace.delete_file("obsolete.py")

        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")

    def test_change_set_previews_and_applies_create_replace_delete_once(self) -> None:
        replace_target = self.root / "replace.py"
        delete_target = self.root / "obsolete.py"
        replace_target.write_text("old\n", encoding="utf-8")
        delete_target.write_text("obsolete\n", encoding="utf-8")
        previews: list[BatchChangePreview] = []
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=lambda preview: previews.append(preview) is None,
            change_journal=journal,
        )
        operations = [
            {"kind": "create", "path": "new.py", "content": "created\n"},
            {
                "kind": "replace",
                "path": "replace.py",
                "old_text": "old",
                "new_text": "updated",
            },
            {"kind": "delete", "path": "obsolete.py"},
        ]

        result = workspace.apply_change_set(operations)

        self.assertEqual((self.root / "new.py").read_text(encoding="utf-8"), "created\n")
        self.assertEqual(replace_target.read_text(encoding="utf-8"), "updated\n")
        self.assertFalse(delete_target.exists())
        self.assertEqual(result["change_ids"], (1, 2, 3))
        self.assertEqual(result["creations"], 1)
        self.assertEqual(result["replacements"], 1)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].paths, ("new.py", "replace.py", "obsolete.py"))
        self.assertIn("+++ b/new.py", previews[0].diff)
        self.assertIn("--- a/replace.py", previews[0].diff)
        self.assertIn("+++ /dev/null", previews[0].diff)
        self.assertEqual(
            tuple(summary.kind for summary in journal.summaries()),
            (ChangeKind.CREATE, ChangeKind.REPLACE, ChangeKind.DELETE),
        )

    def test_change_set_requires_approval_and_rejects_invalid_operations(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        operations = [
            {
                "kind": "replace",
                "path": "replace.py",
                "old_text": "old",
                "new_text": "new",
            }
        ]

        with self.assertRaisesRegex(WorkspaceError, "未配置用户确认"):
            self.workspace.apply_change_set(operations)
        rejected = SafeWorkspace(
            self.root,
            batch_change_approval=lambda _preview: False,
        )
        with self.assertRaisesRegex(WorkspaceError, "用户拒绝"):
            rejected.apply_change_set(operations)
        with self.assertRaisesRegex(WorkspaceError, "1 到 20"):
            rejected.apply_change_set([])
        with self.assertRaisesRegex(WorkspaceError, "不能重复操作"):
            rejected.apply_change_set([operations[0], operations[0]])
        with self.assertRaisesRegex(WorkspaceError, "缺少字段"):
            rejected.apply_change_set(
                [{"kind": "create", "path": "new.py"}]
            )
        with self.assertRaisesRegex(WorkspaceError, "多余字段"):
            rejected.apply_change_set(
                [{"kind": "delete", "path": "replace.py", "content": "bad"}]
            )

        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_change_set_checks_all_journal_capacity_before_approval(self) -> None:
        first = self.root / "first.py"
        second = self.root / "second.py"
        first.write_text("old\n", encoding="utf-8")
        second.write_text("old\n", encoding="utf-8")
        approval_called = False

        def approve(_preview: BatchChangePreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=approve,
            change_journal=ChangeJournal(max_records=1),
        )
        operations = [
            {
                "kind": "replace",
                "path": path,
                "old_text": "old",
                "new_text": "new",
            }
            for path in ("first.py", "second.py")
        ]

        with self.assertRaisesRegex(WorkspaceError, "1 条上限"):
            workspace.apply_change_set(operations)

        self.assertFalse(approval_called)
        self.assertEqual(first.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "old\n")

    def test_change_set_requires_the_combined_diff_to_be_fully_previewable(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        approval_called = False

        def approve(_preview: BatchChangePreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=approve,
        )
        operations = [
            {"kind": "create", "path": "new.py", "content": "created\n"},
            {
                "kind": "replace",
                "path": "replace.py",
                "old_text": "old",
                "new_text": "updated",
            },
        ]

        with (
            patch.object(SafeWorkspace, "_MAX_PATCH_PREVIEW_CHARS", 20),
            self.assertRaisesRegex(WorkspaceError, "批量变更预览超过"),
        ):
            workspace.apply_change_set(operations)

        self.assertFalse(approval_called)
        self.assertFalse((self.root / "new.py").exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_change_set_rechecks_every_path_after_approval(self) -> None:
        replace_target = self.root / "replace.py"
        delete_target = self.root / "obsolete.py"
        replace_target.write_text("old\n", encoding="utf-8")
        delete_target.write_text("obsolete\n", encoding="utf-8")

        def change_during_approval(_preview: BatchChangePreview) -> bool:
            replace_target.write_text("external\n", encoding="utf-8")
            return True

        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=change_during_approval,
        )
        operations = [
            {"kind": "create", "path": "new.py", "content": "created\n"},
            {
                "kind": "replace",
                "path": "replace.py",
                "old_text": "old",
                "new_text": "updated",
            },
            {"kind": "delete", "path": "obsolete.py"},
        ]

        with self.assertRaisesRegex(WorkspaceError, "确认期间文件发生变化"):
            workspace.apply_change_set(operations)

        self.assertFalse((self.root / "new.py").exists())
        self.assertEqual(replace_target.read_text(encoding="utf-8"), "external\n")
        self.assertEqual(delete_target.read_text(encoding="utf-8"), "obsolete\n")

    def test_change_set_restores_applied_files_when_later_change_fails(self) -> None:
        replace_target = self.root / "replace.py"
        delete_target = self.root / "obsolete.py"
        replace_target.write_text("old\n", encoding="utf-8")
        delete_target.write_text("obsolete\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=lambda _preview: True,
            change_journal=journal,
        )
        operations = [
            {"kind": "create", "path": "new.py", "content": "created\n"},
            {
                "kind": "replace",
                "path": "replace.py",
                "old_text": "old",
                "new_text": "updated",
            },
            {"kind": "delete", "path": "obsolete.py"},
        ]
        checked_delete = workspace._checked_delete

        def fail_obsolete_delete(target: Path, content: bytes) -> None:
            if target == delete_target:
                raise WorkspaceError("模拟最后一项失败")
            checked_delete(target, content)

        with (
            patch.object(
                workspace,
                "_checked_delete",
                side_effect=fail_obsolete_delete,
            ),
            self.assertRaisesRegex(WorkspaceError, "已恢复本批次"),
        ):
            workspace.apply_change_set(operations)

        self.assertFalse((self.root / "new.py").exists())
        self.assertEqual(replace_target.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(delete_target.read_text(encoding="utf-8"), "obsolete\n")
        self.assertEqual(journal.record_count, 0)

    def test_change_set_records_integrate_with_full_rollback(self) -> None:
        replace_target = self.root / "replace.py"
        delete_target = self.root / "obsolete.py"
        replace_target.write_text("old\n", encoding="utf-8")
        delete_target.write_text("obsolete\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.apply_change_set(
            [
                {"kind": "create", "path": "new.py", "content": "created\n"},
                {
                    "kind": "replace",
                    "path": "replace.py",
                    "old_text": "old",
                    "new_text": "updated",
                },
                {"kind": "delete", "path": "obsolete.py"},
            ]
        )

        result = workspace.rollback_changes("all")

        self.assertFalse((self.root / "new.py").exists())
        self.assertEqual(replace_target.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(delete_target.read_text(encoding="utf-8"), "obsolete\n")
        self.assertEqual(result["change_ids"], (3, 2, 1))
        self.assertTrue(all(item.rolled_back for item in journal.summaries()))

    def test_rollback_latest_restores_content_after_confirmation(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        previews: list[RollbackPreview] = []
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda preview: previews.append(preview) is None,
        )
        workspace.replace_text("replace.py", "old", "new")

        result = workspace.rollback_changes()

        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(result["change_ids"], (1,))
        self.assertEqual(result["paths"], ("replace.py",))
        self.assertEqual(previews[0].change_ids, (1,))
        self.assertIn("-new", previews[0].diff)
        self.assertIn("+old", previews[0].diff)
        self.assertTrue(journal.summaries()[0].rolled_back)
        self.assertEqual(journal.pending_rollback_paths, ("replace.py",))

    def test_rollback_creation_deletes_only_the_unchanged_created_file(self) -> None:
        previews: list[RollbackPreview] = []
        journal = ChangeJournal()

        def approve_rollback(preview: RollbackPreview) -> bool:
            previews.append(preview)
            return True

        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=approve_rollback,
        )
        workspace.create_file("new.py", "value = 1\n")

        result = workspace.rollback_changes()

        self.assertFalse((self.root / "new.py").exists())
        self.assertEqual(result["deleted_paths"], ("new.py",))
        self.assertEqual(previews[0].deleted_paths, ("new.py",))
        self.assertIn("-value = 1", previews[0].diff)
        self.assertIn("+++ /dev/null", previews[0].diff)
        self.assertTrue(journal.summaries()[0].rolled_back)

    def test_rollback_deletion_recreates_only_the_still_missing_file(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("value = 1\n", encoding="utf-8")
        previews: list[RollbackPreview] = []
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            deletion_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda preview: previews.append(preview) is None,
        )
        workspace.delete_file("obsolete.py")

        result = workspace.rollback_changes()

        self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
        self.assertEqual(result["created_paths"], ("obsolete.py",))
        self.assertEqual(previews[0].created_paths, ("obsolete.py",))
        self.assertIn("--- /dev/null", previews[0].diff)
        self.assertIn("+++ b/obsolete.py", previews[0].diff)
        self.assertTrue(journal.summaries()[0].rolled_back)

    def test_rollback_deletion_rejects_externally_recreated_file(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("agent\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            deletion_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.delete_file("obsolete.py")
        target.write_text("external\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "已在修改后发生变化"):
            workspace.rollback_changes()

        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")
        self.assertFalse(journal.summaries()[0].rolled_back)

    def test_rollback_all_of_create_then_delete_has_no_file_change(self) -> None:
        journal = ChangeJournal()
        previews: list[RollbackPreview] = []
        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
            deletion_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda preview: previews.append(preview) is None,
        )
        workspace.create_file("temporary.py", "value = 1\n")
        workspace.delete_file("temporary.py")

        result = workspace.rollback_changes("all")

        self.assertFalse((self.root / "temporary.py").exists())
        self.assertEqual(result["deleted_paths"], ())
        self.assertEqual(result["created_paths"], ())
        self.assertIn("净文件状态不变", previews[0].diff)
        self.assertTrue(all(item.rolled_back for item in journal.summaries()))

    def test_rollback_all_unwinds_replace_then_deletes_created_file(self) -> None:
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            creation_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.create_file("new.py", "one\n")
        workspace.replace_text("new.py", "one", "two")

        result = workspace.rollback_changes("all")

        self.assertFalse((self.root / "new.py").exists())
        self.assertEqual(result["change_ids"], (2, 1))
        self.assertEqual(result["deleted_paths"], ("new.py",))

    def test_rollback_creation_rejects_externally_modified_file(self) -> None:
        target = self.root / "new.py"
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.create_file("new.py", "agent\n")
        target.write_text("external\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "已在修改后发生变化"):
            workspace.rollback_changes()

        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")
        self.assertFalse(journal.summaries()[0].rolled_back)

    def test_rollback_all_unwinds_repeated_changes_in_reverse_order(self) -> None:
        target = self.root / "replace.py"
        target.write_text("one\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.replace_text("replace.py", "one", "two")
        workspace.replace_text("replace.py", "two", "three")

        result = workspace.rollback_changes("all")

        self.assertEqual(target.read_text(encoding="utf-8"), "one\n")
        self.assertEqual(result["change_ids"], (2, 1))
        self.assertTrue(all(item.rolled_back for item in journal.summaries()))

    def test_rollback_rejects_content_changed_after_agent_write(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        approval_called = False
        journal = ChangeJournal()

        def approve_rollback(_preview: RollbackPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=approve_rollback,
        )
        workspace.replace_text("replace.py", "old", "new")
        target.write_text("external\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "已在修改后发生变化"):
            workspace.rollback_changes()

        self.assertFalse(approval_called)
        self.assertEqual(target.read_text(encoding="utf-8"), "external\n")
        self.assertFalse(journal.summaries()[0].rolled_back)

    def test_rollback_rechecks_content_after_confirmation(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        journal = ChangeJournal()

        def change_during_approval(_preview: RollbackPreview) -> bool:
            target.write_text("concurrent\n", encoding="utf-8")
            return True

        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=change_during_approval,
        )
        workspace.replace_text("replace.py", "old", "new")

        with self.assertRaisesRegex(WorkspaceError, "确认期间发生变化"):
            workspace.rollback_changes()

        self.assertEqual(target.read_text(encoding="utf-8"), "concurrent\n")
        self.assertFalse(journal.summaries()[0].rolled_back)

    def test_multi_file_rollback_restores_earlier_writes_if_a_write_fails(
        self,
    ) -> None:
        first = self.root / "a.py"
        second = self.root / "b.py"
        first.write_text("old-a\n", encoding="utf-8")
        second.write_text("old-b\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.replace_text("a.py", "old-a", "new-a")
        workspace.replace_text("b.py", "old-b", "new-b")
        atomic_write = workspace._atomic_write
        write_count = 0

        def fail_second_write(target: Path, content: bytes) -> None:
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise WorkspaceError("模拟写入失败")
            atomic_write(target, content)

        with (
            patch.object(workspace, "_atomic_write", side_effect=fail_second_write),
            self.assertRaisesRegex(WorkspaceError, "已恢复本次写入"),
        ):
            workspace.rollback_changes("all")

        self.assertEqual(first.read_text(encoding="utf-8"), "new-a\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "new-b\n")
        self.assertTrue(all(not item.rolled_back for item in journal.summaries()))

    def test_multi_file_creation_rollback_restores_deleted_file_on_failure(
        self,
    ) -> None:
        first = self.root / "a.py"
        second = self.root / "b.py"
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
            change_journal=journal,
            rollback_approval=lambda _preview: True,
        )
        workspace.create_file("a.py", "created-a\n")
        workspace.create_file("b.py", "created-b\n")
        checked_delete = workspace._checked_delete

        def fail_second_delete(target: Path, content: bytes) -> None:
            if target == second:
                raise WorkspaceError("模拟删除失败")
            checked_delete(target, content)

        with (
            patch.object(workspace, "_checked_delete", side_effect=fail_second_delete),
            self.assertRaisesRegex(WorkspaceError, "已恢复本次写入"),
        ):
            workspace.rollback_changes("all")

        self.assertEqual(first.read_text(encoding="utf-8"), "created-a\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "created-b\n")
        self.assertTrue(all(not item.rolled_back for item in journal.summaries()))

    def test_workspace_rejects_non_callable_approval_handler(self) -> None:
        with self.assertRaisesRegex(WorkspaceError, "必须是可调用对象"):
            SafeWorkspace(self.root, replacement_approval=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(WorkspaceError, "creation_approval"):
            SafeWorkspace(self.root, creation_approval=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(WorkspaceError, "deletion_approval"):
            SafeWorkspace(self.root, deletion_approval=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(WorkspaceError, "batch_change_approval"):
            SafeWorkspace(self.root, batch_change_approval=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(WorkspaceError, "validation_config"):
            SafeWorkspace(self.root, validation_config=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(WorkspaceError, "protected_paths"):
            SafeWorkspace(self.root, protected_paths="config.toml")  # type: ignore[arg-type]

    def test_replace_text_honors_rejection(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: False,
        )

        with self.assertRaisesRegex(WorkspaceError, "用户拒绝"):
            workspace.replace_text("replace.py", "old", "new")

        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_replace_text_rejects_file_changed_during_approval(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")

        def change_file_during_approval(_preview: ReplacementPreview) -> bool:
            target.write_text("concurrent change\n", encoding="utf-8")
            return True

        workspace = SafeWorkspace(
            self.root,
            replacement_approval=change_file_during_approval,
        )

        with self.assertRaisesRegex(WorkspaceError, "确认期间发生变化"):
            workspace.replace_text("replace.py", "old", "new")

        self.assertEqual(target.read_text(encoding="utf-8"), "concurrent change\n")

    def test_replace_text_rejects_preview_that_cannot_be_shown_completely(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        approval_called = False

        def approve(_preview: ReplacementPreview) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        workspace = SafeWorkspace(self.root, replacement_approval=approve)
        with (
            patch.object(SafeWorkspace, "_MAX_PATCH_PREVIEW_CHARS", 10),
            self.assertRaisesRegex(WorkspaceError, "无法完整展示"),
        ):
            workspace.replace_text("replace.py", "old", "new")

        self.assertFalse(approval_called)
        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_replacement_diff_marks_missing_final_newline(self) -> None:
        diff = SafeWorkspace._replacement_diff("demo.txt", "old\n", "old")

        self.assertIn("文件末尾无换行符", diff)

    def test_empty_file_deletion_diff_still_shows_nonexistent_target(self) -> None:
        diff = SafeWorkspace._deletion_diff("empty.txt", "")

        self.assertEqual(diff, "--- a/empty.txt\n+++ /dev/null\n")

    def test_replace_text_mismatch_leaves_file_unchanged(self) -> None:
        target = self.root / "replace.py"
        original = "same\nsame\n"
        target.write_text(original, encoding="utf-8")

        for expected_replacements in (1, 3):
            with (
                self.subTest(expected_replacements=expected_replacements),
                self.assertRaisesRegex(WorkspaceError, "实际 2 次"),
            ):
                self.workspace.replace_text(
                    "replace.py",
                    "same",
                    "changed",
                    expected_replacements=expected_replacements,
                )
            self.assertEqual(target.read_text(encoding="utf-8"), original)

        with self.assertRaisesRegex(WorkspaceError, "实际 0 次"):
            self.workspace.replace_text("replace.py", "missing", "changed")
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_replace_text_rejects_invalid_parameters_without_writing(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")

        invalid_calls = [
            {"old_text": "", "new_text": "new"},
            {"old_text": "old", "new_text": "old"},
            {"old_text": "old", "new_text": "new", "expected_replacements": 0},
            {"old_text": "old", "new_text": "new", "expected_replacements": 101},
            {"old_text": "old", "new_text": "new", "expected_replacements": True},
            {"old_text": "old\x00", "new_text": "new"},
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(WorkspaceError):
                self.workspace.replace_text("replace.py", **arguments)  # type: ignore[arg-type]

        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_replace_text_rejects_oversized_result_without_writing(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspaceError, "替换后的文件超过"):
            self.workspace.replace_text("replace.py", "old", "x" * 1_000_001)

        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_replace_text_rejects_sensitive_and_non_text_files(self) -> None:
        (self.root / ".env").write_text("TOKEN=old\n", encoding="utf-8")
        (self.root / "binary.bin").write_bytes(b"old\x00value")

        with self.assertRaisesRegex(WorkspaceError, "敏感文件"):
            self.workspace.replace_text(".env", "old", "new")
        with self.assertRaisesRegex(WorkspaceError, "NUL"):
            self.workspace.replace_text("binary.bin", "old", "new")

    def test_replace_text_registry_enforces_read_before_write(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        registry = build_agent_registry(self.workspace)
        replace_call = ToolCall(
            id="replace",
            name="replace_text",
            arguments={
                "path": "replace.py",
                "old_text": "old",
                "new_text": "new",
            },
        )

        blocked = json.loads(registry.execute(replace_call))
        registry.execute(
            ToolCall(id="read", name="read_file", arguments={"path": "replace.py"})
        )
        allowed = json.loads(registry.execute(replace_call))

        self.assertFalse(blocked["ok"])
        self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
        self.assertTrue(allowed["ok"])
        self.assertEqual(registry.state.snapshot().modified_files, ("replace.py",))

    def test_replace_text_uses_canonical_path_for_read_authorization(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        (self.root / "subdirectory").mkdir()
        registry = build_agent_registry(self.workspace)
        registry.execute(
            ToolCall(
                id="read",
                name="read_file",
                arguments={"path": "./replace.py"},
            )
        )

        result = json.loads(
            registry.execute(
                ToolCall(
                    id="replace",
                    name="replace_text",
                    arguments={
                        "path": "subdirectory/../replace.py",
                        "old_text": "old",
                        "new_text": "new",
                    },
                )
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
        self.assertEqual(registry.state.snapshot().read_files, ("replace.py",))
        self.assertEqual(registry.state.snapshot().modified_files, ("replace.py",))

    def test_failed_replace_is_not_recorded_as_modified(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old old\n", encoding="utf-8")
        registry = build_agent_registry(self.workspace)
        registry.execute(
            ToolCall(id="read", name="read_file", arguments={"path": "replace.py"})
        )

        result = json.loads(
            registry.execute(
                ToolCall(
                    id="replace",
                    name="replace_text",
                    arguments={
                        "path": "replace.py",
                        "old_text": "old",
                        "new_text": "new",
                    },
                )
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(target.read_text(encoding="utf-8"), "old old\n")
        self.assertEqual(registry.state.snapshot().modified_files, ())

    def test_create_file_registry_does_not_require_read_and_records_write(self) -> None:
        workspace = SafeWorkspace(
            self.root,
            creation_approval=lambda _preview: True,
        )
        registry = build_agent_registry(workspace)

        result = json.loads(
            registry.execute(
                ToolCall(
                    id="create",
                    name="create_file",
                    arguments={"path": "new.py", "content": "value = 1\n"},
                )
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            (self.root / "new.py").read_text(encoding="utf-8"),
            "value = 1\n",
        )
        snapshot = registry.state.snapshot()
        self.assertEqual(snapshot.read_files, ())
        self.assertEqual(snapshot.modified_files, ("new.py",))
        self.assertEqual(snapshot.blocked_write_attempts, 0)
        self.assertTrue(snapshot.has_unverified_changes)

    def test_delete_file_registry_requires_read_and_records_write(self) -> None:
        target = self.root / "obsolete.py"
        target.write_text("value = 1\n", encoding="utf-8")
        workspace = SafeWorkspace(
            self.root,
            deletion_approval=lambda _preview: True,
        )
        registry = build_agent_registry(workspace)

        blocked = json.loads(
            registry.execute(
                ToolCall(
                    id="blocked-delete",
                    name="delete_file",
                    arguments={"path": "obsolete.py"},
                )
            )
        )
        registry.execute(
            ToolCall(
                id="read",
                name="read_file",
                arguments={"path": "obsolete.py"},
            )
        )
        deleted = json.loads(
            registry.execute(
                ToolCall(
                    id="delete",
                    name="delete_file",
                    arguments={"path": "obsolete.py"},
                )
            )
        )

        self.assertFalse(blocked["ok"])
        self.assertTrue(deleted["ok"])
        self.assertFalse(target.exists())
        snapshot = registry.state.snapshot()
        self.assertEqual(snapshot.read_files, ("obsolete.py",))
        self.assertEqual(snapshot.modified_files, ("obsolete.py",))
        self.assertTrue(snapshot.has_unverified_changes)

    def test_change_set_registry_checks_reads_and_records_all_paths(self) -> None:
        replace_target = self.root / "replace.py"
        delete_target = self.root / "obsolete.py"
        replace_target.write_text("old\n", encoding="utf-8")
        delete_target.write_text("obsolete\n", encoding="utf-8")
        workspace = SafeWorkspace(
            self.root,
            batch_change_approval=lambda _preview: True,
        )
        registry = build_agent_registry(workspace)
        arguments = {
            "operations": [
                {"kind": "create", "path": "new.py", "content": "created\n"},
                {
                    "kind": "replace",
                    "path": "replace.py",
                    "old_text": "old",
                    "new_text": "updated",
                },
                {"kind": "delete", "path": "obsolete.py"},
            ]
        }

        blocked = json.loads(
            registry.execute(
                ToolCall(
                    id="blocked-batch",
                    name="apply_change_set",
                    arguments=arguments,
                )
            )
        )
        for path in ("replace.py", "obsolete.py"):
            registry.execute(
                ToolCall(id=f"read-{path}", name="read_file", arguments={"path": path})
            )
        applied = json.loads(
            registry.execute(
                ToolCall(
                    id="batch",
                    name="apply_change_set",
                    arguments=arguments,
                )
            )
        )

        self.assertFalse(blocked["ok"])
        self.assertTrue(applied["ok"])
        snapshot = registry.state.snapshot()
        self.assertEqual(snapshot.read_files, ("obsolete.py", "replace.py"))
        self.assertEqual(
            snapshot.modified_files,
            ("new.py", "obsolete.py", "replace.py"),
        )
        self.assertEqual(snapshot.blocked_write_attempts, 1)
        self.assertTrue(snapshot.has_unverified_changes)

    def test_agent_registry_exposes_replace_text_schema(self) -> None:
        registry = build_agent_registry(self.workspace)
        tool_names = {schema["function"]["name"] for schema in registry.schemas()}
        replace_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "replace_text"
        )["function"]["parameters"]
        create_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "create_file"
        )["function"]["parameters"]
        delete_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "delete_file"
        )["function"]["parameters"]
        batch_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "apply_change_set"
        )["function"]["parameters"]
        validation_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "run_validation"
        )["function"]["parameters"]

        self.assertEqual(
            tool_names,
            {
                "list_files",
                "read_file",
                "search_code",
                "replace_text",
                "create_file",
                "delete_file",
                "apply_change_set",
                "run_validation",
                "run_tests",
            },
        )
        self.assertEqual(
            replace_schema["required"],
            ["path", "old_text", "new_text"],
        )
        self.assertEqual(
            replace_schema["properties"]["expected_replacements"]["default"],
            1,
        )
        self.assertEqual(create_schema["required"], ["path", "content"])
        self.assertEqual(create_schema["properties"]["content"]["minLength"], 1)
        self.assertEqual(delete_schema["required"], ["path"])
        self.assertEqual(batch_schema["required"], ["operations"])
        self.assertEqual(batch_schema["properties"]["operations"]["maxItems"], 20)
        self.assertEqual(validation_schema["required"], ["name"])
        self.assertEqual(
            validation_schema["properties"]["name"]["enum"],
            ["tests"],
        )

    def test_named_required_validations_use_frozen_commands_and_state_gate(self) -> None:
        validation_config = ValidationConfig(
            tasks=(
                ValidationTask(
                    "tests",
                    "Run tests",
                    ("{python}", "-m", "pytest", "-q"),
                    required=True,
                ),
                ValidationTask(
                    "lint",
                    "Run lint",
                    ("{python}", "-m", "ruff", "check", "."),
                    timeout_seconds=30,
                    required=True,
                ),
            )
        )
        workspace = SafeWorkspace(
            self.root,
            validation_config=validation_config,
        )
        registry = build_agent_registry(workspace)
        registry.state.mark_file_modified("src/app.py")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="passed\n",
        )

        with patch("safe_patch_agent.workspace.subprocess.run", return_value=completed) as run:
            lint = json.loads(
                registry.execute(
                    ToolCall(
                        id="lint",
                        name="run_validation",
                        arguments={"name": "lint"},
                    )
                )
            )
            after_lint = registry.state.snapshot()
            tests = json.loads(
                registry.execute(ToolCall(id="tests", name="run_tests", arguments={}))
            )

        self.assertTrue(lint["passed"])
        self.assertEqual(lint["name"], "lint")
        self.assertEqual(
            run.call_args_list[0].args[0],
            [sys.executable, "-m", "ruff", "check", "."],
        )
        self.assertEqual(run.call_args_list[0].kwargs["timeout"], 30.0)
        self.assertFalse(run.call_args_list[0].kwargs["shell"])
        self.assertTrue(after_lint.has_unverified_changes)
        self.assertEqual(after_lint.pending_validations, ("tests",))
        self.assertTrue(tests["passed"])
        self.assertFalse(registry.state.snapshot().has_unverified_changes)
        self.assertEqual(registry.state.snapshot().validation_runs, 2)

    def test_unknown_validation_name_never_starts_a_process(self) -> None:
        with (
            patch("safe_patch_agent.workspace.subprocess.run") as run,
            self.assertRaisesRegex(WorkspaceError, "可用任务：tests"),
        ):
            self.workspace.run_validation("shell")

        run.assert_not_called()

    def test_protected_control_config_is_hidden_from_all_agent_file_tools(self) -> None:
        config_path = self.root / "safe-patch-agent.toml"
        config_path.write_text("control = true\n", encoding="utf-8")
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            deletion_approval=lambda _preview: True,
            protected_paths=(config_path,),
        )

        listed = workspace.list_files()
        listed_paths = {entry["path"] for entry in listed["entries"]}
        self.assertNotIn("safe-patch-agent.toml", listed_paths)
        self.assertEqual(workspace.search_code("control")["matches"], [])
        with self.assertRaisesRegex(WorkspaceError, "控制面配置"):
            workspace.read_file("safe-patch-agent.toml")
        with self.assertRaisesRegex(WorkspaceError, "控制面配置"):
            workspace.create_file("safe-patch-agent.toml", "replacement = true\n")
        with self.assertRaisesRegex(WorkspaceError, "控制面配置"):
            workspace.replace_text("safe-patch-agent.toml", "true", "false")
        with self.assertRaisesRegex(WorkspaceError, "控制面配置"):
            workspace.delete_file("safe-patch-agent.toml")
        with self.assertRaisesRegex(WorkspaceError, "控制面配置"):
            workspace.apply_change_set(
                [
                    {
                        "kind": "create",
                        "path": "safe-patch-agent.toml",
                        "content": "replacement = true\n",
                    }
                ]
            )

    def test_run_tests_uses_default_named_task_and_sanitized_environment(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "LLM_API_KEY": "secret",
                    "GITHUB_TOKEN": "secret",
                    "SAFE_TEST_VALUE": "visible",
                    "PYTEST_ADDOPTS": "--dangerous-option",
                },
                clear=False,
            ),
            patch("safe_patch_agent.workspace.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="2 passed\n",
            )

            result = self.workspace.run_tests()

        command = run.call_args.args[0]
        options = run.call_args.kwargs
        self.assertEqual(command, [sys.executable, "-m", "pytest", "-q"])
        self.assertEqual(options["cwd"], self.root.resolve())
        self.assertFalse(options["shell"])
        self.assertEqual(options["timeout"], SafeWorkspace._TEST_TIMEOUT_SECONDS)
        self.assertNotIn("LLM_API_KEY", options["env"])
        self.assertNotIn("GITHUB_TOKEN", options["env"])
        self.assertNotIn("PYTEST_ADDOPTS", options["env"])
        self.assertEqual(options["env"]["SAFE_TEST_VALUE"], "visible")
        self.assertEqual(options["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")
        self.assertTrue(result["passed"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["command"], "python -m pytest -q")

    def test_run_tests_reports_failure_and_updates_registry_state(self) -> None:
        registry = build_agent_registry(self.workspace)
        registry.state.mark_file_read("src/app.py")
        registry.state.mark_file_modified("src/app.py")
        with patch("safe_patch_agent.workspace.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="1 failed\n",
            )
            result = json.loads(
                registry.execute(ToolCall(id="tests", name="run_tests", arguments={}))
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["output"], "1 failed\n")
        snapshot = registry.state.snapshot()
        self.assertEqual(snapshot.test_runs, 1)
        self.assertFalse(snapshot.last_test_passed)
        self.assertFalse(snapshot.has_unverified_changes)

    def test_run_tests_updates_change_journal_status(self) -> None:
        target = self.root / "replace.py"
        target.write_text("old\n", encoding="utf-8")
        journal = ChangeJournal()
        workspace = SafeWorkspace(
            self.root,
            replacement_approval=lambda _preview: True,
            change_journal=journal,
        )
        workspace.replace_text("replace.py", "old", "new")
        with patch("safe_patch_agent.workspace.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="1 passed\n",
            )

            workspace.run_tests()

        self.assertTrue(journal.summaries()[0].test_passed)

    def test_run_tests_reports_timeout_with_bounded_output(self) -> None:
        long_output = b"start\n" + b"x" * 500
        with (
            patch.object(SafeWorkspace, "_MAX_TEST_OUTPUT_CHARS", 100),
            patch("safe_patch_agent.workspace.subprocess.run") as run,
        ):
            run.side_effect = subprocess.TimeoutExpired(
                cmd=[sys.executable, "-m", "pytest", "-q"],
                timeout=SafeWorkspace._TEST_TIMEOUT_SECONDS,
                output=long_output,
            )

            result = self.workspace.run_tests()

        self.assertFalse(result["passed"])
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])
        self.assertTrue(result["output_truncated"])
        self.assertLessEqual(len(result["output"]), 100)
        self.assertIn("测试输出已截断", result["output"])

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

    def test_search_code_does_not_follow_file_symlink_outside_workspace(self) -> None:
        with TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "outside.txt"
            outside.write_text("outside-needle\n", encoding="utf-8")
            link = self.root / "linked.txt"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("当前 Windows 环境不允许创建符号链接")

            result = self.workspace.search_code("outside-needle")

        self.assertEqual(result["matches"], [])

    def test_search_code_does_not_follow_directory_symlink_outside_workspace(self) -> None:
        with TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory)
            (outside / "outside.txt").write_text("outside-needle\n", encoding="utf-8")
            link = self.root / "linked-directory"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("当前 Windows 环境不允许创建符号链接")

            result = self.workspace.search_code("outside-needle")

        self.assertEqual(result["matches"], [])


if __name__ == "__main__":
    unittest.main()
