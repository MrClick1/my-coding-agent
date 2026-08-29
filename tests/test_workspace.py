import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from safe_patch_agent.messages import ToolCall
from safe_patch_agent.workspace import (
    ReplacementPreview,
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

    def test_workspace_rejects_non_callable_approval_handler(self) -> None:
        with self.assertRaisesRegex(WorkspaceError, "必须是可调用对象"):
            SafeWorkspace(self.root, replacement_approval=True)  # type: ignore[arg-type]

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

    def test_agent_registry_exposes_replace_text_schema(self) -> None:
        registry = build_agent_registry(self.workspace)
        tool_names = {schema["function"]["name"] for schema in registry.schemas()}
        replace_schema = next(
            schema
            for schema in registry.schemas()
            if schema["function"]["name"] == "replace_text"
        )["function"]["parameters"]

        self.assertEqual(
            tool_names,
            {"list_files", "read_file", "search_code", "replace_text", "run_tests"},
        )
        self.assertEqual(
            replace_schema["required"],
            ["path", "old_text", "new_text"],
        )
        self.assertEqual(
            replace_schema["properties"]["expected_replacements"]["default"],
            1,
        )

    def test_run_tests_uses_fixed_command_and_sanitized_environment(self) -> None:
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
