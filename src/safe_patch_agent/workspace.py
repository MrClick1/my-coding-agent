"""工作区边界检查，以及目录、搜索、读取、精确替换和固定测试工具。"""

from __future__ import annotations

import difflib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safe_patch_agent.changes import ChangeJournal, ChangeJournalError, ChangeKind
from safe_patch_agent.config import ConfigurationError, ValidationConfig
from safe_patch_agent.tooling import ToolDefinition, ToolFileAccess, ToolRegistry


class WorkspaceError(ValueError):
    """请求的路径或操作不安全或无效。"""


@dataclass(frozen=True)
class ReplacementPreview:
    """等待用户确认的一次精确替换预览。"""

    path: str
    diff: str
    replacements: int
    original_bytes: int
    updated_bytes: int


ReplacementApproval = Callable[[ReplacementPreview], bool]


@dataclass(frozen=True)
class CreationPreview:
    """等待用户确认的一次新文件创建预览。"""

    path: str
    diff: str
    updated_bytes: int


CreationApproval = Callable[[CreationPreview], bool]


@dataclass(frozen=True)
class DeletionPreview:
    """等待用户确认的一次现有文件删除预览。"""

    path: str
    diff: str
    original_bytes: int


DeletionApproval = Callable[[DeletionPreview], bool]


@dataclass(frozen=True)
class BatchChangePreview:
    """等待用户统一确认的一组文件变更预览。"""

    paths: tuple[str, ...]
    diff: str
    creations: int
    replacements: int
    deletions: int
    original_bytes: int
    updated_bytes: int


BatchChangeApproval = Callable[[BatchChangePreview], bool]


@dataclass(frozen=True)
class _PreparedChange:
    """已经完成输入和工作区状态校验、等待提交的一项文件变更。"""

    kind: ChangeKind
    path: str
    target: Path
    before_text: str | None
    after_text: str | None
    diff: str
    replacements: int
    original_bytes: int
    updated_bytes: int


@dataclass(frozen=True)
class RollbackPreview:
    """等待用户确认的一次会话修改回滚预览。"""

    change_ids: tuple[int, ...]
    paths: tuple[str, ...]
    diff: str
    original_bytes: int
    updated_bytes: int
    deleted_paths: tuple[str, ...] = ()
    created_paths: tuple[str, ...] = ()


RollbackApproval = Callable[[RollbackPreview], bool]


class SafeWorkspace:
    """开放指定项目目录，同时阻止路径访问到工作区之外。"""

    _MAX_FILE_BYTES = 1_000_000
    _MAX_OUTPUT_CHARS = 40_000
    _MAX_SEARCH_QUERY_CHARS = 200
    _MAX_SEARCH_RESULTS = 100
    _MAX_SEARCH_FILES = 2_000
    _MAX_SEARCH_DIRECTORIES = 2_000
    _MAX_SEARCH_ENTRIES = 20_000
    _MAX_SEARCH_BYTES = 20_000_000
    _MAX_SEARCH_LINE_CHARS = 500
    _MAX_SEARCH_OUTPUT_CHARS = 35_000
    _MAX_REPLACEMENTS = 100
    _MAX_BATCH_CHANGES = 20
    _MAX_PATCH_PREVIEW_CHARS = 30_000
    _TEST_TIMEOUT_SECONDS = 120
    _MAX_TEST_OUTPUT_CHARS = 40_000
    _SENSITIVE_ENV_MARKERS = (
        "ACCESS_KEY",
        "API_KEY",
        "AUTH",
        "CREDENTIAL",
        "PASSWD",
        "PASSWORD",
        "PRIVATE_KEY",
        "SECRET",
        "TOKEN",
    )

    _IGNORED_DIRECTORY_NAMES = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    _SENSITIVE_DIRECTORY_NAMES = {
        ".aws",
        ".azure",
        ".git",
        ".gnupg",
        ".ssh",
    }
    _SENSITIVE_FILE_NAMES = {
        ".env",
        ".envrc",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "credentials.toml",
        "credentials.yaml",
        "credentials.yml",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.py",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
    }
    _SENSITIVE_FILE_SUFFIXES = {".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"}

    def __init__(
        self,
        root: Path,
        *,
        replacement_approval: ReplacementApproval | None = None,
        creation_approval: CreationApproval | None = None,
        deletion_approval: DeletionApproval | None = None,
        batch_change_approval: BatchChangeApproval | None = None,
        validation_config: ValidationConfig | None = None,
        protected_paths: Sequence[Path] = (),
        change_journal: ChangeJournal | None = None,
        rollback_approval: RollbackApproval | None = None,
    ) -> None:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.exists():
            raise WorkspaceError(f"工作区不存在：{resolved_root}")
        if not resolved_root.is_dir():
            raise WorkspaceError(f"工作区路径不是目录：{resolved_root}")
        if replacement_approval is not None and not callable(replacement_approval):
            raise WorkspaceError("replacement_approval 必须是可调用对象")
        if creation_approval is not None and not callable(creation_approval):
            raise WorkspaceError("creation_approval 必须是可调用对象")
        if deletion_approval is not None and not callable(deletion_approval):
            raise WorkspaceError("deletion_approval 必须是可调用对象")
        if batch_change_approval is not None and not callable(batch_change_approval):
            raise WorkspaceError("batch_change_approval 必须是可调用对象")
        if rollback_approval is not None and not callable(rollback_approval):
            raise WorkspaceError("rollback_approval 必须是可调用对象")
        if validation_config is not None and not isinstance(
            validation_config, ValidationConfig
        ):
            raise WorkspaceError("validation_config 必须是 ValidationConfig")
        if not isinstance(protected_paths, Sequence) or isinstance(
            protected_paths, (str, bytes)
        ):
            raise WorkspaceError("protected_paths 必须是路径序列")
        self.root = resolved_root
        self.replacement_approval = replacement_approval
        self.creation_approval = creation_approval
        self.deletion_approval = deletion_approval
        self.batch_change_approval = batch_change_approval
        self.validation_config = validation_config or ValidationConfig.default()
        resolved_protected_paths: set[Path] = set()
        for protected_path in protected_paths:
            if not isinstance(protected_path, Path):
                raise WorkspaceError("protected_paths 必须只包含 Path")
            candidate = (
                protected_path
                if protected_path.is_absolute()
                else self.root / protected_path
            ).expanduser().resolve(strict=False)
            try:
                candidate.relative_to(self.root)
            except ValueError:
                continue
            resolved_protected_paths.add(candidate)
        self._protected_paths = frozenset(resolved_protected_paths)
        self.change_journal = change_journal
        self.rollback_approval = rollback_approval

    def resolve(self, relative_path: str, *, must_exist: bool = True) -> Path:
        """解析相对路径，并确保解析结果仍位于工作区内。"""

        if not isinstance(relative_path, str) or not relative_path.strip():
            raise WorkspaceError("路径必须是非空字符串")

        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise WorkspaceError("不允许使用绝对路径")

        candidate = (self.root / supplied).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("路径超出了已配置的工作区") from exc

        self._assert_read_allowed(candidate)

        if must_exist and not candidate.exists():
            raise WorkspaceError(f"路径不存在：{relative_path}")
        return candidate

    def display_path(self, path: Path) -> str:
        relative = path.relative_to(self.root)
        text = relative.as_posix()
        return text if text else "."

    def canonical_file_path(self, path: str) -> str:
        """返回通过安全检查的规范化文件相对路径。"""

        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"路径不是文件：{path}")
        return self.display_path(target)

    def canonical_new_file_path(self, path: str) -> str:
        """返回父目录存在、目标尚不存在的规范化相对路径。"""

        target = self.resolve(path, must_exist=False)
        if os.path.lexists(target):
            raise WorkspaceError(f"创建目标已经存在：{path}")
        if not target.parent.exists() or not target.parent.is_dir():
            raise WorkspaceError(f"创建目标的父目录不存在：{path}")
        return self.display_path(target)

    def list_files(self, path: str = ".", max_results: int = 200) -> dict[str, Any]:
        """列出安全相对路径下的项目文件和目录。"""

        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise WorkspaceError("max_results 必须是整数")
        if not 1 <= max_results <= 500:
            raise WorkspaceError("max_results 必须在 1 到 500 之间")

        target = self.resolve(path)
        if not target.is_dir():
            raise WorkspaceError(f"路径不是目录：{path}")

        entries: list[dict[str, str]] = []
        truncated = False
        for current_root, directory_names, file_names in os.walk(target, followlinks=False):
            current = Path(current_root)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name.casefold() not in self._IGNORED_DIRECTORY_NAMES
                and name.casefold() not in self._SENSITIVE_DIRECTORY_NAMES
                and self._is_visible_path(current / name)
            )

            for name in directory_names:
                item = current / name
                if not self._is_visible_path(item):
                    continue
                entries.append({"path": self.display_path(item), "type": "directory"})
                if len(entries) >= max_results:
                    truncated = True
                    break
            if truncated:
                break

            for name in sorted(file_names):
                item = current / name
                if not self._is_visible_path(item):
                    continue
                entries.append({"path": self.display_path(item), "type": "file"})
                if len(entries) >= max_results:
                    truncated = True
                    break
            if truncated:
                break

        return {
            "ok": True,
            "path": self.display_path(target),
            "entries": entries,
            "truncated": truncated,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """读取最多 400 行 UTF-8 文本，并附带稳定的行号。"""

        if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
            raise WorkspaceError("start_line 必须是正整数")
        if end_line is not None and (
            not isinstance(end_line, int) or isinstance(end_line, bool) or end_line < start_line
        ):
            raise WorkspaceError("end_line 必须是大于或等于 start_line 的整数")

        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"路径不是文件：{path}")
        text, _ = self._read_utf8_text(target, path)

        lines = text.splitlines()
        if not lines:
            return {
                "ok": True,
                "path": self.display_path(target),
                "content": "",
                "start_line": 0,
                "end_line": 0,
                "total_lines": 0,
                "truncated": False,
            }
        if start_line > max(1, len(lines)):
            raise WorkspaceError(
                f"start_line {start_line} 超出了文件的 {len(lines)} 行范围"
            )
        requested_end = end_line if end_line is not None else len(lines)
        actual_end = min(requested_end, start_line + 399, len(lines))
        selected = lines[start_line - 1 : actual_end]
        numbered_content = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )
        output_was_truncated = len(numbered_content) > self._MAX_OUTPUT_CHARS
        if output_was_truncated:
            numbered_content = (
                numbered_content[: self._MAX_OUTPUT_CHARS]
                + "\n... [工具输出已截断]"
            )

        return {
            "ok": True,
            "path": self.display_path(target),
            "content": numbered_content,
            "start_line": start_line,
            "end_line": actual_end,
            "total_lines": len(lines),
            "truncated": (
                output_was_truncated
                or actual_end < requested_end
                or actual_end < len(lines)
            ),
        }

    def search_code(
        self,
        query: str,
        path: str = ".",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """在安全工作区的 UTF-8 文本中逐行搜索字面量。"""

        if not isinstance(query, str) or not query.strip():
            raise WorkspaceError("query 必须是非空字符串")
        if len(query) > self._MAX_SEARCH_QUERY_CHARS:
            raise WorkspaceError(
                f"query 不能超过 {self._MAX_SEARCH_QUERY_CHARS} 个字符"
            )
        line_separators = "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
        if "\x00" in query or any(character in query for character in line_separators):
            raise WorkspaceError("query 不能包含换行符或 NUL 字符")
        if not isinstance(case_sensitive, bool):
            raise WorkspaceError("case_sensitive 必须是布尔值")
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise WorkspaceError("max_results 必须是整数")
        if not 1 <= max_results <= self._MAX_SEARCH_RESULTS:
            raise WorkspaceError(
                f"max_results 必须在 1 到 {self._MAX_SEARCH_RESULTS} 之间"
            )

        target = self.resolve(path)
        if not target.is_file() and not target.is_dir():
            raise WorkspaceError(f"路径不是普通文件或目录：{path}")

        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)
        target_display_path = self.display_path(target)
        candidates, skipped_directories, traversal_truncated = (
            self._collect_visible_search_files(target)
        )
        matches: list[dict[str, Any]] = []
        matched_paths: set[str] = set()
        scanned_files = 0
        skipped_files = 0
        scanned_bytes = 0
        output_chars = len(
            json.dumps(
                {
                    "ok": True,
                    "query": query,
                    "path": target_display_path,
                    "case_sensitive": case_sensitive,
                    "matches": [],
                    "scanned_files": self._MAX_SEARCH_FILES,
                    "scanned_bytes": self._MAX_SEARCH_BYTES,
                    "skipped_files": self._MAX_SEARCH_FILES,
                    "skipped_directories": self._MAX_SEARCH_DIRECTORIES,
                    "matched_files": self._MAX_SEARCH_FILES,
                    "truncated": True,
                    "complete": False,
                },
                ensure_ascii=False,
            )
        )
        truncated = traversal_truncated
        stop_search = False

        for candidate in candidates:
            remaining_bytes = self._MAX_SEARCH_BYTES - scanned_bytes
            if remaining_bytes <= 0:
                truncated = True
                break

            display_path = self.display_path(candidate)
            text, byte_count, budget_exhausted = self._read_search_text(
                candidate,
                remaining_bytes,
            )
            scanned_bytes += byte_count
            if budget_exhausted:
                truncated = True
                break
            if text is None:
                skipped_files += 1
                continue

            scanned_files += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                match = pattern.search(line)
                if match is None:
                    continue
                if len(matches) >= max_results:
                    truncated = True
                    stop_search = True
                    break

                content, content_truncated = self._search_excerpt(
                    line,
                    match.start(),
                    match.end(),
                )
                result_match = {
                    "path": display_path,
                    "line_number": line_number,
                    "column": match.start() + 1,
                    "content": content,
                    "content_truncated": content_truncated,
                }
                match_output_chars = len(
                    json.dumps(result_match, ensure_ascii=False)
                ) + 2
                if output_chars + match_output_chars > self._MAX_SEARCH_OUTPUT_CHARS:
                    truncated = True
                    stop_search = True
                    break

                matches.append(result_match)
                matched_paths.add(display_path)
                output_chars += match_output_chars

            if stop_search:
                break

        return {
            "ok": True,
            "query": query,
            "path": target_display_path,
            "case_sensitive": case_sensitive,
            "matches": matches,
            "scanned_files": scanned_files,
            "scanned_bytes": scanned_bytes,
            "skipped_files": skipped_files,
            "skipped_directories": skipped_directories,
            "matched_files": len(matched_paths),
            "truncated": truncated,
            "complete": not truncated and skipped_files == 0 and skipped_directories == 0,
        }

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> dict[str, Any]:
        """生成完整差异并获得用户确认后，原子替换 UTF-8 文件。"""

        if not isinstance(old_text, str) or not old_text:
            raise WorkspaceError("old_text 必须是非空字符串")
        if not isinstance(new_text, str):
            raise WorkspaceError("new_text 必须是字符串")
        if "\x00" in old_text or "\x00" in new_text:
            raise WorkspaceError("替换文本不能包含 NUL 字符")
        if old_text == new_text:
            raise WorkspaceError("old_text 和 new_text 不能完全相同")
        if not isinstance(expected_replacements, int) or isinstance(
            expected_replacements,
            bool,
        ):
            raise WorkspaceError("expected_replacements 必须是整数")
        if not 1 <= expected_replacements <= self._MAX_REPLACEMENTS:
            raise WorkspaceError(
                f"expected_replacements 必须在 1 到 {self._MAX_REPLACEMENTS} 之间"
            )

        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"路径不是文件：{path}")
        display_path = self.display_path(target)
        original_text, original_bytes = self._read_utf8_text(target, display_path)
        if "\x00" in original_text:
            raise WorkspaceError(f"文件包含 NUL 字符，拒绝按文本修改：{display_path}")

        actual_replacements = original_text.count(old_text)
        if actual_replacements != expected_replacements:
            raise WorkspaceError(
                "替换次数不符合预期："
                f"期望 {expected_replacements} 次，实际 {actual_replacements} 次；文件未修改"
            )

        updated_text = original_text.replace(old_text, new_text)
        try:
            updated_bytes = updated_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkspaceError("new_text 不能编码为有效的 UTF-8") from exc
        if len(updated_bytes) > self._MAX_FILE_BYTES:
            raise WorkspaceError(
                f"替换后的文件超过 {self._MAX_FILE_BYTES} 字节上限；文件未修改"
            )

        diff = self._replacement_diff(display_path, original_text, updated_text)
        if len(diff) > self._MAX_PATCH_PREVIEW_CHARS:
            raise WorkspaceError(
                "修改预览超过 "
                f"{self._MAX_PATCH_PREVIEW_CHARS} 字符上限，无法完整展示；文件未修改"
            )
        if self.change_journal is not None:
            try:
                self.change_journal.ensure_can_record(original_text, updated_text)
            except ChangeJournalError as exc:
                raise WorkspaceError(str(exc)) from exc
        preview = ReplacementPreview(
            path=display_path,
            diff=diff,
            replacements=actual_replacements,
            original_bytes=original_bytes,
            updated_bytes=len(updated_bytes),
        )
        if self.replacement_approval is None:
            raise WorkspaceError("未配置用户确认，拒绝修改文件")
        try:
            approved = self.replacement_approval(preview)
        except Exception as exc:
            raise WorkspaceError("无法获得用户确认；文件未修改") from exc
        if approved is not True:
            raise WorkspaceError("用户拒绝了修改；文件未修改")

        current_text, current_bytes = self._read_utf8_text(target, display_path)
        if current_bytes != original_bytes or current_text != original_text:
            raise WorkspaceError("文件在确认期间发生变化；为避免覆盖新内容，已取消修改")

        self._atomic_write(target, updated_bytes)
        change_id: int | None = None
        if self.change_journal is not None:
            change_id = self.change_journal.record(
                path=display_path,
                before_text=original_text,
                after_text=updated_text,
                diff=diff,
                replacements=actual_replacements,
            ).change_id
        return {
            "ok": True,
            "path": display_path,
            "approved": True,
            "change_id": change_id,
            "diff": diff,
            "replacements": actual_replacements,
            "original_bytes": original_bytes,
            "updated_bytes": len(updated_bytes),
        }

    def create_file(self, path: str, content: str) -> dict[str, Any]:
        """完整预览并确认后，以排他方式创建新的 UTF-8 文件。"""

        if not isinstance(content, str) or not content:
            raise WorkspaceError("content 必须是非空字符串")
        if "\x00" in content:
            raise WorkspaceError("新文件内容不能包含 NUL 字符")
        display_path = self.canonical_new_file_path(path)
        target = self.resolve(display_path, must_exist=False)
        try:
            content_bytes = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkspaceError("content 不能编码为有效的 UTF-8") from exc
        if len(content_bytes) > self._MAX_FILE_BYTES:
            raise WorkspaceError(
                f"新文件超过 {self._MAX_FILE_BYTES} 字节上限；文件未创建"
            )

        diff = self._creation_diff(display_path, content)
        if len(diff) > self._MAX_PATCH_PREVIEW_CHARS:
            raise WorkspaceError(
                "创建预览超过 "
                f"{self._MAX_PATCH_PREVIEW_CHARS} 字符上限，无法完整展示；文件未创建"
            )
        if self.change_journal is not None:
            try:
                self.change_journal.ensure_can_record(None, content)
            except ChangeJournalError as exc:
                raise WorkspaceError(str(exc).replace("文件未修改", "文件未创建")) from exc
        preview = CreationPreview(
            path=display_path,
            diff=diff,
            updated_bytes=len(content_bytes),
        )
        if self.creation_approval is None:
            raise WorkspaceError("未配置用户确认，拒绝创建文件")
        try:
            approved = self.creation_approval(preview)
        except Exception as exc:
            raise WorkspaceError("无法获得用户确认；文件未创建") from exc
        if approved is not True:
            raise WorkspaceError("用户拒绝了创建；文件未创建")
        if os.path.lexists(target):
            raise WorkspaceError(
                "目标在确认期间被其他操作创建；为避免覆盖，已取消创建"
            )

        self._exclusive_write(target, content_bytes)
        change_id: int | None = None
        if self.change_journal is not None:
            change_id = self.change_journal.record_creation(
                path=display_path,
                after_text=content,
                diff=diff,
            ).change_id
        return {
            "ok": True,
            "path": display_path,
            "approved": True,
            "created": True,
            "change_id": change_id,
            "diff": diff,
            "updated_bytes": len(content_bytes),
        }

    def delete_file(self, path: str) -> dict[str, Any]:
        """完整预览并确认后，删除内容未发生变化的 UTF-8 文件。"""

        display_path = self.canonical_file_path(path)
        target = self.resolve(display_path)
        original_text, original_bytes = self._read_utf8_text(target, display_path)
        if "\x00" in original_text:
            raise WorkspaceError(f"文件包含 NUL 字符，拒绝按文本删除：{display_path}")

        diff = self._deletion_diff(display_path, original_text)
        if len(diff) > self._MAX_PATCH_PREVIEW_CHARS:
            raise WorkspaceError(
                "删除预览超过 "
                f"{self._MAX_PATCH_PREVIEW_CHARS} 字符上限，无法完整展示；文件未删除"
            )
        if self.change_journal is not None:
            try:
                self.change_journal.ensure_can_record(original_text, None)
            except ChangeJournalError as exc:
                raise WorkspaceError(str(exc).replace("文件未修改", "文件未删除")) from exc
        preview = DeletionPreview(
            path=display_path,
            diff=diff,
            original_bytes=original_bytes,
        )
        if self.deletion_approval is None:
            raise WorkspaceError("未配置用户确认，拒绝删除文件")
        try:
            approved = self.deletion_approval(preview)
        except Exception as exc:
            raise WorkspaceError("无法获得用户确认；文件未删除") from exc
        if approved is not True:
            raise WorkspaceError("用户拒绝了删除；文件未删除")

        current_text, current_bytes = self._read_utf8_text(target, display_path)
        if current_bytes != original_bytes or current_text != original_text:
            raise WorkspaceError(
                "文件在确认期间发生变化；为避免删除新内容，已取消删除"
            )

        self._checked_delete(target, original_text.encode("utf-8"))
        change_id: int | None = None
        if self.change_journal is not None:
            change_id = self.change_journal.record_deletion(
                path=display_path,
                before_text=original_text,
                diff=diff,
            ).change_id
        return {
            "ok": True,
            "path": display_path,
            "approved": True,
            "deleted": True,
            "change_id": change_id,
            "diff": diff,
            "original_bytes": original_bytes,
        }

    def change_set_file_accesses(
        self,
        arguments: Mapping[str, Any],
    ) -> tuple[tuple[ToolFileAccess, str], ...]:
        """把批量变更参数解析为 AgentState 可检查的规范化路径访问。"""

        normalized = self._normalize_change_set_operations(arguments.get("operations"))
        access_by_kind = {
            ChangeKind.CREATE: ToolFileAccess.CREATE,
            ChangeKind.REPLACE: ToolFileAccess.WRITE,
            ChangeKind.DELETE: ToolFileAccess.DELETE,
        }
        return tuple(
            (access_by_kind[kind], path)
            for kind, path, _operation in normalized
        )

    def apply_change_set(
        self,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """统一预览、确认并以失败恢复方式应用一组独立文件变更。"""

        normalized = self._normalize_change_set_operations(operations)
        prepared = tuple(
            self._prepare_change(kind, path, operation)
            for kind, path, operation in normalized
        )
        diff = "".join(change.diff for change in prepared)
        if len(diff) > self._MAX_PATCH_PREVIEW_CHARS:
            raise WorkspaceError(
                "批量变更预览超过 "
                f"{self._MAX_PATCH_PREVIEW_CHARS} 字符上限，无法完整展示；"
                "工作区未修改"
            )
        if self.change_journal is not None:
            try:
                self.change_journal.ensure_can_record_many(
                    tuple(
                        (change.before_text, change.after_text)
                        for change in prepared
                    )
                )
            except ChangeJournalError as exc:
                raise WorkspaceError(str(exc)) from exc

        preview = BatchChangePreview(
            paths=tuple(change.path for change in prepared),
            diff=diff,
            creations=sum(change.kind is ChangeKind.CREATE for change in prepared),
            replacements=sum(
                change.kind is ChangeKind.REPLACE for change in prepared
            ),
            deletions=sum(change.kind is ChangeKind.DELETE for change in prepared),
            original_bytes=sum(change.original_bytes for change in prepared),
            updated_bytes=sum(change.updated_bytes for change in prepared),
        )
        if self.batch_change_approval is None:
            raise WorkspaceError("未配置用户确认，拒绝应用批量变更")
        try:
            approved = self.batch_change_approval(preview)
        except Exception as exc:
            raise WorkspaceError("无法获得用户确认；工作区未修改") from exc
        if approved is not True:
            raise WorkspaceError("用户拒绝了批量变更；工作区未修改")

        self._recheck_prepared_changes(prepared)
        applied: list[_PreparedChange] = []
        try:
            for change in prepared:
                self._apply_prepared_change(change)
                applied.append(change)
        except WorkspaceError as exc:
            restore_failed = False
            for change in reversed(applied):
                try:
                    self._restore_prepared_change(change)
                except WorkspaceError:
                    restore_failed = True
            if restore_failed:
                raise WorkspaceError(
                    "批量变更应用失败，且无法完整恢复本批次已经变更的文件；"
                    "请立即检查工作区"
                ) from exc
            raise WorkspaceError(
                "批量变更应用失败，已恢复本批次已经变更的文件"
            ) from exc

        change_ids: list[int] = []
        if self.change_journal is not None:
            for change in prepared:
                record = self.change_journal.record(
                    path=change.path,
                    before_text=change.before_text,
                    after_text=change.after_text,
                    diff=change.diff,
                    replacements=change.replacements,
                    kind=change.kind,
                )
                change_ids.append(record.change_id)
        return {
            "ok": True,
            "approved": True,
            "paths": tuple(change.path for change in prepared),
            "change_ids": tuple(change_ids),
            "creations": preview.creations,
            "replacements": preview.replacements,
            "deletions": preview.deletions,
            "diff": diff,
        }

    def _normalize_change_set_operations(
        self,
        operations: object,
    ) -> tuple[tuple[ChangeKind, str, Mapping[str, Any]], ...]:
        """校验批量操作结构、规范化路径，并拒绝同一路径重复操作。"""

        if not isinstance(operations, list):
            raise WorkspaceError("operations 必须是数组")
        if not 1 <= len(operations) <= self._MAX_BATCH_CHANGES:
            raise WorkspaceError(
                f"operations 必须包含 1 到 {self._MAX_BATCH_CHANGES} 项变更"
            )
        required_keys = {
            ChangeKind.CREATE: {"kind", "path", "content"},
            ChangeKind.REPLACE: {"kind", "path", "old_text", "new_text"},
            ChangeKind.DELETE: {"kind", "path"},
        }
        allowed_keys = {
            **required_keys,
            ChangeKind.REPLACE: required_keys[ChangeKind.REPLACE]
            | {"expected_replacements"},
        }
        normalized: list[tuple[ChangeKind, str, Mapping[str, Any]]] = []
        seen_paths: set[str] = set()
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                raise WorkspaceError(f"operations[{index}] 必须是对象")
            raw_kind = operation.get("kind")
            try:
                kind = ChangeKind(raw_kind)
            except (TypeError, ValueError) as exc:
                raise WorkspaceError(
                    f"operations[{index}].kind 必须是 create、replace 或 delete"
                ) from exc
            missing_keys = required_keys[kind] - operation.keys()
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise WorkspaceError(f"operations[{index}] 缺少字段：{missing}")
            unexpected_keys = operation.keys() - allowed_keys[kind]
            if unexpected_keys:
                unexpected = ", ".join(sorted(str(key) for key in unexpected_keys))
                raise WorkspaceError(f"operations[{index}] 包含多余字段：{unexpected}")
            path = operation.get("path")
            if not isinstance(path, str):
                raise WorkspaceError(f"operations[{index}].path 必须是字符串")
            display_path = (
                self.canonical_new_file_path(path)
                if kind is ChangeKind.CREATE
                else self.canonical_file_path(path)
            )
            path_key = os.path.normcase(display_path)
            if path_key in seen_paths:
                raise WorkspaceError(f"批量变更不能重复操作同一路径：{display_path}")
            seen_paths.add(path_key)
            normalized.append((kind, display_path, operation))
        return tuple(normalized)

    def _prepare_change(
        self,
        kind: ChangeKind,
        path: str,
        operation: Mapping[str, Any],
    ) -> _PreparedChange:
        """构建一项已经完整校验的批量文件变更。"""

        target = self.resolve(path, must_exist=kind is not ChangeKind.CREATE)
        if kind is ChangeKind.CREATE:
            content = operation["content"]
            if not isinstance(content, str) or not content:
                raise WorkspaceError(f"新文件 {path} 的 content 必须是非空字符串")
            if "\x00" in content:
                raise WorkspaceError(f"新文件 {path} 的内容不能包含 NUL 字符")
            try:
                content_bytes = content.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise WorkspaceError(
                    f"新文件 {path} 的 content 不能编码为有效的 UTF-8"
                ) from exc
            if len(content_bytes) > self._MAX_FILE_BYTES:
                raise WorkspaceError(
                    f"新文件 {path} 超过 {self._MAX_FILE_BYTES} 字节上限"
                )
            return _PreparedChange(
                kind=kind,
                path=path,
                target=target,
                before_text=None,
                after_text=content,
                diff=self._creation_diff(path, content),
                replacements=0,
                original_bytes=0,
                updated_bytes=len(content_bytes),
            )

        before_text, original_bytes = self._read_utf8_text(target, path)
        if "\x00" in before_text:
            raise WorkspaceError(f"文件包含 NUL 字符，拒绝批量变更：{path}")
        if kind is ChangeKind.DELETE:
            return _PreparedChange(
                kind=kind,
                path=path,
                target=target,
                before_text=before_text,
                after_text=None,
                diff=self._deletion_diff(path, before_text),
                replacements=0,
                original_bytes=original_bytes,
                updated_bytes=0,
            )

        old_text = operation["old_text"]
        new_text = operation["new_text"]
        expected_replacements = operation.get("expected_replacements", 1)
        if not isinstance(old_text, str) or not old_text:
            raise WorkspaceError(f"文件 {path} 的 old_text 必须是非空字符串")
        if not isinstance(new_text, str):
            raise WorkspaceError(f"文件 {path} 的 new_text 必须是字符串")
        if "\x00" in old_text or "\x00" in new_text:
            raise WorkspaceError(f"文件 {path} 的替换文本不能包含 NUL 字符")
        if old_text == new_text:
            raise WorkspaceError(f"文件 {path} 的 old_text 和 new_text 不能相同")
        if not isinstance(expected_replacements, int) or isinstance(
            expected_replacements, bool
        ):
            raise WorkspaceError(
                f"文件 {path} 的 expected_replacements 必须是整数"
            )
        if not 1 <= expected_replacements <= self._MAX_REPLACEMENTS:
            raise WorkspaceError(
                f"文件 {path} 的 expected_replacements 必须在 1 到 "
                f"{self._MAX_REPLACEMENTS} 之间"
            )
        actual_replacements = before_text.count(old_text)
        if actual_replacements != expected_replacements:
            raise WorkspaceError(
                f"文件 {path} 的替换次数不符合预期：期望 "
                f"{expected_replacements} 次，实际 {actual_replacements} 次"
            )
        after_text = before_text.replace(old_text, new_text)
        try:
            after_bytes = after_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkspaceError(
                f"文件 {path} 的 new_text 不能编码为有效的 UTF-8"
            ) from exc
        if len(after_bytes) > self._MAX_FILE_BYTES:
            raise WorkspaceError(
                f"文件 {path} 替换后超过 {self._MAX_FILE_BYTES} 字节上限"
            )
        return _PreparedChange(
            kind=kind,
            path=path,
            target=target,
            before_text=before_text,
            after_text=after_text,
            diff=self._replacement_diff(path, before_text, after_text),
            replacements=actual_replacements,
            original_bytes=original_bytes,
            updated_bytes=len(after_bytes),
        )

    def _recheck_prepared_changes(
        self,
        changes: tuple[_PreparedChange, ...],
    ) -> None:
        """在批量批准后确认每个路径仍保持预览时的状态。"""

        for change in changes:
            if change.before_text is None:
                if os.path.lexists(change.target):
                    raise WorkspaceError(
                        f"批量变更确认期间目标被创建：{change.path}；工作区未修改"
                    )
                continue
            if not os.path.lexists(change.target) or not change.target.is_file():
                raise WorkspaceError(
                    f"批量变更确认期间文件被删除或替换：{change.path}；"
                    "工作区未修改"
                )
            current_text, current_bytes = self._read_utf8_text(
                change.target,
                change.path,
            )
            if (
                current_text != change.before_text
                or current_bytes != change.original_bytes
            ):
                raise WorkspaceError(
                    f"批量变更确认期间文件发生变化：{change.path}；"
                    "工作区未修改"
                )

    def _apply_prepared_change(self, change: _PreparedChange) -> None:
        """提交一项批量变更。"""

        if change.kind is ChangeKind.CREATE:
            assert change.after_text is not None
            self._exclusive_write(change.target, change.after_text.encode("utf-8"))
        elif change.kind is ChangeKind.REPLACE:
            assert change.after_text is not None
            self._atomic_write(change.target, change.after_text.encode("utf-8"))
        else:
            assert change.before_text is not None
            self._checked_delete(change.target, change.before_text.encode("utf-8"))

    def _restore_prepared_change(self, change: _PreparedChange) -> None:
        """把已经提交的一项批量变更恢复到批次开始前。"""

        if change.before_text is None:
            if os.path.lexists(change.target):
                assert change.after_text is not None
                self._checked_delete(
                    change.target,
                    change.after_text.encode("utf-8"),
                )
            return
        original_content = change.before_text.encode("utf-8")
        if change.after_text is None:
            if os.path.lexists(change.target):
                raise WorkspaceError(
                    f"批量恢复期间删除目标被外部重新创建：{change.path}"
                )
            self._exclusive_write(change.target, original_content)
            return
        if not os.path.lexists(change.target) or not change.target.is_file():
            raise WorkspaceError(
                f"批量恢复期间替换目标被外部删除或替换：{change.path}"
            )
        current_text, current_bytes = self._read_utf8_text(change.target, change.path)
        if (
            current_text != change.after_text
            or current_bytes != change.updated_bytes
        ):
            raise WorkspaceError(
                f"批量恢复期间替换目标被外部修改：{change.path}"
            )
        self._atomic_write(change.target, original_content)

    def rollback_changes(
        self,
        target: int | str | None = None,
    ) -> dict[str, Any]:
        """校验并回滚当前会话的一条或全部活动修改。"""

        if self.change_journal is None:
            raise WorkspaceError("当前工作区未启用会话修改日志")
        try:
            records = self.change_journal.records_for_rollback(target)
        except ChangeJournalError as exc:
            raise WorkspaceError(str(exc)) from exc

        current_texts: dict[str, str | None] = {}
        current_bytes: dict[str, int] = {}
        targets: dict[str, Path] = {}
        for record in records:
            if record.path in targets:
                continue
            resolved = self.resolve(record.path, must_exist=False)
            targets[record.path] = resolved
            if os.path.lexists(resolved):
                if not resolved.is_file():
                    raise WorkspaceError(f"回滚目标不是文件：{record.path}")
                text, byte_count = self._read_utf8_text(resolved, record.path)
                current_texts[record.path] = text
                current_bytes[record.path] = byte_count
            else:
                current_texts[record.path] = None
                current_bytes[record.path] = 0

        restored_texts: dict[str, str | None] = dict(current_texts)
        for record in records:
            if restored_texts[record.path] != record.after_text:
                raise WorkspaceError(
                    f"文件 {record.path} 已在修改后发生变化；"
                    "为避免覆盖新内容，已取消回滚"
                )
            restored_texts[record.path] = record.before_text

        diffs = []
        for path in sorted(restored_texts, key=str.casefold):
            current = current_texts[path]
            restored = restored_texts[path]
            if current == restored:
                continue
            if current is None:
                assert restored is not None
                diffs.append(self._creation_diff(path, restored))
            elif restored is None:
                diffs.append(self._deletion_diff(path, current))
            else:
                diffs.append(self._replacement_diff(path, current, restored))
        diff = "".join(diffs) or "（所选修改的净文件状态不变）\n"
        if len(diff) > self._MAX_PATCH_PREVIEW_CHARS:
            raise WorkspaceError(
                "回滚预览超过 "
                f"{self._MAX_PATCH_PREVIEW_CHARS} 字符上限，无法完整展示；文件未修改"
            )
        encoded_restored = {
            path: text.encode("utf-8")
            for path, text in restored_texts.items()
            if text is not None
        }
        deleted_paths = tuple(
            sorted(
                (
                    path
                    for path, text in restored_texts.items()
                    if current_texts[path] is not None and text is None
                ),
                key=str.casefold,
            )
        )
        created_paths = tuple(
            sorted(
                (
                    path
                    for path, text in restored_texts.items()
                    if current_texts[path] is None and text is not None
                ),
                key=str.casefold,
            )
        )
        preview = RollbackPreview(
            change_ids=tuple(record.change_id for record in records),
            paths=tuple(sorted(restored_texts, key=str.casefold)),
            diff=diff,
            original_bytes=sum(current_bytes.values()),
            updated_bytes=sum(len(content) for content in encoded_restored.values()),
            deleted_paths=deleted_paths,
            created_paths=created_paths,
        )
        if self.rollback_approval is None:
            raise WorkspaceError("未配置用户确认，拒绝回滚文件")
        try:
            approved = self.rollback_approval(preview)
        except Exception as exc:
            raise WorkspaceError("无法获得用户确认；文件未修改") from exc
        if approved is not True:
            raise WorkspaceError("用户拒绝了回滚；文件未修改")

        for path, resolved in targets.items():
            expected_text = current_texts[path]
            if expected_text is None:
                if os.path.lexists(resolved):
                    raise WorkspaceError(
                        f"文件 {path} 在确认期间被重新创建；"
                        "为避免覆盖新内容，已取消回滚"
                    )
                continue
            if not os.path.lexists(resolved) or not resolved.is_file():
                raise WorkspaceError(
                    f"文件 {path} 在确认期间被删除或替换；已取消回滚"
                )
            text, byte_count = self._read_utf8_text(resolved, path)
            if text != expected_text or byte_count != current_bytes[path]:
                raise WorkspaceError(
                    f"文件 {path} 在确认期间发生变化；"
                    "为避免覆盖新内容，已取消回滚"
                )

        written_paths: list[str] = []
        try:
            for path in sorted(targets, key=str.casefold):
                current = current_texts[path]
                restored = restored_texts[path]
                if current == restored:
                    continue
                if restored is None:
                    assert current is not None
                    self._checked_delete(targets[path], current.encode("utf-8"))
                elif current is None:
                    self._exclusive_write(targets[path], encoded_restored[path])
                else:
                    self._atomic_write(targets[path], encoded_restored[path])
                written_paths.append(path)
        except WorkspaceError as exc:
            restore_failed = False
            for path in reversed(written_paths):
                try:
                    original_text = current_texts[path]
                    if original_text is None:
                        restored_text = restored_texts[path]
                        assert restored_text is not None
                        if os.path.lexists(targets[path]):
                            self._checked_delete(
                                targets[path],
                                restored_text.encode("utf-8"),
                            )
                    elif os.path.lexists(targets[path]):
                        if not targets[path].is_file():
                            raise WorkspaceError(
                                f"无法恢复回滚前文件：{path}"
                            )
                        self._atomic_write(
                            targets[path],
                            original_text.encode("utf-8"),
                        )
                    else:
                        self._exclusive_write(
                            targets[path],
                            original_text.encode("utf-8"),
                        )
                except WorkspaceError:
                    restore_failed = True
            if restore_failed:
                raise WorkspaceError(
                    "回滚写入失败，且无法完整恢复已经写入的文件；"
                    "请立即检查工作区"
                ) from exc
            raise WorkspaceError("回滚写入失败，已恢复本次写入的文件") from exc

        self.change_journal.mark_rolled_back(records)
        return {
            "ok": True,
            "change_ids": tuple(record.change_id for record in records),
            "paths": tuple(sorted(restored_texts, key=str.casefold)),
            "deleted_paths": deleted_paths,
            "created_paths": created_paths,
            "diff": diff,
        }

    @staticmethod
    def _replacement_diff(path: str, original_text: str, updated_text: str) -> str:
        """创建适合终端完整展示的 unified diff。"""

        return SafeWorkspace._unified_diff(
            original_text,
            updated_text,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )

    @staticmethod
    def _creation_diff(path: str, content: str) -> str:
        """创建从不存在状态到完整新文件的 unified diff。"""

        return SafeWorkspace._unified_diff(
            "",
            content,
            fromfile="/dev/null",
            tofile=f"b/{path}",
        )

    @staticmethod
    def _deletion_diff(path: str, content: str) -> str:
        """创建从现有文件到不存在状态的 unified diff。"""

        diff = SafeWorkspace._unified_diff(
            content,
            "",
            fromfile=f"a/{path}",
            tofile="/dev/null",
        )
        return diff or f"--- a/{path}\n+++ /dev/null\n"

    @staticmethod
    def _unified_diff(
        original_text: str,
        updated_text: str,
        *,
        fromfile: str,
        tofile: str,
    ) -> str:
        """按指定文件头创建完整且可见的 UTF-8 unified diff。"""

        raw_lines = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            updated_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="\n",
        )
        visible_lines: list[str] = []
        for line in raw_lines:
            has_newline = line.endswith("\n")
            visible_lines.append(line.replace("\r\n", "␍\n").replace("\r", "␍"))
            if not has_newline:
                visible_lines.append("\n\\ 文件末尾无换行符\n")
        return "".join(visible_lines)

    def run_tests(self) -> dict[str, Any]:
        """兼容入口：运行具名的 tests 验证任务。"""

        return self.run_validation("tests")

    def run_validation(self, name: str) -> dict[str, Any]:
        """按启动时冻结的名称运行固定验证任务，不接受模型命令参数。"""

        if not isinstance(name, str) or not name:
            raise WorkspaceError("验证任务名称必须是非空字符串")
        try:
            task = self.validation_config.get(name)
        except ConfigurationError as exc:
            available = ", ".join(self.validation_config.task_names)
            raise WorkspaceError(
                f"未知验证任务：{name}；可用任务：{available}"
            ) from exc
        command = list(task.resolved_command)
        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                env=self._sanitized_test_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                check=False,
                timeout=task.timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            output, output_truncated = self._truncate_test_output(
                self._decode_test_output(exc.stdout)
            )
            result = {
                "ok": True,
                "name": task.name,
                "description": task.description,
                "required": task.required,
                "passed": False,
                "command": task.display_command,
                "exit_code": None,
                "timed_out": True,
                "timeout_seconds": task.timeout_seconds,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "output": output,
                "output_truncated": output_truncated,
            }
            if self.change_journal is not None:
                self.change_journal.record_validation_result(task.name, False)
            return result
        except OSError as exc:
            raise WorkspaceError(f"无法启动验证任务：{task.name}") from exc

        output, output_truncated = self._truncate_test_output(completed.stdout or "")
        passed = completed.returncode == 0
        result = {
            "ok": True,
            "name": task.name,
            "description": task.description,
            "required": task.required,
            "passed": passed,
            "command": task.display_command,
            "exit_code": completed.returncode,
            "timed_out": False,
            "timeout_seconds": task.timeout_seconds,
            "duration_seconds": round(time.monotonic() - started_at, 3),
            "output": output,
            "output_truncated": output_truncated,
        }
        if self.change_journal is not None:
            self.change_journal.record_validation_result(task.name, passed)
        return result

    def _read_utf8_text(self, target: Path, display_path: str) -> tuple[str, int]:
        """在固定字节上限内读取 UTF-8 文本。"""

        try:
            with target.open("rb") as file:
                raw_content = file.read(self._MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise WorkspaceError(f"无法读取文件：{display_path}") from exc
        if len(raw_content) > self._MAX_FILE_BYTES:
            raise WorkspaceError(
                f"文件超过 {self._MAX_FILE_BYTES} 字节读取上限：{display_path}"
            )
        try:
            return raw_content.decode("utf-8"), len(raw_content)
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"文件不是有效的 UTF-8 文本：{display_path}") from exc

    def _read_search_text(
        self,
        target: Path,
        remaining_bytes: int,
    ) -> tuple[str | None, int, bool]:
        """按文件和本轮搜索的双重字节上限尝试读取文本。"""

        read_limit = min(self._MAX_FILE_BYTES + 1, remaining_bytes + 1)
        try:
            with target.open("rb") as file:
                raw_content = file.read(read_limit)
        except OSError:
            return None, 0, False

        if len(raw_content) > remaining_bytes:
            return None, remaining_bytes, True
        if len(raw_content) > self._MAX_FILE_BYTES:
            return None, len(raw_content), False
        try:
            text = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            return None, len(raw_content), False
        if "\x00" in text:
            return None, len(raw_content), False
        return text, len(raw_content), False

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        """在目标目录创建临时文件，并使用原子替换提交完整内容。"""

        descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as file:
                descriptor = None
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            original_mode = stat.S_IMODE(target.stat().st_mode)
            os.chmod(temporary_path, original_mode)
            os.replace(temporary_path, target)
            temporary_path = None
        except OSError as exc:
            raise WorkspaceError(f"无法安全写入文件：{target.name}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _exclusive_write(target: Path, content: bytes) -> None:
        """先完整写入同目录临时文件，再以排他硬链接发布新文件。"""

        descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as file:
                descriptor = None
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise WorkspaceError(
                f"目标文件已经存在，拒绝覆盖：{target.name}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(f"无法安全创建文件：{target.name}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _checked_delete(target: Path, expected_content: bytes) -> None:
        """先原子移入同目录隔离文件，核对内容后再完成删除。"""

        descriptor: int | None = None
        quarantine_path: Path | None = None
        moved = False
        try:
            descriptor, quarantine_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".delete",
                dir=target.parent,
            )
            quarantine_path = Path(quarantine_name)
            os.close(descriptor)
            descriptor = None
            quarantine_path.unlink()

            os.replace(target, quarantine_path)
            moved = True
            try:
                mode = quarantine_path.lstat().st_mode
                actual_content = (
                    quarantine_path.read_bytes() if stat.S_ISREG(mode) else None
                )
            except OSError as exc:
                SafeWorkspace._restore_quarantined_file(quarantine_path, target)
                moved = False
                raise WorkspaceError(
                    f"删除时无法核对文件内容，已恢复：{target.name}"
                ) from exc
            if actual_content != expected_content:
                SafeWorkspace._restore_quarantined_file(quarantine_path, target)
                moved = False
                raise WorkspaceError(
                    f"文件在删除提交时发生变化，已恢复：{target.name}"
                )
            try:
                quarantine_path.unlink()
            except OSError as exc:
                SafeWorkspace._restore_quarantined_file(quarantine_path, target)
                moved = False
                raise WorkspaceError(
                    f"无法安全删除文件，已恢复：{target.name}"
                ) from exc
            moved = False
            quarantine_path = None
        except WorkspaceError:
            raise
        except OSError as exc:
            if moved and quarantine_path is not None:
                try:
                    SafeWorkspace._restore_quarantined_file(quarantine_path, target)
                    moved = False
                except WorkspaceError as restore_exc:
                    raise restore_exc from exc
            raise WorkspaceError(f"无法安全删除文件：{target.name}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if not moved and quarantine_path is not None:
                try:
                    quarantine_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _restore_quarantined_file(quarantine_path: Path, target: Path) -> None:
        """不覆盖并发目标地恢复隔离文件，恢复失败时保留隔离副本。"""

        try:
            os.link(quarantine_path, target, follow_symlinks=False)
            quarantine_path.unlink()
        except OSError as exc:
            raise WorkspaceError(
                "删除过程中检测到并发变化，且无法把原文件恢复到原路径；"
                f"隔离副本保留为 {quarantine_path.name}"
            ) from exc

    @classmethod
    def _sanitized_test_environment(cls) -> dict[str, str]:
        """移除常见凭据和可注入 pytest/Python 行为的环境变量。"""

        environment = {
            name: value
            for name, value in os.environ.items()
            if not any(marker in name.upper() for marker in cls._SENSITIVE_ENV_MARKERS)
        }
        injected_behavior_variables = (
            "PYTHONINSPECT",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTEST_ADDOPTS",
            "PYTEST_PLUGINS",
        )
        for name in injected_behavior_variables:
            environment.pop(name, None)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        return environment

    @classmethod
    def _truncate_test_output(cls, output: str) -> tuple[str, bool]:
        """保留测试输出的开头和结尾，避免失败详情挤出模型上下文。"""

        if len(output) <= cls._MAX_TEST_OUTPUT_CHARS:
            return output, False
        marker = "\n... [测试输出已截断] ...\n"
        head_chars = min(5_000, cls._MAX_TEST_OUTPUT_CHARS // 4)
        tail_chars = cls._MAX_TEST_OUTPUT_CHARS - head_chars - len(marker)
        return f"{output[:head_chars]}{marker}{output[-tail_chars:]}", True

    @staticmethod
    def _decode_test_output(output: str | bytes | None) -> str:
        """统一 TimeoutExpired 可能返回的文本或字节输出。"""

        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output

    def _collect_visible_search_files(
        self,
        target: Path,
    ) -> tuple[list[Path], int, bool]:
        """在目录项、目录数和文件数上限内收集安全搜索候选。"""

        if target.is_file():
            return [target], 0, False

        candidates: list[Path] = []
        pending_directories = [target]
        visited_directories = 0
        visited_entries = 0
        skipped_directories = 0
        truncated = False

        while pending_directories:
            if visited_directories >= self._MAX_SEARCH_DIRECTORIES:
                truncated = True
                break

            current = pending_directories.pop()
            visited_directories += 1
            entries: list[os.DirEntry[str]] = []
            entry_limit_reached = False
            try:
                with os.scandir(current) as iterator:
                    for entry in iterator:
                        if visited_entries >= self._MAX_SEARCH_ENTRIES:
                            truncated = True
                            entry_limit_reached = True
                            break
                        visited_entries += 1
                        entries.append(entry)
            except OSError:
                skipped_directories += 1
                continue

            child_directories: list[Path] = []
            for entry in sorted(entries, key=lambda item: (item.name.casefold(), item.name)):
                candidate = Path(entry.path)
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                    is_symlink = entry.is_symlink()
                except OSError:
                    truncated = True
                    continue

                if is_directory:
                    lowered_name = entry.name.casefold()
                    if (
                        lowered_name not in self._IGNORED_DIRECTORY_NAMES
                        and lowered_name not in self._SENSITIVE_DIRECTORY_NAMES
                        and self._is_visible_path(candidate)
                    ):
                        child_directories.append(candidate)
                    continue

                if not (is_file or is_symlink):
                    continue
                if not self._is_visible_path(candidate) or not candidate.is_file():
                    continue
                if len(candidates) >= self._MAX_SEARCH_FILES:
                    return candidates, skipped_directories, True
                candidates.append(candidate)

            if entry_limit_reached:
                break
            pending_directories.extend(reversed(child_directories))

        return candidates, skipped_directories, truncated

    @classmethod
    def _search_excerpt(
        cls,
        line: str,
        match_start: int,
        match_end: int,
    ) -> tuple[str, bool]:
        """截取包含命中位置的单行片段，避免超长行占满上下文。"""

        if len(line) <= cls._MAX_SEARCH_LINE_CHARS:
            return line, False

        match_length = max(1, match_end - match_start)
        surrounding_chars = max(0, cls._MAX_SEARCH_LINE_CHARS - match_length)
        start = max(0, match_start - surrounding_chars // 2)
        end = min(len(line), start + cls._MAX_SEARCH_LINE_CHARS)
        start = max(0, end - cls._MAX_SEARCH_LINE_CHARS)
        excerpt = line[start:end]
        if start > 0:
            excerpt = f"…{excerpt}"
        if end < len(line):
            excerpt = f"{excerpt}…"
        return excerpt, True

    def _assert_read_allowed(self, path: Path) -> None:
        if path in self._protected_paths:
            raise WorkspaceError("禁止通过 Agent 工具访问控制面配置文件")
        relative = path.relative_to(self.root)
        lowered_parts = tuple(part.casefold() for part in relative.parts)
        if os.name == "nt" and any(":" in part for part in relative.parts):
            raise WorkspaceError("禁止访问 Windows 备用数据流")
        if any(part in self._SENSITIVE_DIRECTORY_NAMES for part in lowered_parts):
            raise WorkspaceError("禁止访问敏感目录")

        file_name = path.name.casefold()
        is_private_env = file_name == ".env" or (
            file_name.startswith(".env.") and file_name != ".env.example"
        )
        if (
            is_private_env
            or file_name in self._SENSITIVE_FILE_NAMES
            or path.suffix.casefold() in self._SENSITIVE_FILE_SUFFIXES
        ):
            raise WorkspaceError("禁止访问敏感文件")

    def _is_visible_path(self, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.root)
            self._assert_read_allowed(resolved)
        except (OSError, ValueError):
            return False
        return True


def build_read_only_registry(workspace: SafeWorkspace) -> ToolRegistry:
    """创建只读代码检查工具注册表。"""

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="list_files",
            description="列出已配置项目工作区内的文件和目录。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对目录路径；使用 '.' 表示工作区根目录。",
                        "default": ".",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回多少个条目，取值范围为 1 到 500。",
                        "default": 200,
                    },
                },
                "additionalProperties": False,
            },
            handler=workspace.list_files,
        )
    )
    registry.register(
        ToolDefinition(
            name="read_file",
            description="读取工作区内的 UTF-8 文本文件，并返回行号。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "待读取文件的相对路径。",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "开始读取的行号，从 1 开始。",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "可选的结束行号，包含该行。",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=workspace.read_file,
            file_access=ToolFileAccess.READ,
            path_argument="path",
            path_normalizer=workspace.canonical_file_path,
        )
    )
    registry.register(
        ToolDefinition(
            name="search_code",
            description=(
                "在工作区内的 UTF-8 文本文件中逐行搜索字面量，并返回相对路径、"
                "行号、列号和命中内容；命中后可使用 read_file 查看上下文。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的非空字面量，不能包含换行符。",
                    },
                    "path": {
                        "type": "string",
                        "description": "相对文件或目录路径；使用 '.' 搜索整个工作区。",
                        "default": ".",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "是否区分大小写，默认为 false。",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回多少条匹配，取值范围为 1 到 100。",
                        "default": 50,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=workspace.search_code,
        )
    )
    return registry


def build_agent_registry(workspace: SafeWorkspace) -> ToolRegistry:
    """创建包含受控单项与批量文件变更能力的 Agent 工具注册表。"""

    registry = build_read_only_registry(workspace)
    registry.state.configure_required_validations(
        workspace.validation_config.required_names
    )
    registry.register(
        ToolDefinition(
            name="replace_text",
            description=(
                "精确替换已读取 UTF-8 文件中的字面量。写入前会向用户展示完整"
                " diff 并要求确认；实际出现次数必须与 expected_replacements 完全一致。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要修改的工作区相对文件路径。",
                        "minLength": 1,
                    },
                    "old_text": {
                        "type": "string",
                        "description": "必须原样存在的非空旧文本。",
                        "minLength": 1,
                    },
                    "new_text": {
                        "type": "string",
                        "description": "用于替换的新文本，可以为空字符串。",
                    },
                    "expected_replacements": {
                        "type": "integer",
                        "description": "旧文本预期出现次数，取值范围为 1 到 100。",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            handler=workspace.replace_text,
            file_access=ToolFileAccess.WRITE,
            path_argument="path",
            path_normalizer=workspace.canonical_file_path,
        )
    )
    registry.register(
        ToolDefinition(
            name="create_file",
            description=(
                "在父目录已经存在时创建新的 UTF-8 文件。目标必须尚不存在；"
                "写入前会向用户展示完整新增 diff 并要求确认。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要创建的工作区相对文件路径。",
                        "minLength": 1,
                    },
                    "content": {
                        "type": "string",
                        "description": "新文件的完整非空 UTF-8 文本内容。",
                        "minLength": 1,
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=workspace.create_file,
            file_access=ToolFileAccess.CREATE,
            path_argument="path",
            path_normalizer=workspace.canonical_new_file_path,
        )
    )
    registry.register(
        ToolDefinition(
            name="delete_file",
            description=(
                "删除已经读取的 UTF-8 文本文件。删除前会向用户展示完整"
                " diff 并要求确认；确认期间内容发生变化时会安全取消。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要删除的工作区相对文件路径。",
                        "minLength": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=workspace.delete_file,
            file_access=ToolFileAccess.DELETE,
            path_argument="path",
            path_normalizer=workspace.canonical_file_path,
        )
    )
    registry.register(
        ToolDefinition(
            name="apply_change_set",
            description=(
                "统一预览并应用 1 到 20 项相互独立的文件创建、精确替换或删除。"
                "replace/delete 目标都必须先读取；同一路径不能在一个批次中出现两次。"
                "用户只确认一次，失败时会恢复本批次已经应用的变更。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "description": (
                            "批量操作。create 需要 kind/path/content；replace 需要 "
                            "kind/path/old_text/new_text，可选 expected_replacements；"
                            "delete 只需要 kind/path。"
                        ),
                        "minItems": 1,
                        "maxItems": SafeWorkspace._MAX_BATCH_CHANGES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": ["create", "replace", "delete"],
                                },
                                "path": {"type": "string", "minLength": 1},
                                "content": {"type": "string", "minLength": 1},
                                "old_text": {"type": "string", "minLength": 1},
                                "new_text": {"type": "string"},
                                "expected_replacements": {
                                    "type": "integer",
                                    "default": 1,
                                    "minimum": 1,
                                    "maximum": SafeWorkspace._MAX_REPLACEMENTS,
                                },
                            },
                            "required": ["kind", "path"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["operations"],
                "additionalProperties": False,
            },
            handler=workspace.apply_change_set,
            file_access_resolver=workspace.change_set_file_accesses,
        )
    )
    registry.register(
        ToolDefinition(
            name="run_validation",
            description=(
                "运行启动时由用户固定配置的具名验证任务；模型只能选择名称，"
                "不能提供命令或参数。可用任务："
                + "；".join(
                    f"{task.name}（{'必选' if task.required else '可选'}）："
                    f"{task.description}"
                    for task in workspace.validation_config.tasks
                )
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": list(workspace.validation_config.task_names),
                        "description": "要运行的预配置验证任务名称。",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=workspace.run_validation,
            records_test_result=True,
            validation_name_argument="name",
        )
    )
    registry.register(
        ToolDefinition(
            name="run_tests",
            description=(
                "兼容入口：运行预配置的 tests 验证任务。该工具不接受命令参数，"
                "并返回通过状态、退出码和受限输出。"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=workspace.run_tests,
            records_test_result=True,
        )
    )
    return registry
