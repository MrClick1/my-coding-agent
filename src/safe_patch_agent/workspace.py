"""工作区边界检查，以及目录、搜索和读取三个只读编程工具。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from safe_patch_agent.tooling import ToolDefinition, ToolFileAccess, ToolRegistry


class WorkspaceError(ValueError):
    """请求的路径或操作不安全或无效。"""


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

    def __init__(self, root: Path) -> None:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.exists():
            raise WorkspaceError(f"工作区不存在：{resolved_root}")
        if not resolved_root.is_dir():
            raise WorkspaceError(f"工作区路径不是目录：{resolved_root}")
        self.root = resolved_root

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
