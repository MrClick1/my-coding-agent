"""工作区边界检查，以及第一阶段的两个只读编程工具。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from safe_patch_agent.tooling import ToolDefinition, ToolRegistry


class WorkspaceError(ValueError):
    """请求的路径或操作不安全或无效。"""


class SafeWorkspace:
    """开放指定项目目录，同时阻止路径访问到工作区之外。"""

    _MAX_FILE_BYTES = 1_000_000
    _MAX_OUTPUT_CHARS = 40_000

    _IGNORED_DIRECTORY_NAMES = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
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
        try:
            with target.open("rb") as file:
                raw_content = file.read(self._MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise WorkspaceError(f"无法读取文件：{path}") from exc
        if len(raw_content) > self._MAX_FILE_BYTES:
            raise WorkspaceError(
                f"文件超过 {self._MAX_FILE_BYTES} 字节读取上限：{path}"
            )
        try:
            text = raw_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"文件不是有效的 UTF-8 文本：{path}") from exc

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
    """创建第一阶段使用的双工具注册表。"""

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
        )
    )
    return registry
