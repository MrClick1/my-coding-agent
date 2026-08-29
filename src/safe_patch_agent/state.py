"""Agent 单次运行期间的可审计状态，以及先读后写约束。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class AgentStateError(ValueError):
    """工具请求违反了 Agent 的状态约束。"""


@dataclass(frozen=True)
class AgentStateSnapshot:
    """一次 Agent 运行结束时返回的不可变状态快照。"""

    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    blocked_write_attempts: int
    test_runs: int
    last_test_passed: bool | None
    has_unverified_changes: bool


@dataclass
class AgentState:
    """记录当前任务读取和修改过的文件。"""

    _read_files: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _modified_files: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    blocked_write_attempts: int = field(default=0, init=False)
    test_runs: int = field(default=0, init=False)
    last_test_passed: bool | None = field(default=None, init=False)
    has_unverified_changes: bool = field(default=False, init=False)

    def reset(self) -> None:
        """开始新任务前清空上一任务的状态。"""

        self._read_files.clear()
        self._modified_files.clear()
        self.blocked_write_attempts = 0
        self.test_runs = 0
        self.last_test_passed = None
        self.has_unverified_changes = False

    def mark_file_read(self, path: str) -> None:
        """记录已成功读取完整或指定范围的文件。"""

        normalized = _normalize_relative_path(path)
        self._read_files[_path_key(normalized)] = normalized

    def require_file_read(self, path: str) -> None:
        """确保目标文件已在当前任务中成功读取。"""

        normalized = _normalize_relative_path(path)
        if _path_key(normalized) not in self._read_files:
            self.blocked_write_attempts += 1
            raise AgentStateError(
                f"修改文件前必须先使用 read_file 读取目标文件：{path}"
            )

    def mark_file_modified(self, path: str) -> None:
        """记录已经成功修改的文件。"""

        normalized = _normalize_relative_path(path)
        self._modified_files[_path_key(normalized)] = normalized
        self.has_unverified_changes = True

    def record_test_result(self, passed: bool) -> None:
        """记录一次固定测试运行，并验证最近一批修改已经接受检查。"""

        if not isinstance(passed, bool):
            raise AgentStateError("测试结果 passed 必须是布尔值")
        self.test_runs += 1
        self.last_test_passed = passed
        self.has_unverified_changes = False

    def snapshot(self) -> AgentStateSnapshot:
        """返回稳定排序的不可变状态。"""

        return AgentStateSnapshot(
            read_files=tuple(sorted(self._read_files.values(), key=str.casefold)),
            modified_files=tuple(sorted(self._modified_files.values(), key=str.casefold)),
            blocked_write_attempts=self.blocked_write_attempts,
            test_runs=self.test_runs,
            last_test_passed=self.last_test_passed,
            has_unverified_changes=self.has_unverified_changes,
        )


def _normalize_relative_path(path: str) -> str:
    """规范化工具使用的工作区相对路径，但不接触文件系统。"""

    if not isinstance(path, str) or not path.strip():
        raise AgentStateError("工具的文件路径必须是非空字符串")

    supplied = Path(path)
    if supplied.is_absolute():
        raise AgentStateError("工具状态只接受工作区相对路径")

    normalized = Path(os.path.normpath(path))
    if normalized == Path(".") or ".." in normalized.parts:
        raise AgentStateError("工具状态需要有效的工作区文件路径")

    return normalized.as_posix()


def _path_key(normalized_path: str) -> str:
    """生成符合当前操作系统大小写语义的内部比较键。"""

    return normalized_path.casefold() if os.name == "nt" else normalized_path
