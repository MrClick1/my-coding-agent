"""当前进程会话中的文件修改日志。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class ChangeJournalError(ValueError):
    """修改日志请求无效，或日志容量不足。"""


class ChangeKind(StrEnum):
    """日志记录所代表的工作区写入类型。"""

    REPLACE = "replace"
    CREATE = "create"
    DELETE = "delete"


@dataclass
class ChangeRecord:
    """一次已经获得用户批准并成功写入的文件修改。"""

    change_id: int
    kind: ChangeKind
    path: str
    replacements: int
    diff: str
    before_sha256: str | None
    after_sha256: str | None
    original_bytes: int
    updated_bytes: int
    created_at: str
    test_passed: bool | None = None
    rolled_back: bool = False
    before_text: str | None = field(default=None, repr=False)
    after_text: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ChangeSummary:
    """适合展示给用户、且不包含完整文件内容的修改摘要。"""

    change_id: int
    kind: ChangeKind
    path: str
    replacements: int
    before_sha256: str | None
    after_sha256: str | None
    created_at: str
    test_passed: bool | None
    rolled_back: bool


class ChangeJournal:
    """保存有界修改历史，并提供回滚选择与测试状态同步。"""

    _DEFAULT_MAX_RECORDS = 100
    _DEFAULT_MAX_STORED_BYTES = 20_000_000

    def __init__(
        self,
        *,
        max_records: int = _DEFAULT_MAX_RECORDS,
        max_stored_bytes: int = _DEFAULT_MAX_STORED_BYTES,
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool):
            raise ChangeJournalError("max_records 必须是整数")
        if max_records < 1:
            raise ChangeJournalError("max_records 必须至少为 1")
        if not isinstance(max_stored_bytes, int) or isinstance(
            max_stored_bytes, bool
        ):
            raise ChangeJournalError("max_stored_bytes 必须是整数")
        if max_stored_bytes < 1:
            raise ChangeJournalError("max_stored_bytes 必须至少为 1")
        self.max_records = max_records
        self.max_stored_bytes = max_stored_bytes
        self._records: list[ChangeRecord] = []
        self._stored_bytes = 0
        self._next_id = 1
        self._pending_rollback_paths: set[str] = set()

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def pending_rollback_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending_rollback_paths, key=str.casefold))

    def ensure_can_record(
        self,
        before_text: str | None,
        after_text: str | None,
    ) -> None:
        """在写文件前确认日志能够完整保存可回滚内容。"""

        if len(self._records) >= self.max_records:
            raise ChangeJournalError(
                f"修改日志已达到 {self.max_records} 条上限；文件未修改"
            )
        required_bytes = _encoded_size(before_text) + _encoded_size(after_text)
        if self._stored_bytes + required_bytes > self.max_stored_bytes:
            raise ChangeJournalError(
                "修改日志保存的可回滚内容将超过 "
                f"{self.max_stored_bytes} 字节上限；文件未修改"
            )

    def record(
        self,
        *,
        path: str,
        before_text: str | None,
        after_text: str | None,
        diff: str,
        replacements: int,
        kind: ChangeKind = ChangeKind.REPLACE,
    ) -> ChangeRecord:
        """记录一次已经成功写入的修改；调用前应先检查容量。"""

        self.ensure_can_record(before_text, after_text)
        if not isinstance(kind, ChangeKind):
            try:
                kind = ChangeKind(kind)
            except (TypeError, ValueError) as exc:
                raise ChangeJournalError("修改日志 kind 无效") from exc
        if kind is ChangeKind.CREATE and before_text is not None:
            raise ChangeJournalError("创建记录的 before_text 必须是 None")
        if kind is ChangeKind.CREATE and after_text is None:
            raise ChangeJournalError("创建记录必须包含 after_text")
        if kind is ChangeKind.DELETE and before_text is None:
            raise ChangeJournalError("删除记录必须包含 before_text")
        if kind is ChangeKind.DELETE and after_text is not None:
            raise ChangeJournalError("删除记录的 after_text 必须是 None")
        if kind is ChangeKind.REPLACE and (
            before_text is None or after_text is None
        ):
            raise ChangeJournalError("替换记录必须包含 before_text 和 after_text")
        required_bytes = _encoded_size(before_text) + _encoded_size(after_text)
        record = ChangeRecord(
            change_id=self._next_id,
            kind=kind,
            path=path,
            replacements=replacements,
            diff=diff,
            before_sha256=_sha256(before_text) if before_text is not None else None,
            after_sha256=_sha256(after_text) if after_text is not None else None,
            original_bytes=_encoded_size(before_text),
            updated_bytes=_encoded_size(after_text),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            before_text=before_text,
            after_text=after_text,
        )
        self._records.append(record)
        self._stored_bytes += required_bytes
        self._next_id += 1
        return record

    def record_creation(
        self,
        *,
        path: str,
        after_text: str,
        diff: str,
    ) -> ChangeRecord:
        """记录一次成功创建的新文件。"""

        return self.record(
            path=path,
            before_text=None,
            after_text=after_text,
            diff=diff,
            replacements=0,
            kind=ChangeKind.CREATE,
        )

    def record_deletion(
        self,
        *,
        path: str,
        before_text: str,
        diff: str,
    ) -> ChangeRecord:
        """记录一次成功删除的现有文件。"""

        return self.record(
            path=path,
            before_text=before_text,
            after_text=None,
            diff=diff,
            replacements=0,
            kind=ChangeKind.DELETE,
        )

    def summaries(self) -> tuple[ChangeSummary, ...]:
        """按发生顺序返回不含文件正文的全部日志摘要。"""

        return tuple(
            ChangeSummary(
                change_id=record.change_id,
                kind=record.kind,
                path=record.path,
                replacements=record.replacements,
                before_sha256=record.before_sha256,
                after_sha256=record.after_sha256,
                created_at=record.created_at,
                test_passed=record.test_passed,
                rolled_back=record.rolled_back,
            )
            for record in self._records
        )

    def records_for_rollback(
        self,
        target: int | str | None = None,
    ) -> tuple[ChangeRecord, ...]:
        """按从新到旧的顺序选出要回滚的活动修改。"""

        if target is not None and target != "all" and (
            not isinstance(target, int) or isinstance(target, bool) or target < 1
        ):
            raise ChangeJournalError("回滚目标必须是正整数修改编号或 all")
        active = [record for record in self._records if not record.rolled_back]
        if not active:
            raise ChangeJournalError("当前会话没有可回滚的修改")
        if target is None:
            return (active[-1],)
        if target == "all":
            return tuple(reversed(active))
        assert isinstance(target, int)
        for record in reversed(active):
            if record.change_id == target:
                return (record,)
        raise ChangeJournalError(f"没有编号为 {target} 的可回滚修改")

    def mark_rolled_back(self, records: tuple[ChangeRecord, ...]) -> None:
        """在文件全部成功恢复后更新日志，并标记回滚结果待测试。"""

        for record in records:
            record.rolled_back = True
            self._pending_rollback_paths.add(record.path)

    def record_test_result(self, passed: bool) -> None:
        """把一次固定测试结果关联到此前尚未测试的活动修改。"""

        if not isinstance(passed, bool):
            raise ChangeJournalError("测试结果 passed 必须是布尔值")
        for record in self._records:
            if not record.rolled_back:
                record.test_passed = passed
        self._pending_rollback_paths.clear()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _encoded_size(text: str | None) -> int:
    return len(text.encode("utf-8")) if text is not None else 0
