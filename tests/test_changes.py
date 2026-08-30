import hashlib
import unittest

from safe_patch_agent.changes import ChangeJournal, ChangeJournalError, ChangeKind


class ChangeJournalTests(unittest.TestCase):
    def test_record_exposes_safe_summary_and_hashes(self) -> None:
        journal = ChangeJournal()

        record = journal.record(
            path="demo.py",
            before_text="old\n",
            after_text="new\n",
            diff="-old\n+new\n",
            replacements=1,
        )
        summary = journal.summaries()[0]

        self.assertEqual(record.change_id, 1)
        self.assertEqual(summary.path, "demo.py")
        self.assertEqual(
            summary.before_sha256,
            hashlib.sha256(b"old\n").hexdigest(),
        )
        self.assertEqual(
            summary.after_sha256,
            hashlib.sha256(b"new\n").hexdigest(),
        )
        self.assertFalse(summary.rolled_back)
        self.assertIsNone(summary.test_passed)
        self.assertIs(summary.kind, ChangeKind.REPLACE)

    def test_creation_record_represents_missing_previous_file(self) -> None:
        journal = ChangeJournal()

        record = journal.record_creation(
            path="new.py",
            after_text="value = 1\n",
            diff="+value = 1\n",
        )
        summary = journal.summaries()[0]

        self.assertIs(record.kind, ChangeKind.CREATE)
        self.assertIsNone(record.before_text)
        self.assertIsNone(summary.before_sha256)
        self.assertEqual(record.original_bytes, 0)
        self.assertEqual(record.updated_bytes, len(b"value = 1\n"))

    def test_latest_test_result_updates_all_active_changes(self) -> None:
        journal = ChangeJournal()
        first = journal.record(
            path="first.py",
            before_text="1",
            after_text="2",
            diff="first",
            replacements=1,
        )
        journal.record_test_result(True)
        second = journal.record(
            path="second.py",
            before_text="a",
            after_text="b",
            diff="second",
            replacements=1,
        )
        journal.mark_rolled_back((second,))
        journal.record_test_result(False)

        self.assertFalse(first.test_passed)
        self.assertIsNone(second.test_passed)
        self.assertEqual(journal.pending_rollback_paths, ())

    def test_rollback_selection_is_newest_first(self) -> None:
        journal = ChangeJournal()
        for index in range(3):
            journal.record(
                path=f"{index}.txt",
                before_text=str(index),
                after_text=str(index + 1),
                diff=str(index),
                replacements=1,
            )

        self.assertEqual(
            [item.change_id for item in journal.records_for_rollback()],
            [3],
        )
        self.assertEqual(
            [item.change_id for item in journal.records_for_rollback("all")],
            [3, 2, 1],
        )
        self.assertEqual(
            [item.change_id for item in journal.records_for_rollback(2)],
            [2],
        )

    def test_capacity_is_checked_before_new_record(self) -> None:
        journal = ChangeJournal(max_records=1, max_stored_bytes=20)
        journal.record(
            path="one.txt",
            before_text="a",
            after_text="b",
            diff="diff",
            replacements=1,
        )

        with self.assertRaisesRegex(ChangeJournalError, "1 条上限"):
            journal.record(
                path="two.txt",
                before_text="c",
                after_text="d",
                diff="diff",
                replacements=1,
            )

    def test_invalid_or_missing_rollback_target_is_rejected(self) -> None:
        journal = ChangeJournal()

        with self.assertRaisesRegex(ChangeJournalError, "没有可回滚"):
            journal.records_for_rollback()
        with self.assertRaisesRegex(ChangeJournalError, "正整数"):
            journal.records_for_rollback(0)


if __name__ == "__main__":
    unittest.main()
