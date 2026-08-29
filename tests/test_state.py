import unittest

from safe_patch_agent.state import AgentState, AgentStateError


class AgentStateTests(unittest.TestCase):
    def test_read_file_authorizes_same_normalized_path(self) -> None:
        state = AgentState()

        state.mark_file_read("src/../src/app.py")
        state.require_file_read("./src/app.py")

        self.assertEqual(state.snapshot().read_files, ("src/app.py",))

    def test_unread_file_is_rejected_and_counted(self) -> None:
        state = AgentState()

        with self.assertRaisesRegex(AgentStateError, "必须先使用 read_file"):
            state.require_file_read("src/app.py")

        self.assertEqual(state.snapshot().blocked_write_attempts, 1)

    def test_reset_clears_all_task_state(self) -> None:
        state = AgentState()
        state.mark_file_read("src/app.py")
        state.mark_file_modified("src/app.py")
        state.record_test_result(False)
        state.mark_file_modified("src/app.py")
        with self.assertRaises(AgentStateError):
            state.require_file_read("src/other.py")

        state.reset()

        self.assertEqual(state.snapshot().read_files, ())
        self.assertEqual(state.snapshot().modified_files, ())
        self.assertEqual(state.snapshot().blocked_write_attempts, 0)
        self.assertEqual(state.snapshot().test_runs, 0)
        self.assertIsNone(state.snapshot().last_test_passed)
        self.assertFalse(state.snapshot().has_unverified_changes)

    def test_test_result_verifies_latest_changes_even_when_tests_fail(self) -> None:
        state = AgentState()
        state.mark_file_read("src/app.py")
        state.mark_file_modified("src/app.py")

        self.assertTrue(state.snapshot().has_unverified_changes)

        state.record_test_result(False)

        snapshot = state.snapshot()
        self.assertEqual(snapshot.test_runs, 1)
        self.assertFalse(snapshot.last_test_passed)
        self.assertFalse(snapshot.has_unverified_changes)

        state.mark_file_modified("src/app.py")
        self.assertTrue(state.snapshot().has_unverified_changes)

    def test_invalid_state_paths_are_rejected(self) -> None:
        for path in ("", ".", "../outside.py"):
            with self.subTest(path=path), self.assertRaises(AgentStateError):
                AgentState().mark_file_read(path)


if __name__ == "__main__":
    unittest.main()
