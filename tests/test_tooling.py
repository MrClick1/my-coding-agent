import json
import unittest

from safe_patch_agent.messages import ToolCall
from safe_patch_agent.tooling import (
    ToolDefinition,
    ToolFileAccess,
    ToolRegistrationError,
    ToolRegistry,
)


class ToolingTests(unittest.TestCase):
    def test_argument_binding_error_does_not_invoke_handler(self) -> None:
        invoked = False

        def requires_path(path: str) -> dict[str, object]:
            nonlocal invoked
            invoked = True
            return {"ok": True}

        registry = ToolRegistry()
        registry.register(ToolDefinition("read", "Read", {"type": "object"}, requires_path))

        result = json.loads(
            registry.execute(ToolCall(id="call-1", name="read", arguments={}))
        )

        self.assertFalse(invoked)
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_internal_type_error_is_contained_without_leaking_detail(self) -> None:
        def broken_handler() -> None:
            raise TypeError("private implementation detail")

        registry = ToolRegistry()
        registry.register(
            ToolDefinition("broken", "Broken", {"type": "object"}, broken_handler)
        )

        serialized = registry.execute(ToolCall(id="call-1", name="broken", arguments={}))
        result = json.loads(serialized)

        self.assertEqual(result["error"]["type"], "internal_tool_error")
        self.assertNotIn("private implementation detail", serialized)

    def test_write_tool_requires_successful_read_in_same_state(self) -> None:
        writes: list[str] = []

        def read_file(path: str) -> dict[str, object]:
            return {"ok": True, "path": path, "content": "old"}

        def write_file(path: str, content: str) -> dict[str, object]:
            writes.append(content)
            return {"ok": True, "path": path}

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "read_file",
                "读取",
                {"type": "object"},
                read_file,
                file_access=ToolFileAccess.READ,
                path_argument="path",
            )
        )
        registry.register(
            ToolDefinition(
                "write_file",
                "写入",
                {"type": "object"},
                write_file,
                file_access=ToolFileAccess.WRITE,
                path_argument="path",
            )
        )

        blocked = json.loads(
            registry.execute(
                ToolCall(
                    id="write-before-read",
                    name="write_file",
                    arguments={"path": "demo.py", "content": "new"},
                )
            )
        )
        registry.execute(
            ToolCall(id="read", name="read_file", arguments={"path": "demo.py"})
        )
        allowed = json.loads(
            registry.execute(
                ToolCall(
                    id="write-after-read",
                    name="write_file",
                    arguments={"path": "demo.py", "content": "new"},
                )
            )
        )

        self.assertFalse(blocked["ok"])
        self.assertIn("必须先使用 read_file", blocked["error"]["message"])
        self.assertEqual(writes, ["new"])
        self.assertTrue(allowed["ok"])
        self.assertEqual(registry.state.snapshot().read_files, ("demo.py",))
        self.assertEqual(registry.state.snapshot().modified_files, ("demo.py",))
        self.assertEqual(registry.state.snapshot().blocked_write_attempts, 1)

    def test_failed_read_does_not_authorize_write(self) -> None:
        def failed_read(path: str) -> dict[str, object]:
            return {"ok": False, "path": path}

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "read_file",
                "读取",
                {"type": "object"},
                failed_read,
                file_access=ToolFileAccess.READ,
                path_argument="path",
            )
        )
        registry.register(
            ToolDefinition(
                "write_file",
                "写入",
                {"type": "object"},
                lambda path: {"ok": True, "path": path},
                file_access=ToolFileAccess.WRITE,
                path_argument="path",
            )
        )

        registry.execute(
            ToolCall(id="failed-read", name="read_file", arguments={"path": "demo.py"})
        )
        result = json.loads(
            registry.execute(
                ToolCall(id="write", name="write_file", arguments={"path": "demo.py"})
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(registry.state.snapshot().read_files, ())

    def test_create_tool_does_not_require_read_but_activates_test_gate(self) -> None:
        created: list[str] = []
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "create_file",
                "创建",
                {"type": "object"},
                lambda path: created.append(path) or {"ok": True, "path": path},
                file_access=ToolFileAccess.CREATE,
                path_argument="path",
            )
        )

        result = json.loads(
            registry.execute(
                ToolCall(
                    id="create",
                    name="create_file",
                    arguments={"path": "new.py"},
                )
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(created, ["new.py"])
        snapshot = registry.state.snapshot()
        self.assertEqual(snapshot.read_files, ())
        self.assertEqual(snapshot.modified_files, ("new.py",))
        self.assertEqual(snapshot.blocked_write_attempts, 0)
        self.assertTrue(snapshot.has_unverified_changes)

    def test_file_access_metadata_must_be_complete(self) -> None:
        with self.assertRaisesRegex(ToolRegistrationError, "同时配置"):
            ToolDefinition(
                "read_file",
                "读取",
                {"type": "object"},
                lambda path: path,
                file_access=ToolFileAccess.READ,
            )

    def test_test_runner_updates_verification_state(self) -> None:
        registry = ToolRegistry()
        registry.state.mark_file_read("demo.py")
        registry.state.mark_file_modified("demo.py")
        registry.register(
            ToolDefinition(
                "run_tests",
                "测试",
                {"type": "object"},
                lambda: {"ok": True, "passed": False},
                records_test_result=True,
            )
        )

        result = json.loads(
            registry.execute(ToolCall(id="tests", name="run_tests", arguments={}))
        )

        self.assertTrue(result["ok"])
        snapshot = registry.state.snapshot()
        self.assertEqual(snapshot.test_runs, 1)
        self.assertFalse(snapshot.last_test_passed)
        self.assertFalse(snapshot.has_unverified_changes)

    def test_test_runner_must_return_boolean_passed_field(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "run_tests",
                "测试",
                {"type": "object"},
                lambda: {"ok": True, "passed": "yes"},
                records_test_result=True,
            )
        )

        result = json.loads(
            registry.execute(ToolCall(id="tests", name="run_tests", arguments={}))
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "tool_error")
        self.assertEqual(registry.state.snapshot().test_runs, 0)


if __name__ == "__main__":
    unittest.main()
