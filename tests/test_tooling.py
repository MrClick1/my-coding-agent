import json
import unittest

from safe_patch_agent.messages import ToolCall
from safe_patch_agent.tooling import ToolDefinition, ToolRegistry


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


if __name__ == "__main__":
    unittest.main()
