import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from safe_patch_agent.config import (
    ConfigurationError,
    LLMConfig,
    ValidationConfig,
    read_env_file,
)


class ConfigTests(unittest.TestCase):
    def test_config_normalizes_values_and_hides_api_key(self) -> None:
        config = LLMConfig.from_mapping(
            {
                "LLM_API_KEY": " secret ",
                "LLM_BASE_URL": "https://example.test/v1/",
                "LLM_MODEL": " test-model ",
                "LLM_TIMEOUT_SECONDS": "12.5",
            }
        )

        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.timeout_seconds, 12.5)
        self.assertEqual(config.chat_completions_url, "https://example.test/v1/chat/completions")
        self.assertNotIn("secret", repr(config))

    def test_missing_values_are_reported_together(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "LLM_API_KEY, LLM_MODEL"):
            LLMConfig.from_mapping({"LLM_BASE_URL": "https://example.test/v1"})

    def test_invalid_timeout_is_rejected(self) -> None:
        values = {
            "LLM_API_KEY": "key",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_MODEL": "model",
            "LLM_TIMEOUT_SECONDS": "soon",
        }
        with self.assertRaisesRegex(ConfigurationError, "必须是数字"):
            LLMConfig.from_mapping(values)

    def test_remote_http_endpoint_is_rejected_to_protect_api_key(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "必须使用 HTTPS"):
            LLMConfig(
                api_key="key",
                base_url="http://example.test/v1",
                model="model",
            )

    def test_loopback_http_endpoint_is_allowed_for_local_models(self) -> None:
        config = LLMConfig(
            api_key="local-key",
            base_url="http://127.0.0.1:8000/v1",
            model="local-model",
        )

        self.assertEqual(config.chat_completions_url, "http://127.0.0.1:8000/v1/chat/completions")

    def test_non_finite_timeout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "有限的正数"):
            LLMConfig(
                api_key="key",
                base_url="https://example.test/v1",
                model="model",
                timeout_seconds=float("nan"),
            )

    def test_env_file_supports_comments_quotes_and_export(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / ".env"
            path.write_text(
                "# comment\nexport LLM_API_KEY='abc'\nLLM_MODEL=demo\n",
                encoding="utf-8",
            )

            self.assertEqual(
                read_env_file(path),
                {"LLM_API_KEY": "abc", "LLM_MODEL": "demo"},
            )

    def test_validation_config_loads_frozen_named_tasks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "safe-patch-agent.toml"
            path.write_text(
                """[validation.tasks.tests]
description = "Run tests"
command = ["{python}", "-m", "pytest", "-q"]
timeout_seconds = 45
required = true

[validation.tasks.lint]
description = "Run lint"
command = ["ruff", "check", "."]
required = false
""",
                encoding="utf-8",
            )

            config = ValidationConfig.load(path)

        self.assertEqual(config.task_names, ("tests", "lint"))
        self.assertEqual(config.required_names, ("tests",))
        self.assertEqual(config.get("tests").resolved_command[0], sys.executable)
        self.assertEqual(config.get("tests").display_command, "python -m pytest -q")
        self.assertEqual(config.get("tests").timeout_seconds, 45.0)

    def test_missing_validation_config_uses_compatible_pytest_default(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config = ValidationConfig.load(
                Path(temporary_directory) / "missing.toml"
            )

        self.assertEqual(config.task_names, ("tests",))
        self.assertEqual(config.required_names, ("tests",))
        self.assertEqual(
            config.get("tests").resolved_command,
            (sys.executable, "-m", "pytest", "-q"),
        )

    def test_validation_config_rejects_shell_string_and_missing_required_task(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "safe-patch-agent.toml"
            path.write_text(
                """[validation.tasks.tests]
description = "Unsafe"
command = "pytest -q && echo unsafe"
required = true
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "字符串数组"):
                ValidationConfig.load(path)

            path.write_text(
                """[validation.tasks.tests]
description = "Optional"
command = ["pytest", "-q"]
required = false
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "至少需要一个"):
                ValidationConfig.load(path)


if __name__ == "__main__":
    unittest.main()
