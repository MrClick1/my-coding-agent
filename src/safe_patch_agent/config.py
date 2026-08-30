"""模型客户端的配置加载逻辑。"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """必需的模型配置缺失或无效。"""


_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALIDATION_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def read_env_file(path: Path) -> dict[str, str]:
    """读取本项目所需的简化版 ``KEY=VALUE`` 配置。

    解析器刻意不支持变量插值和 Shell 求值，确保配置行为可预测，并避免
    ``.env`` 文件执行命令。
    """

    if not path.exists():
        return {}
    if not path.is_file():
        raise ConfigurationError(f"环境配置路径不是文件：{path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f".env 第 {line_number} 行无效：缺少 '='")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY_PATTERN.fullmatch(key):
            raise ConfigurationError(f".env 第 {line_number} 行的键名无效：{key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class ValidationTask:
    """由用户配置、只能按名称调用的一项固定验证命令。"""

    name: str
    description: str
    command: tuple[str, ...]
    timeout_seconds: float = 120.0
    required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _VALIDATION_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ConfigurationError(
                "验证任务名称必须以字母开头，且只能包含字母、数字、下划线和连字符"
            )
        if not isinstance(self.description, str):
            raise ConfigurationError("验证任务 description 必须是字符串")
        description = self.description.strip()
        if not description or len(description) > 300:
            raise ConfigurationError("验证任务 description 必须为 1 到 300 个字符")
        if not isinstance(self.command, tuple) or not 1 <= len(self.command) <= 32:
            raise ConfigurationError("验证任务 command 必须包含 1 到 32 个参数")
        for argument in self.command:
            if (
                not isinstance(argument, str)
                or not argument
                or "\x00" in argument
                or len(argument) > 1_000
            ):
                raise ConfigurationError(
                    "验证任务 command 参数必须是 1 到 1000 个字符的安全字符串"
                )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 1 <= self.timeout_seconds <= 600
        ):
            raise ConfigurationError("验证任务 timeout_seconds 必须在 1 到 600 之间")
        if not isinstance(self.required, bool):
            raise ConfigurationError("验证任务 required 必须是布尔值")
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    @property
    def resolved_command(self) -> tuple[str, ...]:
        """展开受支持的 Python 解释器占位符，但不进行 Shell 求值。"""

        return tuple(
            sys.executable if argument == "{python}" else argument
            for argument in self.command
        )

    @property
    def display_command(self) -> str:
        """返回不暴露虚拟环境绝对路径的可读命令。"""

        return " ".join(
            "python" if argument == "{python}" else argument
            for argument in self.command
        )


@dataclass(frozen=True)
class ValidationConfig:
    """启动时冻结的具名验证任务集合。"""

    tasks: tuple[ValidationTask, ...]

    _MAX_CONFIG_BYTES = 64_000
    _MAX_TASKS = 20

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, tuple) or not 1 <= len(self.tasks) <= self._MAX_TASKS:
            raise ConfigurationError(
                f"验证配置必须包含 1 到 {self._MAX_TASKS} 个任务"
            )
        if any(not isinstance(task, ValidationTask) for task in self.tasks):
            raise ConfigurationError("验证配置 tasks 必须只包含 ValidationTask")
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ConfigurationError("验证任务名称不能重复")
        if "tests" not in names:
            raise ConfigurationError("验证配置必须包含兼容任务 tests")
        if not any(task.required for task in self.tasks):
            raise ConfigurationError("验证配置至少需要一个 required = true 的任务")

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks if task.required)

    def get(self, name: str) -> ValidationTask:
        for task in self.tasks:
            if task.name == name:
                return task
        raise ConfigurationError(f"未知验证任务：{name}")

    @classmethod
    def default(cls) -> ValidationConfig:
        """返回与旧版固定 pytest 行为兼容的默认配置。"""

        return cls(
            tasks=(
                ValidationTask(
                    name="tests",
                    description="运行项目 pytest 测试",
                    command=("{python}", "-m", "pytest", "-q"),
                    timeout_seconds=120,
                    required=True,
                ),
            )
        )

    @classmethod
    def load(cls, path: Path) -> ValidationConfig:
        """从专用 TOML 文件加载任务；文件不存在时使用兼容默认值。"""

        if not path.exists():
            return cls.default()
        if not path.is_file():
            raise ConfigurationError(f"验证配置路径不是文件：{path}")
        try:
            raw_content = path.read_bytes()
        except OSError as exc:
            raise ConfigurationError(f"无法读取验证配置：{path}") from exc
        if len(raw_content) > cls._MAX_CONFIG_BYTES:
            raise ConfigurationError(
                f"验证配置超过 {cls._MAX_CONFIG_BYTES} 字节上限"
            )
        try:
            data = tomllib.loads(raw_content.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError("验证配置不是有效的 UTF-8 TOML") from exc
        if set(data) != {"validation"}:
            raise ConfigurationError("验证配置顶层只能包含 [validation]")
        validation = data.get("validation")
        if not isinstance(validation, Mapping) or set(validation) != {"tasks"}:
            raise ConfigurationError("[validation] 必须且只能包含 tasks")
        raw_tasks = validation.get("tasks")
        if not isinstance(raw_tasks, Mapping):
            raise ConfigurationError("[validation.tasks] 必须包含具名任务表")

        tasks: list[ValidationTask] = []
        allowed_fields = {"description", "command", "timeout_seconds", "required"}
        for name, raw_task in raw_tasks.items():
            if not isinstance(name, str) or not isinstance(raw_task, Mapping):
                raise ConfigurationError("每个验证任务都必须是 TOML 表")
            unexpected = set(raw_task) - allowed_fields
            if unexpected:
                fields = ", ".join(sorted(str(field) for field in unexpected))
                raise ConfigurationError(f"验证任务 {name} 包含未知字段：{fields}")
            command = raw_task.get("command")
            if not isinstance(command, list):
                raise ConfigurationError(f"验证任务 {name} 的 command 必须是字符串数组")
            try:
                command_tuple = tuple(command)
                task = ValidationTask(
                    name=name,
                    description=raw_task.get("description", name),
                    command=command_tuple,
                    timeout_seconds=raw_task.get("timeout_seconds", 120),
                    required=raw_task.get("required", False),
                )
            except TypeError as exc:
                raise ConfigurationError(f"验证任务 {name} 的字段类型无效") from exc
            tasks.append(task)
        return cls(tuple(tasks))


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible Chat Completions 接口配置。"""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()

        if not api_key:
            raise ConfigurationError("LLM_API_KEY 不能为空")
        if not model:
            raise ConfigurationError("LLM_MODEL 不能为空")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("LLM_BASE_URL 必须是绝对 http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ConfigurationError("LLM_BASE_URL 不能包含嵌入式账号或密码")
        if parsed.query or parsed.fragment:
            raise ConfigurationError("LLM_BASE_URL 不能包含查询字符串或 URL 片段")
        if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
            raise ConfigurationError(
                "LLM_BASE_URL 必须使用 HTTPS，只有本机回环地址可以使用 HTTP"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("LLM_TIMEOUT_SECONDS 必须是有限的正数")

        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)

    @property
    def chat_completions_url(self) -> str:
        """返回约定的 OpenAI-compatible Chat Completions 地址。"""

        return f"{self.base_url}/chat/completions"

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> LLMConfig:
        """从 ``os.environ`` 等键值映射创建配置。"""

        required = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        missing = [name for name in required if not values.get(name, "").strip()]
        if missing:
            joined = ", ".join(missing)
            raise ConfigurationError(f"缺少必需配置：{joined}")

        timeout_text = values.get("LLM_TIMEOUT_SECONDS", "60").strip()
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise ConfigurationError("LLM_TIMEOUT_SECONDS 必须是数字") from exc

        return cls(
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"],
            model=values["LLM_MODEL"],
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def load(cls, env_file: Path | None = Path(".env")) -> LLMConfig:
        """先加载 ``.env``，再用进程环境变量覆盖同名配置。"""

        values = read_env_file(env_file) if env_file is not None else {}
        values.update(os.environ)
        return cls.from_mapping(values)
