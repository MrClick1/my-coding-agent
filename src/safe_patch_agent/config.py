"""模型客户端的配置加载逻辑。"""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """必需的模型配置缺失或无效。"""


_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
