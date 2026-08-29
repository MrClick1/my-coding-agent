"""在隔离临时工作区中运行并评分小型 Agent 基准测试。"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from safe_patch_agent.agent import AgentError, AgentResult, CodingAgent
from safe_patch_agent.cli import ChineseArgumentParser
from safe_patch_agent.config import ConfigurationError, LLMConfig
from safe_patch_agent.llm_client import (
    ChatCompletion,
    LLMClient,
    LLMError,
    OpenAICompatibleClient,
)
from safe_patch_agent.messages import ChatMessage, ToolCall
from safe_patch_agent.state import AgentStateSnapshot
from safe_patch_agent.tooling import ToolRegistry
from safe_patch_agent.workspace import SafeWorkspace, WorkspaceError, build_agent_registry

_CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ALLOWED_CASE_FIELDS = {
    "id",
    "description",
    "fixture",
    "goal",
    "expected_answer_contains",
    "required_tools",
    "expected_files",
    "require_tests_run",
    "require_tests_passed",
}


class BenchmarkError(ValueError):
    """基准测试清单、fixture 或报告写入无效。"""


@dataclass(frozen=True)
class ExpectedFile:
    """一次运行结束后，文件应具有的精确内容。"""

    path: str
    content: str


@dataclass(frozen=True)
class BenchmarkCase:
    """一个可复制到独立工作区的基准测试案例。"""

    id: str
    description: str
    fixture: Path
    goal: str
    expected_answer_contains: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    expected_files: tuple[ExpectedFile, ...] = ()
    require_tests_run: bool = False
    require_tests_passed: bool = False


@dataclass(frozen=True)
class BenchmarkCheck:
    """一个独立且可解释的自动评分检查。"""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """单个案例的运行轨迹摘要与评分结果。"""

    case_id: str
    description: str
    passed: bool
    checks: tuple[BenchmarkCheck, ...]
    error: str | None
    answer: str
    duration_seconds: float
    model_rounds: int
    tool_calls: int
    observed_tools: tuple[str, ...]
    successful_tools: tuple[str, ...]
    state: AgentStateSnapshot

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checks"] = [asdict(check) for check in self.checks]
        data["observed_tools"] = list(self.observed_tools)
        data["successful_tools"] = list(self.successful_tools)
        data["state"] = asdict(self.state)
        return data


@dataclass(frozen=True)
class BenchmarkReport:
    """一批案例的可序列化汇总报告。"""

    generated_at: str
    model: str
    total: int
    passed: int
    success_rate: float
    duration_seconds: float
    results: tuple[BenchmarkCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model": self.model,
            "total": self.total,
            "passed": self.passed,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "results": [result.to_dict() for result in self.results],
        }


ClientFactory = Callable[[BenchmarkCase], LLMClient]


class _RecordingClient:
    """记录模型轮次，确保失败案例仍能报告已发起的工具调用。"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.completions: list[ChatCompletion] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatCompletion:
        completion = self.client.complete(messages, tools)
        self.completions.append(completion)
        return completion

    @property
    def observed_tools(self) -> tuple[str, ...]:
        return tuple(
            call.name
            for completion in self.completions
            for call in completion.message.tool_calls
        )


class _RecordingRegistry(ToolRegistry):
    """记录实际执行和成功的工具，保留失败案例中的可审计轨迹。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.state = registry.state
        self.attempted_tools: list[str] = []
        self.successful_tools: list[str] = []

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas()

    def execute(self, call: ToolCall) -> str:
        self.attempted_tools.append(call.name)
        output = self.registry.execute(call)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return output
        if isinstance(payload, dict) and payload.get("ok", True) is True:
            self.successful_tools.append(call.name)
        return output


class BenchmarkRunner:
    """顺序执行案例；每个案例都使用全新的临时工作区和 Agent。"""

    def __init__(
        self,
        client_factory: ClientFactory,
        *,
        model_name: str = "未指定",
        max_rounds: int = 8,
        max_tool_calls: int = 32,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须至少为 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须至少为 1")
        self.client_factory = client_factory
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls

    def run(self, cases: Sequence[BenchmarkCase]) -> BenchmarkReport:
        if not cases:
            raise BenchmarkError("至少需要一个基准测试案例")

        started_at = time.perf_counter()
        results = tuple(self.run_case(case) for case in cases)
        passed = sum(result.passed for result in results)
        duration = time.perf_counter() - started_at
        return BenchmarkReport(
            generated_at=datetime.now(UTC).isoformat(),
            model=self.model_name,
            total=len(results),
            passed=passed,
            success_rate=passed / len(results),
            duration_seconds=round(duration, 6),
            results=results,
        )

    def run_case(self, case: BenchmarkCase) -> BenchmarkCaseResult:
        started_at = time.perf_counter()
        with TemporaryDirectory(prefix=f"safe-patch-{case.id}-") as temporary_directory:
            workspace_root = Path(temporary_directory) / "workspace"
            shutil.copytree(case.fixture, workspace_root)
            registry = _RecordingRegistry(
                build_agent_registry(
                    SafeWorkspace(
                        workspace_root,
                        replacement_approval=lambda _preview: True,
                    )
                )
            )
            client = _RecordingClient(self.client_factory(case))
            result: AgentResult | None = None
            error: str | None = None

            try:
                result = CodingAgent(
                    client=client,
                    registry=registry,
                    max_rounds=self.max_rounds,
                    max_tool_calls=self.max_tool_calls,
                ).run(case.goal)
            except (AgentError, LLMError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # pragma: no cover - 最后一层案例隔离
                error = f"{type(exc).__name__}: 未预期的案例运行错误：{exc}"

            answer = result.answer if result is not None else ""
            state = result.state if result is not None else registry.state.snapshot()
            observed_tools = client.observed_tools
            successful_tools = tuple(registry.successful_tools)
            checks = self._score_case(
                case=case,
                workspace_root=workspace_root,
                answer=answer,
                successful_tools=successful_tools,
                state=state,
                error=error,
            )

        duration = time.perf_counter() - started_at
        return BenchmarkCaseResult(
            case_id=case.id,
            description=case.description,
            passed=all(check.passed for check in checks),
            checks=checks,
            error=error,
            answer=answer,
            duration_seconds=round(duration, 6),
            model_rounds=(result.model_rounds if result is not None else len(client.completions)),
            tool_calls=(
                result.tool_calls
                if result is not None
                else len(registry.attempted_tools)
            ),
            observed_tools=observed_tools,
            successful_tools=successful_tools,
            state=state,
        )

    @staticmethod
    def _score_case(
        *,
        case: BenchmarkCase,
        workspace_root: Path,
        answer: str,
        successful_tools: tuple[str, ...],
        state: AgentStateSnapshot,
        error: str | None,
    ) -> tuple[BenchmarkCheck, ...]:
        checks = [
            BenchmarkCheck(
                name="Agent 正常结束",
                passed=error is None,
                detail="运行完成" if error is None else error,
            )
        ]

        folded_answer = answer.casefold()
        for marker in case.expected_answer_contains:
            found = marker.casefold() in folded_answer
            checks.append(
                BenchmarkCheck(
                    name=f"回答包含：{marker}",
                    passed=found,
                    detail="已找到" if found else "最终回答中未找到该文本",
                )
            )

        for tool_name in case.required_tools:
            used = tool_name in successful_tools
            checks.append(
                BenchmarkCheck(
                    name=f"调用工具：{tool_name}",
                    passed=used,
                    detail="已调用" if used else "未调用",
                )
            )

        for expected_file in case.expected_files:
            target = workspace_root / Path(expected_file.path)
            try:
                actual_content = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                matched = False
                detail = "文件不存在"
            except UnicodeError:
                matched = False
                detail = "文件不是有效的 UTF-8 文本"
            else:
                matched = actual_content == expected_file.content
                detail = "内容完全匹配" if matched else "文件内容与预期不一致"
            checks.append(
                BenchmarkCheck(
                    name=f"文件内容：{expected_file.path}",
                    passed=matched,
                    detail=detail,
                )
            )

        if case.require_tests_run:
            checks.append(
                BenchmarkCheck(
                    name="运行测试",
                    passed=state.test_runs > 0,
                    detail=f"共运行 {state.test_runs} 次",
                )
            )
        if case.require_tests_passed:
            checks.append(
                BenchmarkCheck(
                    name="测试通过",
                    passed=state.last_test_passed is True,
                    detail=f"最近结果：{state.last_test_passed}",
                )
            )
        if state.modified_files:
            checks.append(
                BenchmarkCheck(
                    name="修改已验证",
                    passed=not state.has_unverified_changes,
                    detail=(
                        "没有待验证修改"
                        if not state.has_unverified_changes
                        else "仍有修改未运行测试"
                    ),
                )
            )
        return tuple(checks)


def load_benchmark_cases(manifest_path: Path) -> tuple[BenchmarkCase, ...]:
    """加载并严格校验版本 1 的 JSON 基准测试清单。"""

    manifest_path = manifest_path.expanduser().resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"基准测试清单不存在：{manifest_path}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"基准测试清单不是有效的 UTF-8 JSON：{exc}") from exc

    if not isinstance(raw, dict):
        raise BenchmarkError("基准测试清单顶层必须是对象")
    if raw.get("version") != 1:
        raise BenchmarkError("基准测试清单 version 必须为 1")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BenchmarkError("基准测试清单 cases 必须是非空列表")

    manifest_root = manifest_path.parent
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise BenchmarkError(f"第 {index} 个案例必须是对象")
        unknown_fields = set(raw_case) - _ALLOWED_CASE_FIELDS
        if unknown_fields:
            joined = ", ".join(sorted(unknown_fields))
            raise BenchmarkError(f"第 {index} 个案例包含未知字段：{joined}")

        case_id = _required_string(raw_case, "id", index)
        if not _CASE_ID_PATTERN.fullmatch(case_id):
            raise BenchmarkError(
                f"第 {index} 个案例 id 只能包含小写字母、数字、下划线和连字符"
            )
        if case_id in seen_ids:
            raise BenchmarkError(f"案例 id 重复：{case_id}")
        seen_ids.add(case_id)

        fixture_text = _required_string(raw_case, "fixture", index)
        fixture_relative = _safe_relative_path(fixture_text, f"案例 {case_id} 的 fixture")
        fixture = (manifest_root / fixture_relative).resolve()
        try:
            fixture.relative_to(manifest_root)
        except ValueError as exc:
            raise BenchmarkError(f"案例 {case_id} 的 fixture 超出了清单目录") from exc
        if not fixture.is_dir():
            raise BenchmarkError(f"案例 {case_id} 的 fixture 目录不存在：{fixture_text}")
        if any(path.is_symlink() for path in fixture.rglob("*")):
            raise BenchmarkError(f"案例 {case_id} 的 fixture 不能包含符号链接")

        expected_files_raw = raw_case.get("expected_files", {})
        if not isinstance(expected_files_raw, dict):
            raise BenchmarkError(f"案例 {case_id} 的 expected_files 必须是对象")
        expected_files: list[ExpectedFile] = []
        for file_path, content in expected_files_raw.items():
            if not isinstance(file_path, str) or not isinstance(content, str):
                raise BenchmarkError(
                    f"案例 {case_id} 的 expected_files 必须将字符串路径映射到字符串内容"
                )
            normalized = _safe_relative_path(
                file_path, f"案例 {case_id} 的 expected_files 路径"
            ).as_posix()
            expected_files.append(ExpectedFile(normalized, content))

        require_tests_run = _optional_bool(raw_case, "require_tests_run", case_id)
        require_tests_passed = _optional_bool(
            raw_case, "require_tests_passed", case_id
        )
        if require_tests_passed:
            require_tests_run = True

        case = BenchmarkCase(
            id=case_id,
            description=_required_string(raw_case, "description", index),
            fixture=fixture,
            goal=_required_string(raw_case, "goal", index),
            expected_answer_contains=_string_tuple(
                raw_case.get("expected_answer_contains", []),
                f"案例 {case_id} 的 expected_answer_contains",
            ),
            required_tools=_string_tuple(
                raw_case.get("required_tools", []),
                f"案例 {case_id} 的 required_tools",
            ),
            expected_files=tuple(expected_files),
            require_tests_run=require_tests_run,
            require_tests_passed=require_tests_passed,
        )
        if not (
            case.expected_answer_contains
            or case.required_tools
            or case.expected_files
            or case.require_tests_run
        ):
            raise BenchmarkError(f"案例 {case_id} 至少需要一项评分条件")
        cases.append(case)
    return tuple(cases)


def _required_string(raw: Mapping[str, Any], field: str, index: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"第 {index} 个案例的 {field} 必须是非空字符串")
    return value.strip()


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise BenchmarkError(f"{label} 必须是非空字符串组成的列表")
    return tuple(item.strip() for item in value)


def _optional_bool(raw: Mapping[str, Any], field: str, case_id: str) -> bool:
    value = raw.get(field, False)
    if not isinstance(value, bool):
        raise BenchmarkError(f"案例 {case_id} 的 {field} 必须是布尔值")
    return value


def _safe_relative_path(value: str, label: str) -> Path:
    if not value or "\\" in value:
        raise BenchmarkError(f"{label} 必须使用非空的正斜杠相对路径")
    path = Path(value)
    if path.is_absolute() or path.anchor or any(part in {"", ".", ".."} for part in path.parts):
        raise BenchmarkError(f"{label} 必须是不能包含 '.' 或 '..' 的相对路径")
    return path


def build_parser() -> ChineseArgumentParser:
    parser = ChineseArgumentParser(
        prog="safe-patch-benchmark",
        description="用真实模型在隔离工作区运行基准测试；运行会产生模型 API 调用。",
        add_help=False,
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/cases.json"),
        help="基准测试清单（默认：benchmarks/cases.json）",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="模型配置文件（默认：.env）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/latest.json"),
        help="JSON 报告路径（默认：benchmark-results/latest.json）",
    )
    parser.add_argument("--max-rounds", type=int, default=8, help="每个案例的模型轮数上限")
    parser.add_argument(
        "--max-tool-calls", type=int, default=32, help="每个案例的工具调用上限"
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = LLMConfig.load(args.env_file)
        cases = load_benchmark_cases(args.manifest)
        report = BenchmarkRunner(
            lambda _case: OpenAICompatibleClient(config),
            model_name=config.model,
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
        ).run(cases)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        BenchmarkError,
        ConfigurationError,
        WorkspaceError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    for result in report.results:
        status = "通过" if result.passed else "失败"
        print(f"[{status}] {result.case_id}：{result.description}")
        for check in result.checks:
            if not check.passed:
                print(f"  - {check.name}：{check.detail}")
    print(
        f"总计：{report.passed}/{report.total}，"
        f"成功率 {report.success_rate:.1%}，报告：{output}"
    )
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
