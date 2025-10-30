"""Lightweight smoke tests for the DiagnosAI FastAPI backend."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, TypedDict

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
BASE_URL = DEFAULT_BASE_URL


class HealthResponse(TypedDict):
    status: str
    services: dict[str, str]


class ChatPayload(TypedDict, total=False):
    patient_question: str
    context: str
    conversation_id: str


class ChatResponse(TypedDict):
    model: str
    mode: Literal["fallback", "inference"]
    response: dict[str, object]


@dataclass(frozen=True)
class CheckResult:
    name: str
    success: bool
    detail: str


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.success else "FAIL"
    print(f"[{status}] {result.name}: {result.detail}")


def call(endpoint: str, method: str = "GET", json_payload: dict | None = None) -> requests.Response:
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.request(method, url, json=json_payload, timeout=10)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def check_health() -> CheckResult:
    try:
        resp = call("/health")
        data: HealthResponse = resp.json()
    except Exception as exc:  # pragma: no cover - manual smoke script
        return CheckResult("/health", False, str(exc))

    if data.get("status") != "ok":
        return CheckResult("/health", False, json.dumps(data))

    return CheckResult("/health", True, "service healthy")


def check_chat() -> CheckResult:
    payload: ChatPayload = {
        "patient_question": "I have a mild cough and slight fever. Should I see a doctor?",
        "context": """Patient reports cough for 3 days, low-grade fever, no shortness of breath.""",
    }
    try:
        resp = call("/api/chat", method="POST", json_payload=payload)
        data: ChatResponse = resp.json()
    except Exception as exc:  # pragma: no cover - manual smoke script
        return CheckResult("/api/chat", False, str(exc))

    response = data.get("response")
    if not isinstance(response, dict):
        return CheckResult("/api/chat", False, "missing structured response")

    if "triage" not in response or "recommendation" not in response:
        return CheckResult("/api/chat", False, f"unexpected keys: {sorted(response.keys())}")

    mode = data.get("mode", "fallback")
    detail = f"mode={mode}, recommendation={response.get('recommendation')}"
    return CheckResult("/api/chat", True, detail)


def run_checks(checks: Iterable[Callable[[], CheckResult]]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:  # pragma: no cover
            results.append(CheckResult(check.__name__, False, str(exc)))
    return results


def resolve_base_url(argv: list[str]) -> str:
    if len(argv) > 1:
        return argv[1].rstrip("/")

    env_value = os.environ.get("DIAGNOSAI_BASE_URL")
    if env_value:
        return env_value.rstrip("/")

    return DEFAULT_BASE_URL


def main() -> int:
    global BASE_URL
    BASE_URL = resolve_base_url(sys.argv)

    checks = (check_health, check_chat)
    results = run_checks(checks)

    failures = [r for r in results if not r.success]
    for result in results:
        _print_result(result)

    if failures:
        print("\nSmoke tests failed", file=sys.stderr)
        return 1

    print("\nAll smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
