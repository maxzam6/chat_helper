from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_agent.active_memory_cache import ActiveMemoryCache
from memory_agent.graph_agent import GraphMemoryAgent
from memory_agent.llm_client import BaseLLMClient, LLMClient, MockLLMClient
from memory_agent.memory_store import MemoryStore
from memory_agent.semantic_retriever import SemanticRetriever
from memory_agent.vision_llm_client import BaseVisionLLMClient, MockVisionLLMClient, VisionLLMClient


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "id": "general_question",
        "payload": {
            "me_id": "default",
            "user_input": "LangGraph 是什么？",
            "screenshot_region": {"left": 1, "top": 1, "width": 20, "height": 20, "mock_image_base64": "mock"},
        },
        "expected_intent": "general_question",
    },
    {
        "id": "reply_advice_with_context",
        "seed_memories": [
            {
                "user_id": "bench_A001",
                "memory_type": "conversation_style",
                "content": "对方在压力大时倾向于短回复，不喜欢连续追问",
                "confidence": 0.86,
                "memory_status": "stable",
            }
        ],
        "payload": {
            "me_id": "default",
            "current_user_id": "bench_A001",
            "user_input": "她只回我哦，我该怎么回？",
            "chat_context": {
                "recent_messages": [
                    {"role": "target", "content": "哦"},
                    {"role": "me", "content": "你是不是不想聊了"},
                ],
                "previous_recent_messages": [],
            },
            "screenshot_region": {"left": 1, "top": 1, "width": 20, "height": 20, "mock_image_base64": "mock"},
        },
        "expected_intent": "reply_advice",
        "expected_memory_contents": ["对方在压力大时倾向于短回复，不喜欢连续追问"],
    },
    {
        "id": "profile_update",
        "payload": {
            "me_id": "default",
            "current_user_id": "bench_A001",
            "user_input": "记一下，她平时话少，不是冷淡",
            "screenshot_region": {"left": 1, "top": 1, "width": 20, "height": 20, "mock_image_base64": "mock"},
        },
        "expected_intent": "profile_update",
    },
]


class MetricsRecorder:
    def __init__(self) -> None:
        self.durations: dict[str, list[float]] = {}
        self.cache_decisions: list[bool] = []
        self.sync_successes: list[bool] = []

    def record_duration(self, name: str, seconds: float) -> None:
        self.durations.setdefault(name, []).append(seconds)

    def average(self, name: str) -> float | None:
        values = self.durations.get(name, [])
        if not values:
            return None
        return statistics.mean(values)


class TimedLLMClient(BaseLLMClient):
    def __init__(self, inner: BaseLLMClient, recorder: MetricsRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            return self.inner.generate_json(task, inputs)
        finally:
            self.recorder.record_duration(f"llm.{task}", time.perf_counter() - start)


class TimedVisionLLMClient(BaseVisionLLMClient):
    def __init__(self, inner: BaseVisionLLMClient, recorder: MetricsRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def parse_chat_screenshot(self, image_base64: str, user_input: str) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            return self.inner.parse_chat_screenshot(image_base64, user_input)
        finally:
            self.recorder.record_duration("vision.parse_chat_screenshot", time.perf_counter() - start)


class TimedGraphMemoryAgent(GraphMemoryAgent):
    def __init__(self, *args: Any, recorder: MetricsRecorder, **kwargs: Any) -> None:
        self.recorder = recorder
        super().__init__(*args, **kwargs)

    def _time_node(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            self.recorder.record_duration(f"node.{name}", time.perf_counter() - start)
        return result

    def pre_capture_screenshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._time_node("pre_capture_screenshot", super().pre_capture_screenshot, state)

    def classify_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._time_node("classify_intent", super().classify_intent, state)

    def capture_and_parse_chat(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._time_node("capture_and_parse_chat", super().capture_and_parse_chat, state)

    def retrieve_and_build_cache(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._time_node("retrieve_and_build_cache", super().retrieve_and_build_cache, state)

    def query_similarity_check(self, state: dict[str, Any]) -> dict[str, Any]:
        result = self._time_node("query_similarity_check", super().query_similarity_check, state)
        self.recorder.cache_decisions.append(result.get("reuse_cache") is True)
        return result

    def sync_dirty_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        result = self._time_node("sync_dirty_memory", super().sync_dirty_memory, state)
        self.recorder.sync_successes.append(not bool(result.get("sync_errors")))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real GraphMemoryAgent metrics.")
    parser.add_argument("--cases", type=Path, help="JSONL benchmark cases. Uses built-in smoke cases if omitted.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_metrics_report.json"))
    parser.add_argument("--mock", action="store_true", help="Use mock LLM/Vision clients. Not for final real metrics.")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    cases = load_cases(args.cases)
    recorder = MetricsRecorder()

    # ChromaDB can briefly keep Windows file handles open after use. Ignore
    # cleanup errors so metric reporting is not lost because of locked index files.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        store = MemoryStore(tmp_path / "benchmark_memory.db")
        store.init_db()
        retriever = SemanticRetriever(persist_path=tmp_path / "benchmark_chroma")

        llm: BaseLLMClient = MockLLMClient() if args.mock else LLMClient()
        vision: BaseVisionLLMClient | None
        if args.mock:
            vision = MockVisionLLMClient()
        elif os.getenv("VISION_API_KEY") and os.getenv("VISION_MODEL"):
            vision = VisionLLMClient()
        else:
            vision = None

        agent = TimedGraphMemoryAgent(
            memory_store=store,
            llm_client=TimedLLMClient(llm, recorder),
            vision_llm_client=TimedVisionLLMClient(vision, recorder) if vision else None,
            semantic_retriever=retriever,
            active_memory_cache=ActiveMemoryCache(),
            recorder=recorder,
        )

        case_results: list[dict[str, Any]] = []
        for _ in range(args.repeat):
            for case in cases:
                seed_case_memories(case, store, retriever)
                payload = normalize_payload(case.get("payload", {}), base_dir=args.cases.parent if args.cases else Path.cwd())
                start = time.perf_counter()
                error = None
                result: dict[str, Any] = {}
                try:
                    result = agent.process(payload, return_full_state=True)
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"
                total_seconds = time.perf_counter() - start

                case_result = evaluate_case(case, result, total_seconds, error)
                if "reply_advice" in result.get("task_list", []) or case.get("expected_intent") == "reply_advice":
                    recorder.record_duration("chain.reply_advice_total", total_seconds)
                case_results.append(case_result)

    report = build_report(recorder, case_results)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved metrics report: {args.output}")


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def normalize_payload(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized = dict(payload)
    image_path = normalized.pop("image_path", None)
    if image_path and not normalized.get("screenshot_base64"):
        path = Path(image_path)
        if not path.is_absolute():
            path = base_dir / path
        image_bytes = path.read_bytes()
        normalized["screenshot_base64"] = base64.b64encode(image_bytes).decode("ascii")
    return normalized


def seed_case_memories(case: dict[str, Any], store: MemoryStore, retriever: SemanticRetriever) -> None:
    for memory in case.get("seed_memories", []):
        if store.find_duplicate_memory(memory["user_id"], memory.get("memory_type"), memory["content"]):
            continue
        memory_id = store.save_memory(
            user_id=memory["user_id"],
            memory_type=memory.get("memory_type"),
            content=memory["content"],
            confidence=float(memory.get("confidence", 0.8)),
            memory_status=memory.get("memory_status", "stable"),
            source_type="benchmark",
            source_summary=case.get("id"),
        )
        retriever.add_memory(
            memory_id=memory_id,
            user_id=memory["user_id"],
            content=memory["content"],
            memory_status=memory.get("memory_status", "stable"),
            memory_type=memory.get("memory_type"),
        )


def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
    total_seconds: float,
    error: str | None,
) -> dict[str, Any]:
    relevant_memories = result.get("relevant_memories", [])
    expected_memory_ids = set(case.get("expected_memory_ids", []))
    expected_memory_contents = set(case.get("expected_memory_contents", []))
    retrieved_ids = {memory.get("id") for memory in relevant_memories[:5]}
    retrieved_contents = {memory.get("content") for memory in relevant_memories[:5]}

    retrieval_hit: bool | None = None
    if expected_memory_ids:
        retrieval_hit = bool(expected_memory_ids & retrieved_ids)
    elif expected_memory_contents:
        retrieval_hit = bool(expected_memory_contents & retrieved_contents)

    expected_is_valid = case.get("expected_is_valid_chat_window")
    invalid_window_correct: bool | None = None
    if expected_is_valid is not None:
        invalid_window_correct = result.get("is_valid_chat_window") is expected_is_valid

    expected_switch = case.get("expected_user_id_change_detected")
    user_switch_correct: bool | None = None
    if expected_switch is not None:
        user_switch_correct = result.get("user_id_change_detected") is expected_switch

    expected_intent = case.get("expected_intent")
    intent_correct: bool | None = None
    if expected_intent:
        intent_correct = expected_intent in result.get("task_list", [])

    return {
        "id": case.get("id"),
        "total_seconds": total_seconds,
        "error": error,
        "status": result.get("status"),
        "task_list": result.get("task_list", []),
        "intent_correct": intent_correct,
        "retrieval_top5_hit": retrieval_hit,
        "invalid_window_correct": invalid_window_correct,
        "user_switch_correct": user_switch_correct,
        "reuse_cache": result.get("reuse_cache"),
        "sync_errors": result.get("sync_errors", []),
    }


def build_report(recorder: MetricsRecorder, case_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metrics": {
            "intent_classification_avg_seconds": summarize_duration(recorder, "llm.intent_classifier"),
            "vision_screenshot_parsing_avg_seconds": summarize_duration(recorder, "vision.parse_chat_screenshot"),
            "retrieval_avg_seconds": summarize_duration(recorder, "node.retrieve_and_build_cache"),
            "reply_advice_main_chain_avg_seconds": summarize_duration(recorder, "chain.reply_advice_total"),
            "cache_reuse_hit_rate": summarize_ratio(recorder.cache_decisions),
            "memory_retrieval_top5_hit_rate": summarize_case_bool(case_results, "retrieval_top5_hit"),
            "invalid_chat_window_accuracy": summarize_case_bool(case_results, "invalid_window_correct"),
            "user_id_switch_success_rate": summarize_case_bool(case_results, "user_switch_correct"),
            "dirty_sync_success_rate": summarize_ratio(recorder.sync_successes),
        },
        "case_results": case_results,
        "notes": [
            "Timing metrics are measured from actual code execution.",
            "Accuracy/hit-rate metrics are only computed when benchmark cases include expected labels.",
            "Use --mock only for script validation, not for real project metrics.",
        ],
    }


def summarize_duration(recorder: MetricsRecorder, name: str) -> dict[str, Any]:
    values = recorder.durations.get(name, [])
    if not values:
        return {"value": None, "n": 0, "status": "not_measured"}
    return {
        "value": round(statistics.mean(values), 4),
        "n": len(values),
        "unit": "seconds",
    }


def summarize_ratio(values: list[bool]) -> dict[str, Any]:
    if not values:
        return {"value": None, "n": 0, "status": "not_measured"}
    return {
        "value": round(sum(1 for value in values if value) / len(values), 4),
        "n": len(values),
    }


def summarize_case_bool(case_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [case[key] for case in case_results if case.get(key) is not None]
    if not values:
        return {"value": None, "n": 0, "status": "insufficient_labels"}
    return {
        "value": round(sum(1 for value in values if value) / len(values), 4),
        "n": len(values),
    }


if __name__ == "__main__":
    main()
