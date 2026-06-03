from __future__ import annotations

from pathlib import Path
import uuid

from app.agent.memory import MemoryStore
from app.agent.memory_curator import (
    DEFAULT_AUTO_MEMORY_TRIGGER_TURNS,
    MemoryCurationState,
    MemoryCurator,
    _entries_for_model,
)
from app.storage.chat_history import ChatHistoryEntry


def test_memory_curator_writes_history_through_mem0() -> None:
    fake = FakeMem0()
    store = MemoryStore(
        base_dir=_runtime_root("memory_curator"),
        scope_id="sakura",
        memory_client=fake,
    )
    curator = MemoryCurator(store)

    result = curator.curate_entries([_entry("user", "以后默认中文和我说话")])

    assert result.created == 1
    assert result.processed_entries == 1
    assert fake.calls[0]["infer"] is True
    assert fake.calls[0]["user_id"] == "sakura"
    assert fake.calls[0]["messages"][0]["content"] == "以后默认中文和我说话"


def test_memory_curator_ignores_non_dialog_entries() -> None:
    fake = FakeMem0()
    store = MemoryStore(
        base_dir=_runtime_root("memory_curator_empty"),
        memory_client=fake,
    )
    curator = MemoryCurator(store)

    result = curator.curate_entries([_entry("system", "内部记录")])

    assert result.processed_entries == 1
    assert result.created == 0
    assert fake.calls == []


def test_memory_curation_state_waits_until_trigger_turns() -> None:
    state = MemoryCurationState(_runtime_json_path("memory_curation_state"))

    for _ in range(DEFAULT_AUTO_MEMORY_TRIGGER_TURNS - 1):
        state.increment_pending_turns()

    assert state.pending_turns() == DEFAULT_AUTO_MEMORY_TRIGGER_TURNS - 1
    assert state.pending_turns() < DEFAULT_AUTO_MEMORY_TRIGGER_TURNS

    state.increment_pending_turns()

    assert state.pending_turns() == DEFAULT_AUTO_MEMORY_TRIGGER_TURNS


def test_memory_entries_ignore_tone_and_portrait_metadata() -> None:
    entries = _entries_for_model(
        [
            ChatHistoryEntry(
                created_at="2026-05-31T12:00:00+08:00",
                role="assistant",
                content="覚えておくね。",
                translation="我会记住。",
                tone="中性",
                portrait="站立待机",
            )
        ]
    )

    assert entries == [
        {
            "created_at": "2026-05-31T12:00:00+08:00",
            "role": "assistant",
            "content": "覚えておくね。",
            "translation": "我会记住。",
        }
    ]


def test_mem0_openai_llm_retries_empty_structured_response() -> None:
    from mem0.llms.openai import OpenAILLM

    llm = OpenAILLM({"api_key": "test-key", "model": "test-model"})
    fake_client = FakeOpenAIClient()
    llm.client = fake_client

    response = llm.generate_response(
        messages=[{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_object"},
    )

    assert response == '{"memory":[]}'
    assert len(fake_client.chat.completions.calls) == 2
    assert "response_format" in fake_client.chat.completions.calls[0]
    assert "response_format" not in fake_client.chat.completions.calls[1]


def _entry(role: str, content: str) -> ChatHistoryEntry:
    return ChatHistoryEntry(
        created_at="2026-05-31T12:00:00+08:00",
        role=role,
        content=content,
    )


def _runtime_json_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "__pycache__"
        / "test_runtime"
        / name
        / uuid.uuid4().hex
        / f"{name}.json"
    )


def _runtime_root(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "__pycache__"
        / "test_runtime"
        / name
        / uuid.uuid4().hex
    )


class FakeMem0:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def add(self, messages, *, user_id=None, infer=True, metadata=None):  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "infer": infer,
                "metadata": metadata,
            }
        )
        return {
            "results": [
                {
                    "id": "mem1",
                    "memory": "主人希望默认用中文沟通",
                    "user_id": user_id,
                    "event": "ADD",
                }
            ]
        }


class FakeOpenAIClient:
    def __init__(self) -> None:
        completions = FakeChatCompletions()
        self.chat = type("FakeChat", (), {"completions": completions})()


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **params):  # type: ignore[no-untyped-def]
        self.calls.append(params)
        content = "" if "response_format" in params else '{"memory":[]}'
        return _fake_openai_response(content)


def _fake_openai_response(content: str):  # type: ignore[no-untyped-def]
    message = type("FakeMessage", (), {"content": content, "tool_calls": None})()
    choice = type("FakeChoice", (), {"message": message})()
    return type("FakeResponse", (), {"choices": [choice]})()
