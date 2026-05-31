from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from app.agent.memory import MemoryStore
from app.agent.memory_curator import (
    DEFAULT_AUTO_MEMORY_TRIGGER_TURNS,
    MemoryCurationState,
    MemoryCurator,
    _entries_for_model,
)
from app.chat_history import ChatHistoryEntry


def test_memory_curator_merges_duplicate_created_memories() -> None:
    class Client:
        def complete_raw(self, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            return json.dumps(
                {
                    "operations": [
                        {
                            "action": "create",
                            "category": "preference",
                            "content": "主人希望默认用中文沟通",
                            "importance": 0.8,
                            "confidence": 0.9,
                        },
                        {
                            "action": "create",
                            "category": "preference",
                            "content": "主人希望默认用中文沟通",
                            "importance": 0.7,
                            "confidence": 0.85,
                        },
                    ]
                },
                ensure_ascii=False,
            )

    store = MemoryStore(_runtime_json_path("memory_curator_merge"))
    curator = MemoryCurator(Client(), store)  # type: ignore[arg-type]

    result = curator.curate_entries([_entry("user", "以后默认中文和我说话")])

    memories = store.snapshot()["memories"]
    assert result.created == 1
    assert result.updated == 1
    assert len(memories) == 1
    assert memories[0]["seen_count"] == 2


def test_memory_curator_invalid_json_does_not_write_memory() -> None:
    class Client:
        def complete_raw(self, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            return "不是 JSON"

    store = MemoryStore(_runtime_json_path("memory_curator_invalid"))
    curator = MemoryCurator(Client(), store)  # type: ignore[arg-type]

    with pytest.raises(json.JSONDecodeError):
        curator.curate_entries([_entry("user", "记住一个重要偏好")])

    assert store.snapshot()["memories"] == []


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


def _entry(role: str, content: str) -> ChatHistoryEntry:
    return ChatHistoryEntry(
        created_at="2026-05-31T12:00:00+08:00",
        role=role,
        content=content,
    )


def _runtime_json_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "__pycache__"
        / "test_runtime"
        / name
        / uuid.uuid4().hex
        / f"{name}.json"
    )
