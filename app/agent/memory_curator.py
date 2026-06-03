from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.memory import MemoryStore
from app.storage.chat_history import ChatHistoryEntry


DEFAULT_AUTO_MEMORY_TRIGGER_TURNS = 8
DEFAULT_AUTO_MEMORY_BACKFILL_LIMIT = 200


@dataclass(frozen=True)
class MemoryCurationSettings:
    enabled: bool = True
    trigger_turns: int = DEFAULT_AUTO_MEMORY_TRIGGER_TURNS
    backfill_limit: int = DEFAULT_AUTO_MEMORY_BACKFILL_LIMIT


@dataclass(frozen=True)
class MemoryCurationResult:
    created: int = 0
    updated: int = 0
    archived: int = 0
    ignored: int = 0
    processed_entries: int = 0
    returned: int = 0
    unclassified: int = 0
    event_counts: dict[str, int] | None = None

    def summary(self) -> str:
        return (
            f"整理完成：新增 {self.created} 条，更新 {self.updated} 条，"
            f"删除 {self.archived} 条，忽略 {self.ignored} 条。"
        )


class MemoryCurationState:
    """记录自动整理进度，避免重复处理历史。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def snapshot(self) -> dict[str, Any]:
        if not self.path.exists():
            return _normalize_state({})
        try:
            raw_data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _normalize_state({})
        return _normalize_state(raw_data)

    def pending_turns(self) -> int:
        return int(self.snapshot()["pending_turns"])

    def increment_pending_turns(self) -> int:
        state = self.snapshot()
        state["pending_turns"] = int(state["pending_turns"]) + 1
        self._save(state)
        return int(state["pending_turns"])

    def mark_processed(
        self,
        processed_history_count: int,
        *,
        consumed_turns: int = 0,
        backfill_completed: bool | None = None,
    ) -> None:
        state = self.snapshot()
        state["processed_history_count"] = max(0, processed_history_count)
        state["pending_turns"] = max(0, int(state["pending_turns"]) - max(0, consumed_turns))
        if backfill_completed is not None:
            state["backfill_completed"] = bool(backfill_completed)
        self._save(state)

    def mark_history_cleared(self) -> None:
        state = self.snapshot()
        state["processed_history_count"] = 0
        state["pending_turns"] = 0
        state["backfill_completed"] = True
        self._save(state)

    def unprocessed_entries(self, entries: list[ChatHistoryEntry]) -> list[ChatHistoryEntry]:
        state = self.snapshot()
        processed = int(state["processed_history_count"])
        if processed < 0 or processed > len(entries):
            processed = 0
        return entries[processed:]

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(_normalize_state(state), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class MemoryCurator:
    """调用 mem0 把聊天历史整理为长期记忆。"""

    def __init__(
        self,
        api_client_or_memory_store: Any,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.api_client = None if memory_store is None else api_client_or_memory_store
        self.memory_store = (
            api_client_or_memory_store
            if memory_store is None
            else memory_store
        )

    def curate_entries(self, entries: list[ChatHistoryEntry]) -> MemoryCurationResult:
        if not _entries_for_model(entries):
            return MemoryCurationResult(processed_entries=len(entries))

        counts = self.memory_store.add_history_entries(entries)
        return MemoryCurationResult(
            created=counts.created,
            updated=counts.updated,
            archived=counts.deleted,
            ignored=counts.ignored,
            processed_entries=len(entries),
            returned=counts.returned,
            unclassified=counts.unclassified,
            event_counts=dict(counts.event_counts),
        )


def _entries_for_model(entries: list[ChatHistoryEntry]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for entry in entries:
        if entry.role not in {"user", "assistant"}:
            continue
        content = entry.content.strip()
        if not content:
            continue
        result.append(
            {
                "created_at": entry.created_at,
                "role": entry.role,
                "content": content,
                "translation": entry.translation.strip(),
            }
        )
    return result


def _normalize_state(raw_data: Any) -> dict[str, Any]:
    data = raw_data if isinstance(raw_data, dict) else {}
    return {
        "processed_history_count": max(0, _int_value(data.get("processed_history_count"), default=0)),
        "pending_turns": max(0, _int_value(data.get("pending_turns"), default=0)),
        "backfill_completed": bool(data.get("backfill_completed", False)),
    }


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
