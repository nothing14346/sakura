from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from app.agent import AgentEvent, AgentResult, AgentRuntime, PendingToolAction
from app.debug_log import debug_log, summarize_messages


class ChatWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        messages: list[dict[str, Any]] | None = None,
        confirmed_action: PendingToolAction | None = None,
        cancelled_action: PendingToolAction | None = None,
    ) -> None:
        super().__init__()
        self.agent_runtime = agent_runtime
        self.messages = messages or []
        self.confirmed_action = confirmed_action
        self.cancelled_action = cancelled_action

    @Slot()
    def run(self) -> None:
        started_at = time.perf_counter()
        try:
            if self.confirmed_action is not None:
                debug_log("ChatWorker", "开始处理已确认动作", self.confirmed_action.to_dict())
                result: AgentResult = self.agent_runtime.handle_confirmed_action(self.confirmed_action)
            elif self.cancelled_action is not None:
                debug_log("ChatWorker", "开始处理已取消动作", self.cancelled_action.to_dict())
                result = self.agent_runtime.handle_cancelled_action(self.cancelled_action)
            else:
                debug_log(
                    "ChatWorker",
                    "开始处理用户消息",
                    {
                        "message_count": len(self.messages),
                        "messages": summarize_messages(self.messages),
                    },
                )
                result = self.agent_runtime.handle_user_message(self.messages)
        except Exception as exc:  # UI 边界统一转成可读错误。
            debug_log(
                "ChatWorker",
                "处理失败",
                {
                    "error": str(exc),
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                },
            )
            self.failed.emit(str(exc))
            return
        debug_log(
            "ChatWorker",
            "处理完成",
            {
                "segments": len(result.reply.segments),
                "actions": [action.type for action in result.actions],
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            },
        )
        self.finished.emit(result)


class EventWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, agent_runtime: AgentRuntime, event: AgentEvent) -> None:
        super().__init__()
        self.agent_runtime = agent_runtime
        # 避免覆盖 QObject.event() 虚函数名；PySide 在 moveToThread 时会访问该方法。
        self.agent_event = event

    @Slot()
    def run(self) -> None:
        started_at = time.perf_counter()
        try:
            debug_log(
                "EventWorker",
                "开始处理主动事件",
                {
                    "type": self.agent_event.type,
                    "payload": self.agent_event.payload,
                },
            )
            result = self.agent_runtime.handle_event(self.agent_event)
        except Exception as exc:  # 主动事件同样在 UI 边界转成可读错误。
            debug_log(
                "EventWorker",
                "处理失败",
                {
                    "error": str(exc),
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                },
            )
            self.failed.emit(str(exc))
            return
        debug_log(
            "EventWorker",
            "处理完成",
            {
                "segments": len(result.reply.segments),
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            },
        )
        self.finished.emit(result)
