from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEvent, QThread, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.agent import AgentResult, AgentRuntime, MemoryStore, ReminderStore, create_builtin_tool_registry
from app.api_client import OpenAICompatibleClient
from app.chat_worker import ChatWorker
from app.debug_log import debug_log, summarize_messages
from app.tts import TTSProvider


class ChatWindow(QWidget):
    assistant_replied = Signal(str)

    def __init__(
        self,
        api_client: OpenAICompatibleClient,
        system_prompt: str,
        tts_provider: TTSProvider,
        base_dir: Path | None = None,
    ) -> None:
        super().__init__()
        base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.api_client = api_client
        self.system_prompt = system_prompt
        self.memory_store = MemoryStore(base_dir / "data" / "memory.json")
        self.reminder_store = ReminderStore(base_dir / "data" / "reminders.json")
        self.agent_runtime = AgentRuntime(
            api_client=api_client,
            system_prompt=system_prompt,
            tools=create_builtin_tool_registry(base_dir, self.memory_store, self.reminder_store),
            memory=self.memory_store,
        )
        self.tts_provider = tts_provider
        self.messages: list[dict[str, str]] = []
        self.worker_thread: QThread | None = None
        self.worker: ChatWorker | None = None
        self.interaction_sequence = 0
        self.active_interaction_id = ""
        self.active_interaction_started_at: float | None = None
        self.active_interaction_last_at: float | None = None

        self.setWindowTitle("夜乃桜")
        self.resize(520, 640)
        self.setStyleSheet(
            """
            QWidget {
                background: #f4fbfd;
                color: #24343a;
                font-family: "Microsoft YaHei", "Yu Gothic UI", sans-serif;
                font-size: 14px;
            }
            QTextBrowser {
                background: rgba(226, 246, 250, 0.86);
                border: 1px solid rgba(120, 176, 188, 0.55);
                border-radius: 12px;
                padding: 14px;
                selection-background-color: #7cc8d7;
            }
            QLineEdit {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(120, 176, 188, 0.65);
                border-radius: 18px;
                padding: 9px 14px;
            }
            QPushButton {
                background: #72c7d6;
                border: none;
                border-radius: 18px;
                color: white;
                min-width: 72px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #5eb7c8;
            }
            QPushButton:disabled {
                background: #a9c7ce;
            }
            """
        )

        self.history_view = QTextBrowser()
        self.history_view.setOpenExternalLinks(True)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入要对桜说的话...")
        self.input_edit.installEventFilter(self)
        self.input_edit.returnPressed.connect(self._handle_return_pressed)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self._handle_send_button_clicked)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_edit, 1)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addWidget(self.history_view, 1)
        layout.addLayout(input_layout)
        self.setLayout(layout)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self.input_edit and event.type() == QEvent.Type.KeyPress:
            key_event = event if isinstance(event, QKeyEvent) else None
            debug_log(
                "Input",
                "聊天窗口输入框按键事件",
                {
                    "key": int(key_event.key()) if key_event is not None else "",
                    "text": key_event.text() if key_event is not None else "",
                    "input_chars": len(self.input_edit.text()),
                    "worker_busy": self.worker_thread is not None,
                },
            )
        return super().eventFilter(watched, event)

    def _begin_interaction(self, source: str) -> None:
        self.interaction_sequence += 1
        now = time.perf_counter()
        self.active_interaction_id = f"chat-window-{self.interaction_sequence}"
        self.active_interaction_started_at = now
        self.active_interaction_last_at = now
        debug_log(
            "Latency",
            "聊天窗口输入事件开始",
            {
                "interaction_id": self.active_interaction_id,
                "source": source,
                "input_chars": len(self.input_edit.text()),
                "worker_busy": self.worker_thread is not None,
            },
        )

    def _log_interaction_stage(self, stage: str, data: dict[str, object] | None = None) -> None:
        if not self.active_interaction_id or self.active_interaction_started_at is None:
            return
        now = time.perf_counter()
        previous = self.active_interaction_last_at or self.active_interaction_started_at
        self.active_interaction_last_at = now
        payload: dict[str, object] = {
            "interaction_id": self.active_interaction_id,
            "stage": stage,
            "elapsed_ms": int((now - self.active_interaction_started_at) * 1000),
            "delta_ms": int((now - previous) * 1000),
        }
        if data:
            payload.update(data)
        debug_log("Latency", "聊天窗口交互阶段", payload)

    def _end_interaction(self, outcome: str) -> None:
        self._log_interaction_stage("interaction_finished", {"outcome": outcome})
        self.active_interaction_id = ""
        self.active_interaction_started_at = None
        self.active_interaction_last_at = None

    @Slot()
    def _handle_return_pressed(self) -> None:
        self._begin_interaction("return_pressed")
        self.send_message("return_pressed")

    @Slot()
    def _handle_send_button_clicked(self) -> None:
        self._begin_interaction("send_button_clicked")
        self.send_message("send_button_clicked")

    @Slot()
    def send_message(self, source: str = "direct_call") -> None:
        text = self.input_edit.text().strip()
        if not self.active_interaction_id:
            self._begin_interaction(source)
        self._log_interaction_stage(
            "send_message_enter",
            {"source": source, "text": text, "worker_busy": self.worker_thread is not None},
        )
        if not text or self.worker_thread is not None:
            self._log_interaction_stage(
                "send_message_ignored",
                {"has_text": bool(text), "worker_busy": self.worker_thread is not None},
            )
            self._end_interaction("ignored")
            return

        self.input_edit.clear()
        self._log_interaction_stage("input_cleared")
        self._append_message("你", text)
        next_messages = [*self.messages, {"role": "user", "content": text}]
        self.messages = next_messages
        self._set_busy(True)
        self._log_interaction_stage(
            "chat_worker_start",
            {
                "message_count": len(next_messages),
                "messages": summarize_messages(next_messages),
            },
        )

        self.worker_thread = QThread(self)
        self.worker = ChatWorker(self.agent_runtime, next_messages)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_reply)
        self.worker.failed.connect(self._handle_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()
        self._log_interaction_stage("chat_worker_started")

    @Slot(object)
    def _handle_reply(self, result: AgentResult) -> None:
        self._log_interaction_stage(
            "agent_result_received",
            {
                "segments": len(result.reply.segments),
                "actions": [action.type for action in result.actions],
            },
        )
        reply = result.reply
        reply_text = reply.text
        self.messages.append({"role": "assistant", "content": reply_text})
        self._append_message("桜", reply_text)
        self.assistant_replied.emit(reply_text)
        self.tts_provider.speak(reply_text, reply.tone)
        self._end_interaction("reply_received")

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self._log_interaction_stage("worker_error", {"message": message})
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()
        self._append_message("错误", message)
        QMessageBox.warning(self, "请求失败", message)
        self._end_interaction("error")

    @Slot()
    def _cleanup_worker(self) -> None:
        self._log_interaction_stage("cleanup_worker_enter")
        if self.worker is not None:
            self.worker.deleteLater()
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None
        self._set_busy(False)
        self._log_interaction_stage("ui_busy_disabled")

    def _set_busy(self, busy: bool) -> None:
        self.input_edit.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.send_button.setText("等待中..." if busy else "发送")
        self._log_interaction_stage("set_busy", {"busy": busy})

    def _append_message(self, sender: str, message: str) -> None:
        safe_sender = _escape_html(sender)
        safe_message = _escape_html(message).replace("\n", "<br>")
        self.history_view.append(f"<b>{safe_sender}：</b><br>{safe_message}<br>")


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
