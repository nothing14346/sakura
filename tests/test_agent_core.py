from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import uuid

import pytest

from app.agent.actions import PendingToolAction
from app.agent.builtin_tools import create_builtin_tool_registry
from app.agent.memory import MemoryStore
from app.agent.reminders import ReminderStore
from app.agent.runtime import AgentRuntime, SCREEN_OBSERVATION_REQUEST_ACTION, _build_tool_results_message
from app.agent.tool_registry import Tool, ToolExecutionResult, ToolRegistry
from app.api_client import ApiRequestError, is_vision_unsupported_error, messages_contain_image
from app.screen_observation import (
    SCREEN_OBSERVATION_HISTORY_MARKER,
    ScreenObservation,
    append_observation_marker,
    build_screen_observation_user_message,
    should_observe_screen,
)


def test_add_reminder_delay_seconds_generates_future_time() -> None:
    store = ReminderStore(_runtime_json_path("reminders"))
    before = datetime.now().astimezone()

    result = store.add_reminder({"text": "喝水", "delay_seconds": 30})

    trigger_at = datetime.fromisoformat(result["reminder"]["trigger_at"])
    after = datetime.now().astimezone()
    assert before + timedelta(seconds=25) <= trigger_at <= after + timedelta(seconds=35)


def test_add_reminder_delay_minutes_generates_future_time() -> None:
    store = ReminderStore(_runtime_json_path("reminders"))
    before = datetime.now().astimezone()

    result = store.add_reminder({"text": "休息", "delay_minutes": 2})

    trigger_at = datetime.fromisoformat(result["reminder"]["trigger_at"])
    after = datetime.now().astimezone()
    assert before + timedelta(seconds=115) <= trigger_at <= after + timedelta(seconds=125)


def test_add_reminder_rejects_past_trigger_at() -> None:
    store = ReminderStore(_runtime_json_path("reminders"))
    past = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat(timespec="seconds")

    with pytest.raises(ValueError, match="提醒时间必须晚于当前时间"):
        store.add_reminder({"text": "过期提醒", "trigger_at": past})


def test_due_reminders_and_mark_completed() -> None:
    store = ReminderStore(_runtime_json_path("reminders"))
    now = datetime.now().astimezone()
    due = store.add_reminder({"text": "到点", "delay_seconds": 1})["reminder"]
    future = store.add_reminder({"text": "稍后", "delay_minutes": 5})["reminder"]

    due["trigger_at"] = (now - timedelta(seconds=1)).isoformat(timespec="seconds")
    future["trigger_at"] = (now + timedelta(minutes=5)).isoformat(timespec="seconds")
    store._save({"reminders": [due, future]})

    due_reminders = store.due_reminders(now)
    assert [reminder["id"] for reminder in due_reminders] == [due["id"]]

    store.mark_completed(due["id"])

    assert store.due_reminders(now) == []


def test_memory_propose_update_only_creates_pending_record() -> None:
    store = MemoryStore(_runtime_json_path("memory"))

    result = store.propose_memory_update(
        {
            "category": "preference",
            "content": "主人喜欢中文回复",
            "reason": "长期偏好",
        }
    )

    snapshot = store.snapshot()
    assert snapshot["memories"] == []
    assert snapshot["pending_updates"] == [result["pending_update"]]


def test_memory_confirm_update_moves_pending_to_memories() -> None:
    store = MemoryStore(_runtime_json_path("memory"))
    pending = store.propose_memory_update(
        {
            "category": "project",
            "content": "Sakura 正在稳定 Agent 内核",
        }
    )["pending_update"]

    result = store.confirm_memory_update({"id": pending["id"]})

    snapshot = store.snapshot()
    assert snapshot["pending_updates"] == []
    assert snapshot["memories"] == [result["memory"]]


def test_tool_registry_requires_confirmation_returns_pending_action() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="open_url",
                description="打开网页",
                handler=lambda _arguments: {"opened": True},
                requires_confirmation=True,
            )
        ]
    )

    result = registry.prepare_or_execute(
        "open_url",
        {"url": "https://example.com"},
        "用户要求打开网页",
    )

    assert isinstance(result, PendingToolAction)
    assert result.tool_name == "open_url"
    assert result.arguments == {"url": "https://example.com"}


def test_tool_registry_free_access_executes_normal_confirmation_tool() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="open_url",
                description="打开网页",
                handler=lambda _arguments: {"opened": True},
                requires_confirmation=True,
            )
        ]
    )
    registry.set_free_access_enabled(True)

    result = registry.prepare_or_execute("open_url", {"url": "https://example.com"})

    assert not isinstance(result, PendingToolAction)
    assert result.success
    assert result.content == {"opened": True}


def test_tool_registry_free_access_keeps_file_delete_confirmation() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="delete_file",
                description="删除本地文件",
                handler=lambda _arguments: {"deleted": True},
                requires_confirmation=True,
                confirmation_risk="delete_file",
            )
        ]
    )
    registry.set_free_access_enabled(True)

    result = registry.prepare_or_execute("delete_file", {"path": "a.txt"})

    assert isinstance(result, PendingToolAction)
    assert result.tool_name == "delete_file"


def test_builtin_registry_includes_browser_tools() -> None:
    registry = create_builtin_tool_registry(Path(__file__).resolve().parents[1])

    names = {tool["name"] for tool in registry.describe_tools()}

    assert {
        "browser_open_url",
        "browser_get_content",
        "browser_scroll",
        "browser_click",
        "browser_get_state",
    }.issubset(names)


def test_browser_open_url_rejects_non_http_url() -> None:
    registry = create_builtin_tool_registry(Path(__file__).resolve().parents[1])

    result = registry.execute("browser_open_url", {"url": "file:///C:/secret.txt"})

    assert not result.success
    assert "URL 只支持" in result.error


def test_browser_confirmation_tools_return_pending_actions() -> None:
    registry = create_builtin_tool_registry(
        Path(__file__).resolve().parents[1],
        browser_executor=_FakeBrowserExecutor(),
    )

    open_result = registry.prepare_or_execute("browser_open_url", {"url": "https://example.com"})
    scroll_result = registry.prepare_or_execute("browser_scroll", {"direction": "down"})
    click_result = registry.prepare_or_execute("browser_click", {"selector": "button"})

    assert isinstance(open_result, PendingToolAction)
    assert isinstance(scroll_result, PendingToolAction)
    assert isinstance(click_result, PendingToolAction)


def test_browser_confirmation_tools_obey_free_access() -> None:
    registry = create_builtin_tool_registry(
        Path(__file__).resolve().parents[1],
        browser_executor=_FakeBrowserExecutor(),
    )
    registry.set_free_access_enabled(True)

    result = registry.prepare_or_execute("browser_scroll", {"direction": "down", "amount": 1200})

    assert not isinstance(result, PendingToolAction)
    assert result.success
    assert result.content["scroll_y"] == 1200


def test_browser_get_content_truncates_text_and_links() -> None:
    registry = create_builtin_tool_registry(
        Path(__file__).resolve().parents[1],
        browser_executor=_FakeBrowserExecutor(),
    )

    result = registry.execute("browser_get_content", {"max_chars": 5})

    assert result.success
    assert result.content["text"] == "abcde"
    assert len(result.content["links"]) == 20


def test_browser_tools_validate_arguments() -> None:
    registry = create_builtin_tool_registry(
        Path(__file__).resolve().parents[1],
        browser_executor=_FakeBrowserExecutor(),
    )

    bad_direction = registry.execute("browser_scroll", {"direction": "sideways"})
    bad_selector = registry.execute("browser_click", {"selector": ""})

    assert not bad_direction.success
    assert "direction 只支持" in bad_direction.error
    assert not bad_selector.success
    assert "缺少必填参数" in bad_selector.error


def test_browser_screenshot_fallback_is_attached_as_image_url() -> None:
    result = ToolExecutionResult(
        tool_name="browser_get_content",
        success=True,
        content={
            "url": "https://example.com",
            "title": "Canvas Page",
            "text": "",
            "screenshot_data_url": "data:image/jpeg;base64,abc123",
            "screenshot_fallback": True,
        },
    )

    message = _build_tool_results_message([result], include_images=True)

    content = message["content"]
    assert isinstance(content, list)
    assert content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,abc123",
            "detail": "low",
        },
    }
    assert "screenshot_data_url" not in content[0]["text"]
    assert "screenshot_attached" in content[0]["text"]


def test_browser_screenshot_fallback_is_not_attached_without_vision() -> None:
    result = ToolExecutionResult(
        tool_name="browser_get_content",
        success=True,
        content={
            "url": "https://example.com",
            "title": "Canvas Page",
            "text": "",
            "screenshot_data_url": "data:image/jpeg;base64,abc123",
        },
    )

    message = _build_tool_results_message([result], include_images=False)

    assert isinstance(message["content"], str)
    assert "screenshot_data_url" not in message["content"]
    assert "screenshot_attached" in message["content"]


def test_model_vision_enabled_allows_model_to_request_screen_observation() -> None:
    class ScreenRequestClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return (
                '{"reply":{"segments":[{"ja":"見るね。","zh":"我看看。","tone":"提醒"}]},'
                '"tool_calls":[{"name":"observe_screen","arguments":{},"reason":"需要当前画面"}]}'
            )

    client = ScreenRequestClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry(),
    )
    runtime.set_model_vision_enabled(True)

    result = runtime.handle_user_message([{"role": "user", "content": "这个界面哪里不对"}])

    assert "observe_screen" in client.prompts[0]
    assert result.actions
    assert result.actions[0].type == SCREEN_OBSERVATION_REQUEST_ACTION


def test_screen_observation_message_uses_openai_image_url_format() -> None:
    observation = ScreenObservation(
        data_url="data:image/jpeg;base64,abc123",
        width=1280,
        height=720,
        captured_at="2026-05-29T20:00:00+08:00",
        screen_name="DISPLAY1",
    )

    message = build_screen_observation_user_message("帮我看这个", observation)

    assert message["role"] == "user"
    content = message["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,abc123",
            "detail": "low",
        },
    }
    assert messages_contain_image([message])


def test_screen_observation_history_marker_does_not_store_image_data() -> None:
    observation = ScreenObservation(
        data_url="data:image/jpeg;base64,secret",
        width=800,
        height=600,
        captured_at="2026-05-29T20:00:00+08:00",
        screen_name="DISPLAY1",
    )

    history_text = append_observation_marker("看看屏幕", observation)

    assert SCREEN_OBSERVATION_HISTORY_MARKER in history_text
    assert "data:image/jpeg;base64" not in history_text
    assert "secret" not in history_text


def test_screen_observation_trigger_requires_explicit_text() -> None:
    assert should_observe_screen("帮我看这个界面哪里不对")
    assert should_observe_screen("看看当前画面")
    assert not should_observe_screen("今天聊点什么")


def test_vision_unsupported_error_gets_local_fallback_reply() -> None:
    class VisionUnsupportedClient:
        def complete_raw(self, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            raise ApiRequestError("model does not support image_url content")

    observation = ScreenObservation(
        data_url="data:image/jpeg;base64,abc123",
        width=1280,
        height=720,
        captured_at="2026-05-29T20:00:00+08:00",
        screen_name="DISPLAY1",
    )
    runtime = AgentRuntime(
        api_client=VisionUnsupportedClient(),  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry(),
    )

    result = runtime.handle_user_message([build_screen_observation_user_message("看看屏幕", observation)])

    assert "不支持图片输入" in result.reply.translation
    assert not result.actions


def test_plain_text_messages_do_not_contain_image() -> None:
    assert not messages_contain_image([{"role": "user", "content": "普通聊天"}])
    assert is_vision_unsupported_error("This model does not support image input")


def _runtime_json_path(name: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "__pycache__" / "test_runtime" / uuid.uuid4().hex
    return root / f"{name}.json"


class _FakeBrowserExecutor:
    def execute_browser_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name == "browser_open_url":
            return {
                "url": arguments["url"],
                "title": "Example Domain",
                "opened": True,
                "loaded": True,
            }
        if name == "browser_get_content":
            return {
                "url": "https://example.com",
                "title": "Example Domain",
                "text": "abcdefghijklmnopqrstuvwxyz",
                "links": [
                    {"text": f"Link {index}", "href": f"https://example.com/{index}"}
                    for index in range(25)
                ],
            }
        if name == "browser_scroll":
            amount = int(arguments.get("amount", 800))
            direction = str(arguments.get("direction", "down"))
            return {
                "url": "https://example.com",
                "title": "Example Domain",
                "scroll_y": -amount if direction == "up" else amount,
            }
        if name == "browser_click":
            return {
                "ok": True,
                "url": "https://example.com",
                "title": "Example Domain",
                "selector": arguments["selector"],
            }
        if name == "browser_get_state":
            return {
                "url": "https://example.com",
                "title": "Example Domain",
                "scroll_y": 0,
                "loading": False,
            }
        raise ValueError(name)
