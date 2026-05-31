from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import uuid

import pytest

from app.agent.actions import AgentEvent, PendingToolAction
from app.agent.builtin_tools import create_builtin_tool_registry
from app.agent.memory import MemoryStore
from app.agent.mcp.bridge import MCPToolSpec
from app.agent.mcp.config import load_mcp_config
from app.agent.mcp.provider import MCPToolProvider, register_mcp_tools_from_config
from app.agent.reminders import ReminderStore
from app.agent.runtime import AgentRuntime, _build_event_messages, _build_tool_results_message
from app.agent.runtime import _redact_event_for_model, _redact_tool_result_for_model
from app.agent.screen_tools import (
    OBSERVE_SCREEN_TOOL_NAME,
    SCREEN_OBSERVATION_DISABLED_ERROR,
    SCREEN_OBSERVATION_REQUEST_ACTION,
    create_screen_observation_tool,
)
from app.agent.tool_registry import Tool, ToolExecutionResult, ToolRegistry
from app.api_client import ApiRequestError, is_vision_unsupported_error, messages_contain_image
from app.context_trimming import MAX_MODEL_CONTEXT_MESSAGES, trim_messages_for_model
from app.proactive_care import (
    PROACTIVE_CHECK_INTERVAL_MINUTES_KEY,
    PROACTIVE_COOLDOWN_MINUTES_KEY,
    PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY,
    PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY,
    ProactiveCareSettings,
)
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


def test_proactive_care_settings_default_to_disabled() -> None:
    settings = ProactiveCareSettings.load(_runtime_root_path("proactive_defaults") / ".env")

    assert not settings.enabled
    assert not settings.screen_context_enabled
    assert settings.check_interval_minutes == 20
    assert settings.cooldown_minutes == 10
    assert settings.screen_context_batch_limit == 6


def test_proactive_care_settings_clamp_intervals() -> None:
    env_path = _runtime_root_path("proactive_interval") / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "\n".join(
            [
                "PROACTIVE_CARE_ENABLED=true",
                f"{PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY}=true",
                f"{PROACTIVE_CHECK_INTERVAL_MINUTES_KEY}=999",
                f"{PROACTIVE_COOLDOWN_MINUTES_KEY}=999",
                f"{PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY}=999",
            ]
        ),
        encoding="utf-8",
    )

    settings = ProactiveCareSettings.load(env_path)

    assert settings.enabled
    assert settings.screen_context_enabled
    assert settings.check_interval_minutes == 120
    assert settings.cooldown_minutes == 120
    assert settings.screen_context_batch_limit == 20


def test_proactive_care_settings_min_intervals_are_one_minute() -> None:
    env_path = _runtime_root_path("proactive_min_interval") / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "\n".join(
            [
                f"{PROACTIVE_CHECK_INTERVAL_MINUTES_KEY}=0",
                f"{PROACTIVE_COOLDOWN_MINUTES_KEY}=0",
                f"{PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY}=0",
            ]
        ),
        encoding="utf-8",
    )

    settings = ProactiveCareSettings.load(env_path)

    assert settings.check_interval_minutes == 1
    assert settings.cooldown_minutes == 1
    assert settings.screen_context_batch_limit == 1


def test_proactive_care_settings_invalid_cooldown_uses_default() -> None:
    env_path = _runtime_root_path("proactive_invalid_cooldown") / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f"{PROACTIVE_COOLDOWN_MINUTES_KEY}=soon",
        encoding="utf-8",
    )

    settings = ProactiveCareSettings.load(env_path)

    assert settings.cooldown_minutes == 10


def test_proactive_care_settings_invalid_batch_limit_uses_default() -> None:
    env_path = _runtime_root_path("proactive_invalid_batch_limit") / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f"{PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY}=many",
        encoding="utf-8",
    )

    settings = ProactiveCareSettings.load(env_path)

    assert settings.screen_context_batch_limit == 6


def test_proactive_care_settings_save_writes_cooldown() -> None:
    env_path = _runtime_root_path("proactive_save_cooldown") / ".env"

    ProactiveCareSettings(
        enabled=True,
        screen_context_enabled=True,
        check_interval_minutes=3,
        cooldown_minutes=7,
        screen_context_batch_limit=4,
    ).save(env_path)

    text = env_path.read_text(encoding="utf-8")
    assert "PROACTIVE_CARE_ENABLED=true" in text
    assert f"{PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY}=true" in text
    assert f"{PROACTIVE_CHECK_INTERVAL_MINUTES_KEY}=3" in text
    assert f"{PROACTIVE_COOLDOWN_MINUTES_KEY}=7" in text
    assert f"{PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY}=4" in text


def test_proactive_care_settings_save_syncs_legacy_enabled_key() -> None:
    env_path = _runtime_root_path("proactive_save_sync_enabled") / ".env"

    ProactiveCareSettings(
        enabled=True,
        screen_context_enabled=False,
        check_interval_minutes=3,
        cooldown_minutes=7,
    ).save(env_path)

    text = env_path.read_text(encoding="utf-8")
    assert "PROACTIVE_CARE_ENABLED=false" in text
    assert f"{PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY}=false" in text


def test_proactive_care_screen_context_flag_controls_active_care() -> None:
    enabled_settings = ProactiveCareSettings(
        enabled=True,
        screen_context_enabled=True,
        check_interval_minutes=20,
    )
    care_disabled_settings = ProactiveCareSettings(
        enabled=False,
        screen_context_enabled=True,
        check_interval_minutes=20,
    )
    screen_disabled_settings = ProactiveCareSettings(
        enabled=True,
        screen_context_enabled=False,
        check_interval_minutes=20,
    )

    assert enabled_settings.allows_screen_context()
    assert care_disabled_settings.allows_screen_context()
    assert not screen_disabled_settings.allows_screen_context()


def test_memory_store_reads_legacy_memory_without_pending_updates() -> None:
    store = MemoryStore(_runtime_json_path("memory"))
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "abc123",
                        "category": "fact",
                        "content": "主人是 Sakura 的开发者",
                        "created_at": "2026-05-30T17:13:44+08:00",
                        "updated_at": "2026-05-30T17:14:16+08:00",
                    }
                ],
                "pending_updates": [
                    {
                        "id": "pending",
                        "category": "preference",
                        "content": "旧候选记忆会被忽略",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshot = store.snapshot()

    assert len(snapshot["memories"]) == 1
    assert snapshot["memories"][0]["source"] == "legacy"
    assert "pending_updates" not in snapshot


def test_memory_store_create_update_search_and_delete() -> None:
    store = MemoryStore(_runtime_json_path("memory"))
    created = store.create_memory(
        {
            "category": "project",
            "content": "Sakura 正在实现自动记忆整理",
            "importance": 0.8,
            "confidence": 0.9,
        }
    )["memory"]

    updated = store.update_memory(
        {
            "id": created["id"],
            "content": "Sakura 正在实现自动记忆整理和管理页",
            "importance": 0.95,
        }
    )["memory"]

    search_result = store.search_memory({"keyword": "管理页", "category": "project"})
    assert search_result["memories"] == [updated]

    removed = store.delete_memory({"id": created["id"]})["memory"]
    assert removed["content"] == "Sakura 正在实现自动记忆整理和管理页"
    assert store.snapshot()["memories"] == []


def test_memory_store_rejects_sensitive_auto_memory() -> None:
    store = MemoryStore(_runtime_json_path("memory"))

    result = store.upsert_auto_memory(
        {
            "category": "fact",
            "content": "API_KEY=sk-1234567890abcdef1234567890abcdef",
        }
    )

    assert result["reason"] == "包含敏感凭据或隐私字段"
    assert store.snapshot()["memories"] == []


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
    registry.set_free_access_enabled(False)

    result = registry.prepare_or_execute(
        "open_url",
        {"url": "https://example.com"},
        "用户要求打开网页",
    )

    assert isinstance(result, PendingToolAction)
    assert result.tool_name == "open_url"
    assert result.arguments == {"url": "https://example.com"}


def test_tool_registry_free_access_is_enabled_by_default() -> None:
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


def test_tool_registry_free_access_keeps_high_risk_confirmation() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="run_external_action",
                description="执行高风险动作",
                handler=lambda _arguments: {"done": True},
                requires_confirmation=True,
                risk="high",
            )
        ]
    )
    registry.set_free_access_enabled(True)

    result = registry.prepare_or_execute("run_external_action", {"value": "x"})

    assert isinstance(result, PendingToolAction)
    assert result.tool_name == "run_external_action"


def test_tool_registry_describes_group_and_risk_metadata() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="search_memory",
                description="搜索记忆",
                group="memory",
                risk="low",
            )
        ]
    )

    description = registry.describe_tools()[0]

    assert description["group"] == "memory"
    assert description["risk"] == "low"


def test_tool_registry_filters_tools_by_capability() -> None:
    registry = ToolRegistry(
        [
            Tool(name="plain_tool", description="普通工具"),
            create_screen_observation_tool(),
        ]
    )

    visible_without_capability = {tool["name"] for tool in registry.describe_tools(set())}
    visible_with_capability = {
        tool["name"]
        for tool in registry.describe_tools({"screen_observation"})
    }

    assert OBSERVE_SCREEN_TOOL_NAME not in visible_without_capability
    assert OBSERVE_SCREEN_TOOL_NAME in visible_with_capability


def test_mcp_missing_config_does_not_register_tools() -> None:
    root = _runtime_root_path("mcp_missing")
    registry = ToolRegistry()

    provider = register_mcp_tools_from_config(root, registry)

    assert provider is None
    assert registry.describe_tools() == []


def test_mcp_disabled_config_does_not_register_tools() -> None:
    root = _runtime_root_path("mcp_disabled")
    config_dir = root / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "mcp.yaml").write_text(
        """
enabled: false
servers:
  demo:
    transport: stdio
    command: python
""".strip(),
        encoding="utf-8",
    )

    registry = ToolRegistry()
    provider = register_mcp_tools_from_config(
        root,
        registry,
        bridge_factory=_FakeMCPBridge,
    )

    assert provider is None
    assert registry.describe_tools() == []


def test_mcp_empty_servers_do_not_register_tools() -> None:
    root = _runtime_root_path("mcp_empty")
    config_dir = root / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "mcp.yaml").write_text("enabled: true\nservers: {}\n", encoding="utf-8")

    registry = ToolRegistry()
    provider = register_mcp_tools_from_config(root, registry)

    assert provider is None
    assert registry.describe_tools() == []


def test_mcp_invalid_config_does_not_break_registry() -> None:
    root = _runtime_root_path("mcp_invalid")
    config_dir = root / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "mcp.yaml").write_text("enabled: [not_bool]\n", encoding="utf-8")
    registry = ToolRegistry()

    provider = register_mcp_tools_from_config(root, registry)

    assert provider is None
    assert registry.describe_tools() == []


def test_mcp_config_parses_prefix_and_low_risk() -> None:
    config_path = _runtime_root_path("mcp_parse") / "mcp.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
enabled: true
default_call_timeout: 12
servers:
  demo:
    transport: sse
    url: http://127.0.0.1:8000/sse
    name_prefix: demo_
    risk: low
""".strip(),
        encoding="utf-8",
    )

    config = load_mcp_config(config_path)

    assert config.enabled
    assert config.default_call_timeout == 12
    assert config.servers[0].effective_name_prefix() == "demo_"
    assert config.servers[0].risk == "low"
    assert not config.servers[0].effective_requires_confirmation()


def test_mcp_config_parses_tool_filter_and_policies() -> None:
    config_path = _runtime_root_path("mcp_tool_policy_parse") / "mcp.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
enabled: true
servers:
  demo:
    transport: stdio
    command: python
    include_tools: ["Read*", "Danger", "DangerousRead"]
    exclude_tools: ["DangerousRead"]
    tool_policies:
      "Read*":
        risk: medium
        requires_confirmation: false
      Danger:
        risk: high
""".strip(),
        encoding="utf-8",
    )

    config = load_mcp_config(config_path)
    server = config.servers[0]

    assert server.allows_tool("ReadPage")
    assert server.allows_tool("Danger")
    assert not server.allows_tool("DangerousRead")
    assert server.effective_tool_risk("ReadPage") == "medium"
    assert not server.effective_tool_requires_confirmation("ReadPage")
    assert server.effective_tool_risk("Danger") == "high"
    assert server.effective_tool_requires_confirmation("Danger")


def test_mcp_provider_registers_prefixed_tools_and_schema() -> None:
    root = _runtime_root_path("mcp_register")
    config_dir = root / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "mcp.yaml").write_text(
        """
enabled: true
servers:
  demo:
    transport: stdio
    command: python
    name_prefix: demo_
""".strip(),
        encoding="utf-8",
    )
    registry = ToolRegistry()

    provider = register_mcp_tools_from_config(
        root,
        registry,
        bridge_factory=_FakeMCPBridge,
    )

    assert provider is not None
    tool = registry.get("demo_echo")
    assert tool is not None
    assert tool.parameters["properties"]["message"]["type"] == "string"
    assert tool.group == "mcp"
    assert tool.risk == "medium"
    assert tool.requires_confirmation
    provider.close()


def test_mcp_provider_filters_tools_and_applies_tool_policies() -> None:
    root = _runtime_root_path("mcp_tool_policy_register")
    registry = ToolRegistry()
    provider = MCPToolProvider(
        load_mcp_config(
            _write_mcp_config(
                root,
                """
name_prefix: demo_
risk: high
include_tools: ["read_state", "click_target"]
tool_policies:
  read_state:
    risk: medium
    requires_confirmation: false
""".strip(),
            )
        ),
        bridge_factory=_MultiToolMCPBridge,
    )

    provider.register_tools(registry)

    read_tool = registry.get("demo_read_state")
    click_tool = registry.get("demo_click_target")
    assert read_tool is not None
    assert click_tool is not None
    assert registry.get("demo_hidden_tool") is None
    assert read_tool.risk == "medium"
    assert not read_tool.requires_confirmation
    assert click_tool.risk == "high"
    assert click_tool.requires_confirmation
    provider.close()


def test_mcp_tool_call_uses_external_name() -> None:
    root = _runtime_root_path("mcp_call")
    registry = ToolRegistry()
    provider = MCPToolProvider(
        load_mcp_config(_write_mcp_config(root, "name_prefix: demo_\nrisk: low")),
        bridge_factory=_FakeMCPBridge,
    )
    provider.register_tools(registry)

    result = registry.execute("demo_echo", {"message": "hello"})

    assert result.success
    assert result.content["server_tool_name"] == "echo"
    assert result.content["arguments"] == {"message": "hello"}
    provider.close()


def test_mcp_tool_call_exception_returns_failed_result() -> None:
    root = _runtime_root_path("mcp_fail")
    registry = ToolRegistry()
    provider = MCPToolProvider(
        load_mcp_config(_write_mcp_config(root, "name_prefix: fail_\nrisk: low")),
        bridge_factory=_FailingMCPBridge,
    )
    provider.register_tools(registry)

    result = registry.execute("fail_echo", {"message": "hello"})

    assert not result.success
    assert "MCP 调用失败" in result.error
    provider.close()


def test_mcp_high_risk_still_requires_confirmation_with_free_access() -> None:
    root = _runtime_root_path("mcp_high_risk")
    registry = ToolRegistry()
    provider = MCPToolProvider(
        load_mcp_config(_write_mcp_config(root, "name_prefix: dangerous_\nrisk: high")),
        bridge_factory=_FakeMCPBridge,
    )
    provider.register_tools(registry)
    registry.set_free_access_enabled(True)

    result = registry.prepare_or_execute("dangerous_echo", {"message": "hello"})

    assert isinstance(result, PendingToolAction)
    assert result.tool_name == "dangerous_echo"
    provider.close()


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
    assert OBSERVE_SCREEN_TOOL_NAME in names


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
    registry.set_free_access_enabled(False)

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


def test_tool_result_for_model_truncates_large_content() -> None:
    result = ToolExecutionResult(
        tool_name="browser_get_content",
        success=True,
        content={
            "url": "https://example.com",
            "text": "x" * 7000,
        },
    )

    redacted = _redact_tool_result_for_model(result)

    content = redacted["content"]
    assert isinstance(content, dict)
    assert content["truncated"] is True
    assert content["original_chars"] > 6000
    assert "head" in content
    assert "tail" in content


def test_agent_runtime_can_continue_tool_loop_after_tool_results() -> None:
    class MultiStepClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.messages: list[list[dict[str, object]]] = []
            self.final_chat_called = False

        def complete_raw(self, system_prompt, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            self.messages.append(messages)
            call_index = len(self.prompts)
            if call_index == 1:
                return (
                    '{"reply":{"segments":[{"ja":"まず確認するね。","zh":"我先确认一下。","tone":"中性"}]},'
                    '"tool_calls":[{"name":"first_tool","arguments":{"value":"start"},"reason":"先取第一步结果"}]}'
                )
            if call_index == 2:
                assert "first_result" in str(messages)
                return (
                    '{"reply":{"segments":[{"ja":"次も見るね。","zh":"我再看下一步。","tone":"中性"}]},'
                    '"tool_calls":[{"name":"second_tool","arguments":{"value":"next"},"reason":"根据第一步结果继续"}]}'
                )
            assert "second_result" in str(messages)
            return (
                '{"reply":{"segments":[{"ja":"二つとも確認できたよ。","zh":"两步都确认好了。","tone":"中性"}]},'
                '"tool_calls":[]}'
            )

        def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.final_chat_called = True
            from app.chat_reply import parse_chat_reply

            return parse_chat_reply(
                '{"segments":[{"ja":"fallback","zh":"fallback","tone":"中性"}]}'
            )

    registry = ToolRegistry(
        [
            Tool(
                name="first_tool",
                description="第一步工具",
                handler=lambda arguments: {"first_result": arguments["value"]},
            ),
            Tool(
                name="second_tool",
                description="第二步工具",
                handler=lambda arguments: {"second_result": arguments["value"]},
            ),
        ]
    )
    client = MultiStepClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=registry,
    )
    progress_replies = []

    result = runtime.handle_user_message(
        [{"role": "user", "content": "连续处理这个任务"}],
        progress_callback=progress_replies.append,
    )

    assert result.reply.translation == "两步都确认好了。"
    assert [progress.reply.translation for progress in progress_replies] == [
        "我先确认一下。",
        "我再看下一步。",
    ]
    assert [action.payload["tool_name"] for action in result.actions] == ["first_tool", "second_tool"]
    assert len(client.prompts) == 3
    assert not client.final_chat_called
    assert "这是第 1 步" in client.prompts[0]
    assert "这是第 2 步" in client.prompts[1]


def test_agent_runtime_stops_tool_loop_at_turn_limit() -> None:
    class RepeatingToolClient:
        def __init__(self) -> None:
            self.raw_calls = 0
            self.chat_messages: list[dict[str, object]] = []

        def complete_raw(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.raw_calls += 1
            return (
                '{"reply":{"segments":[{"ja":"続けるね。","zh":"继续。","tone":"中性"}]},'
                '"tool_calls":[{"name":"echo_tool","arguments":{},"reason":"测试上限"},'
                '{"name":"echo_tool","arguments":{},"reason":"测试上限"},'
                '{"name":"echo_tool","arguments":{},"reason":"测试上限"}]}'
            )

        def chat(self, _system_prompt, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.chat_messages = messages
            from app.chat_reply import parse_chat_reply

            return parse_chat_reply(
                '{"segments":[{"ja":"上限で止めたよ。","zh":"已经按上限停止了。","tone":"中性"}]}'
            )

    registry = ToolRegistry(
        [
            Tool(
                name="echo_tool",
                description="回显工具",
                handler=lambda _arguments: {"ok": True},
            )
        ]
    )
    client = RepeatingToolClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=registry,
    )

    result = runtime.handle_user_message([{"role": "user", "content": "一直调用工具"}])

    assert result.reply.translation == "已经按上限停止了。"
    assert len([action for action in result.actions if action.payload["tool_name"] == "echo_tool"]) == 8
    assert any(action.payload["tool_name"] == "runtime" for action in result.actions)
    assert client.raw_calls == 3
    assert "已跳过" in str(client.chat_messages)


def test_agent_runtime_emits_progress_before_pending_confirmation() -> None:
    class ConfirmToolClient:
        def complete_raw(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return (
                '{"reply":{"segments":[{"ja":"開く前に確認するね。","zh":"打开前我先确认一下。","tone":"中性"}]},'
                '"tool_calls":[{"name":"open_tool","arguments":{"url":"https://example.com"},"reason":"需要打开网页"}]}'
            )

    registry = ToolRegistry(
        [
            Tool(
                name="open_tool",
                description="需要确认的工具",
                handler=lambda arguments: {"opened": arguments["url"]},
                requires_confirmation=True,
                risk="high",
            )
        ]
    )
    runtime = AgentRuntime(
        api_client=ConfirmToolClient(),  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=registry,
    )
    progress_replies = []

    result = runtime.handle_user_message(
        [{"role": "user", "content": "打开这个网页"}],
        progress_callback=progress_replies.append,
    )

    assert [progress.reply.translation for progress in progress_replies] == ["打开前我先确认一下。"]
    assert any(action.type == "pending_action" for action in result.actions)


def test_agent_runtime_extracts_planner_json_from_mixed_output() -> None:
    class MixedPlannerClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete_raw(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return (
                    "（先说一句自然话，然后错误地贴了工具规划）\n"
                    "`observe_screen` 获取上下文\n"
                    '{'
                    '"reply":{"segments":[{"ja":"少し確認するね。","zh":"我先确认一下。","tone":"中性"}]},'
                    '"tool_calls":[{"name":"echo_tool","arguments":{"value":"ok"},"reason":"补充上下文"}]'
                    "}"
                )
            return '{"reply":{"segments":[{"ja":"確認できたよ。","zh":"确认好了。","tone":"中性"}]},"tool_calls":[]}'

    registry = ToolRegistry(
        [
            Tool(
                name="echo_tool",
                description="回显工具",
                handler=lambda arguments: {"value": arguments["value"]},
            )
        ]
    )
    client = MixedPlannerClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=registry,
    )
    progress_replies = []

    result = runtime.handle_user_message(
        [{"role": "user", "content": "下午好"}],
        progress_callback=progress_replies.append,
    )

    assert [progress.reply.translation for progress in progress_replies] == ["我先确认一下。"]
    assert result.reply.translation == "确认好了。"
    assert "tool_calls" not in result.reply.text
    assert [action.payload["tool_name"] for action in result.actions] == ["echo_tool"]


def test_trim_messages_for_model_keeps_recent_messages_without_mutating_history() -> None:
    messages = [
        {"role": "user", "content": f"message {index}"}
        for index in range(MAX_MODEL_CONTEXT_MESSAGES + 5)
    ]

    trimmed = trim_messages_for_model(messages)

    assert len(trimmed) == MAX_MODEL_CONTEXT_MESSAGES
    assert trimmed[0]["content"] == "message 5"
    assert len(messages) == MAX_MODEL_CONTEXT_MESSAGES + 5


def test_autonomous_screen_observation_can_request_screen_without_explicit_user_command() -> None:
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
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)
    progress_replies = []
    result = runtime.handle_user_message(
        [{"role": "user", "content": "你觉得我现在是不是卡住了？"}],
        progress_callback=progress_replies.append,
    )

    assert "observe_screen" in client.prompts[0]
    assert "主动获取上下文策略" in client.prompts[0]
    assert "用户输入很短、模糊、寒暄、状态化" in client.prompts[0]
    assert "优先调用 observe_screen 获取当前屏幕" in client.prompts[0]
    assert [progress.reply.translation for progress in progress_replies] == ["我看看。"]
    assert result.actions
    assert result.actions[0].type == SCREEN_OBSERVATION_REQUEST_ACTION


def test_tool_planning_prompt_encourages_web_search_for_uncertain_external_info() -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"確認するね。","zh":"我来确认一下。","tone":"中性"}]}}'

    client = PromptCaptureClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry(
            [
                Tool(
                    name="web__web_search",
                    description="搜索公开网页，并返回标题、链接和简短摘要。",
                    handler=lambda arguments: {"results": []},
                )
            ]
        ),
    )

    runtime.handle_user_message([{"role": "user", "content": "这个库现在最新版本是多少？"}])

    assert "web__web_search" in client.prompts[0]
    assert "最新、外部、公开或不确定的信息" in client.prompts[0]
    assert "主动使用可用的网页搜索工具" in client.prompts[0]
    assert "搜索摘要不足以回答时，再读取具体网页" in client.prompts[0]


def test_autonomous_screen_observation_disabled_hides_screen_tool() -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"見えないよ。","zh":"我先不看屏幕。","tone":"中性"}]}}'

    client = PromptCaptureClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )

    runtime.handle_user_message([{"role": "user", "content": "你觉得我现在是不是卡住了？"}])

    assert OBSERVE_SCREEN_TOOL_NAME not in client.prompts[0]
    assert "当前没有可用的自主屏幕观察工具" in client.prompts[0]


def test_screen_observation_tool_hidden_without_user_message() -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"うん。","zh":"嗯。","tone":"中性"}]}}'

    client = PromptCaptureClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)

    runtime.handle_user_message([{"role": "assistant", "content": "少し見るね。"}])

    assert OBSERVE_SCREEN_TOOL_NAME not in client.prompts[0]


def test_screen_observation_tool_hidden_after_observation_marker() -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"見たよ。","zh":"我看过了。","tone":"中性"}]}}'

    client = PromptCaptureClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)

    runtime.handle_user_message(
        [{"role": "user", "content": f"下午好\n{SCREEN_OBSERVATION_HISTORY_MARKER}"}]
    )

    assert OBSERVE_SCREEN_TOOL_NAME not in client.prompts[0]


def test_model_vision_disabled_hides_screen_observation_tool() -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"見えないよ。","zh":"我看不到哦。","tone":"中性"}]}}'

    client = PromptCaptureClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)
    runtime.set_model_vision_enabled(False)

    runtime.handle_user_message([{"role": "user", "content": "普通聊天"}])

    assert OBSERVE_SCREEN_TOOL_NAME not in client.prompts[0]


def test_screen_observation_tool_hidden_after_image_is_attached() -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"見るよ。","zh":"我看看。","tone":"中性"}]}}'

    client = PromptCaptureClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)
    runtime.set_model_vision_enabled(True)

    runtime.handle_user_message(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "已经有图了"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            }
        ]
    )

    assert OBSERVE_SCREEN_TOOL_NAME not in client.prompts[0]


def test_hidden_screen_observation_call_returns_failure_without_action() -> None:
    class HiddenScreenRequestClient:
        def complete_raw(self, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            return (
                '{"reply":{"segments":[{"ja":"見るね。","zh":"我看看。","tone":"提醒"}]},'
                '"tool_calls":[{"name":"observe_screen","arguments":{},"reason":"需要当前画面"}]}'
            )

        def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            from app.chat_reply import parse_chat_reply

            return parse_chat_reply(
                '{"segments":[{"ja":"今は見えないよ。","zh":"现在看不到哦。","tone":"中性"}]}'
            )

    runtime = AgentRuntime(
        api_client=HiddenScreenRequestClient(),  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)
    runtime.set_model_vision_enabled(False)

    result = runtime.handle_user_message([{"role": "user", "content": "普通聊天"}])

    assert not any(action.type == SCREEN_OBSERVATION_REQUEST_ACTION for action in result.actions)
    assert result.actions[0].payload["error"] == SCREEN_OBSERVATION_DISABLED_ERROR


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


def test_proactive_check_event_generates_segmented_reply() -> None:
    class ProactiveClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.messages: list[list[dict[str, object]]] = []

        def complete_raw(self, system_prompt, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            self.messages.append(messages)
            return (
                '{"reply":{"segments":[{"ja":"少し休もう。","zh":"稍微休息一下吧。","tone":"提醒"}]},'
                '"tool_calls":[]}'
            )

    client = ProactiveClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
    )

    result = runtime.handle_event(
        AgentEvent(
            type="proactive_check",
            payload={
                "seconds_since_pet_interaction": 1800,
                "check_interval_minutes": 20,
                "screen_context_allowed": False,
            },
        )
    )

    assert "低打扰主动搭话" in client.prompts[0]
    assert "只表示用户一段时间没有和桌宠交互" in client.prompts[0]
    assert "不要据此推断用户离开" in client.prompts[0]
    assert "真实可见或已知的具体内容" in client.prompts[0]
    assert "自然搭话、提问或提醒用户" in client.prompts[0]
    assert "tone 和 portrait 要根据内容选择" in client.prompts[0]
    assert "自然搭话" in str(client.messages[0][0]["content"])
    assert "seconds_since_pet_interaction" in str(client.messages[0][0]["content"])
    assert "idle_seconds" not in str(client.messages[0][0]["content"])
    assert result.reply.translation == "稍微休息一下吧。"
    assert result.actions[0].payload["event_type"] == "proactive_check"


def test_proactive_check_event_attaches_screen_context_image() -> None:
    class ProactiveImageClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.messages: list[list[dict[str, object]]] = []

        def complete_raw(self, system_prompt, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            self.messages.append(messages)
            return (
                '{"reply":{"segments":[{"ja":"画面は見たよ。","zh":"我看过画面了。","tone":"提醒"}]},'
                '"tool_calls":[]}'
            )

    client = ProactiveImageClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
    )

    runtime.handle_event(
        AgentEvent(
            type="proactive_check",
            payload={
                "screen_context": {
                    "data_url": "data:image/jpeg;base64,abc123",
                    "width": 800,
                    "height": 600,
                    "captured_at": "2026-05-30T12:00:00+08:00",
                    "screen_name": "DISPLAY1",
                }
            },
        )
    )

    content = client.messages[0][0]["content"]
    assert isinstance(content, list)
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert "abc123" not in content[0]["text"]
    assert "image_attached" in content[0]["text"]
    assert "先理解屏幕画面本身" in client.prompts[0]
    assert "自然评论、提问或轻提醒" in client.prompts[0]
    assert "不要编造看不清" in client.prompts[0]
    assert "不要再请求 observe_screen" in client.prompts[0]
    assert "主动搭话时不要固定使用“提醒”语气" in client.prompts[0]
    assert "自然搭话" in content[0]["text"]


def test_proactive_check_event_attaches_screen_context_image_batch() -> None:
    event = AgentEvent(
        type="proactive_check",
        payload={
            "screen_contexts": [
                {
                    "data_url": "data:image/jpeg;base64,first",
                    "width": 800,
                    "height": 600,
                    "captured_at": "2026-05-30T12:00:00+08:00",
                    "screen_name": "DISPLAY1",
                },
                {
                    "data_url": "data:image/jpeg;base64,second",
                    "width": 800,
                    "height": 600,
                    "captured_at": "2026-05-30T12:10:00+08:00",
                    "screen_name": "DISPLAY1",
                },
            ],
            "screen_context_count": 2,
        },
    )

    messages = _build_event_messages(event)
    content = messages[0]["content"]

    assert isinstance(content, list)
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,first"
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,second"
    assert "first" not in content[0]["text"]
    assert "second" not in content[0]["text"]
    assert "image_attached" in content[0]["text"]
    assert "screen_contexts" in content[0]["text"]


def test_proactive_check_event_redacts_screen_context_image_batch() -> None:
    redacted = _redact_event_for_model(
        AgentEvent(
            type="proactive_check",
            payload={
                "screen_contexts": [
                    {"data_url": "data:image/jpeg;base64,first", "width": 800},
                    {"data_url": "data:image/jpeg;base64,second", "width": 800},
                ],
            },
        )
    )

    contexts = redacted["payload"]["screen_contexts"]
    assert contexts == [
        {"width": 800, "image_attached": True},
        {"width": 800, "image_attached": True},
    ]


def test_proactive_check_event_can_continue_tool_loop_after_tool_results() -> None:
    class ProactiveToolClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.messages: list[list[dict[str, object]]] = []

        def complete_raw(self, system_prompt, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            self.messages.append(messages)
            call_index = len(self.prompts)
            if call_index == 1:
                return (
                    '{"reply":{"segments":[{"ja":"少し確認するね。","zh":"我稍微确认一下。","tone":"中性"}]},'
                    '"tool_calls":[{"name":"browser_get_state","arguments":{},"reason":"看看当前受控浏览器状态"}]}'
                )
            assert "Example Page" in str(messages)
            return (
                '{"reply":{"segments":[{"ja":"ページは開いているみたい。ここで詰まってる？","zh":"页面像是已经打开了。你是卡在这里了吗？","tone":"中性"}]},'
                '"tool_calls":[]}'
            )

    registry = ToolRegistry(
        [
            Tool(
                name="browser_get_state",
                description="读取当前浏览器状态",
                handler=lambda _arguments: {
                    "url": "https://example.com",
                    "title": "Example Page",
                    "loading": False,
                },
            )
        ]
    )
    client = ProactiveToolClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=registry,
    )

    result = runtime.handle_event(
        AgentEvent(
            type="proactive_check",
            payload={
                "seconds_since_pet_interaction": 1800,
                "screen_context_allowed": False,
            },
        )
    )

    assert result.reply.translation == "页面像是已经打开了。你是卡在这里了吗？"
    assert [action.type for action in result.actions] == ["event", "tool_call"]
    assert result.actions[1].payload["tool_name"] == "browser_get_state"
    assert len(client.prompts) == 2
    assert "主动检查事件" in client.prompts[0]
    assert "不要为了显得主动而循环调用工具" in client.prompts[0]


def test_proactive_check_can_request_screen_when_screen_context_allowed() -> None:
    class ProactiveScreenClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return (
                '{"reply":{"segments":[{"ja":"ちょっと見てみるね。","zh":"我稍微看一下。","tone":"中性"}]},'
                '"tool_calls":[{"name":"observe_screen","arguments":{},"reason":"想看看主人现在在做什么"}]}'
            )

    client = ProactiveScreenClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )

    result = runtime.handle_event(
        AgentEvent(
            type="proactive_check",
            payload={
                "seconds_since_pet_interaction": 1800,
                "screen_context_allowed": True,
            },
        )
    )

    assert "observe_screen" in client.prompts[0]
    assert result.actions[0].type == "event"
    assert result.actions[1].type == SCREEN_OBSERVATION_REQUEST_ACTION
    assert result.actions[1].payload["reason"] == "想看看主人现在在做什么"


def test_proactive_check_hides_screen_tool_when_screen_context_disallowed() -> None:
    class ProactiveScreenClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_raw(self, system_prompt, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.prompts.append(system_prompt)
            return '{"reply":{"segments":[{"ja":"声をかけるね。","zh":"我轻轻叫你一下。","tone":"中性"}]}}'

    client = ProactiveScreenClient()
    runtime = AgentRuntime(
        api_client=client,  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
        tools=ToolRegistry([create_screen_observation_tool()]),
    )
    runtime.set_autonomous_screen_observation_enabled(True)

    result = runtime.handle_event(
        AgentEvent(
            type="proactive_check",
            payload={
                "seconds_since_pet_interaction": 1800,
                "screen_context_allowed": False,
            },
        )
    )

    assert f'"name": "{OBSERVE_SCREEN_TOOL_NAME}"' not in client.prompts[0]
    assert result.actions[0].type == "event"
    assert not any(action.type == SCREEN_OBSERVATION_REQUEST_ACTION for action in result.actions)


def test_proactive_check_vision_unsupported_uses_safe_fallback() -> None:
    class ProactiveVisionUnsupportedClient:
        def complete_raw(self, *_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            raise ApiRequestError("model does not support image_url content")

    runtime = AgentRuntime(
        api_client=ProactiveVisionUnsupportedClient(),  # type: ignore[arg-type]
        system_prompt="你是 Sakura。",
    )

    result = runtime.handle_event(
        AgentEvent(
            type="proactive_check",
            payload={
                "screen_context": {
                    "data_url": "data:image/jpeg;base64,abc123",
                    "width": 800,
                    "height": 600,
                }
            },
        )
    )

    assert "不会乱猜" in result.reply.translation
    assert result.actions[0].payload["event_type"] == "proactive_check"


def test_plain_text_messages_do_not_contain_image() -> None:
    assert not messages_contain_image([{"role": "user", "content": "普通聊天"}])
    assert is_vision_unsupported_error("This model does not support image input")


def _runtime_json_path(name: str) -> Path:
    return _runtime_root_path(name) / f"{name}.json"


def _runtime_root_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "__pycache__" / "test_runtime" / name / uuid.uuid4().hex


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


def _write_mcp_config(root: Path, server_body: str) -> Path:
    config_path = root / "mcp.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    indented_body = "\n".join(f"    {line}" if line else "" for line in server_body.splitlines())
    config_path.write_text(
        f"""
enabled: true
servers:
  demo:
    transport: stdio
    command: python
{indented_body}
""".strip(),
        encoding="utf-8",
    )
    return config_path


class _FakeMCPBridge:
    def __init__(self, _server_config, _default_call_timeout) -> None:  # type: ignore[no-untyped-def]
        self.closed = False

    def connect(self) -> None:
        return None

    def list_tools(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                name="echo",
                description="回显消息。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                },
            )
        ]

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        return {
            "server_tool_name": name,
            "arguments": arguments,
        }

    def close(self) -> None:
        self.closed = True


class _MultiToolMCPBridge(_FakeMCPBridge):
    def list_tools(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                name="read_state",
                description="读取状态。",
                input_schema={"type": "object", "properties": {}},
            ),
            MCPToolSpec(
                name="click_target",
                description="点击目标。",
                input_schema={"type": "object", "properties": {}},
            ),
            MCPToolSpec(
                name="hidden_tool",
                description="不应暴露。",
                input_schema={"type": "object", "properties": {}},
            ),
        ]


class _FailingMCPBridge(_FakeMCPBridge):
    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        _ = name, arguments
        raise RuntimeError("MCP 调用失败")
