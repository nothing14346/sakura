# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sakura Desktop Pet** is a proactive desktop Agent framework written in Python/PySide6. Unlike passive chatbots that only respond when prompted, Sakura actively observes your screen and initiates conversations—commenting on what you're doing, offering help, expressing emotion, and remembering interactions over time.

**Core Concept:**
- **Desktop Agent**: Combines LLM (OpenAI-compatible API) + tool calling + screen observation + persistent memory
- **Character-driven**: All behavior (personality, emotions, voice, expressions) is defined in character packages
- **Extensible**: Plugins, MCP tools, and native tools provide capabilities like browser automation, screen capture, file access, web search, and reminders

**Key Technologies:**
- **UI**: PySide6 (Qt-based) with frosted glass effects, tray menu, dynamic portrait expressions
- **LLM Integration**: OpenAI-compatible `tool_calls` protocol with native tool loops
- **Voice**: GPT-SoVITS TTS with emotion-driven tone/weight switching
- **Memory**: Long-term memory with candidate confirmation workflow
- **Tools**: Built-in tools + MCP (Model Context Protocol) servers + plugin system
- **Storage**: YAML configs, JSONL chat history, local vector DB (Qdrant) for memory recall

---

## Quick Start

### Installation & Running

```powershell
# Windows (with pre-bundled runtime)
# Double-click install.bat, then start.bat
.\install.bat
.\start.bat

# From source (Linux/macOS/Windows without runtime)
bash scripts/install.sh
bash scripts/start.sh
```

> **注意**：`runtime/` 目录包含预置 Python 环境。若从 GitHub 源码克隆，需从 Release 页面单独下载 `runtime-windows-x64.zip` 并解压到项目根目录。

### Essential Commands

项目使用 `runtime/python.exe` 作为解释器（预置运行环境），不使用系统 Python：

```powershell
# Run the application
.\runtime\python.exe main.py

# Run all tests
.\runtime\python.exe -m pytest

# Run only unit tests (fast)
.\runtime\python.exe -m pytest tests/unit -v

# Run only UI tests
.\runtime\python.exe -m pytest tests/ui -v

# Run integration tests
.\runtime\python.exe -m pytest tests/integration -v

# Run a single test file
.\runtime\python.exe -m pytest tests/unit/test_agent_runtime.py -v

# Run a single test by name
.\runtime\python.exe -m pytest tests/unit/test_agent_runtime.py::test_handle_user_message -v

# Run tests matching a pattern
.\runtime\python.exe -m pytest -k "tool_registry" -v

# Run with detailed output and timing
.\runtime\python.exe -m pytest tests/unit -vv --durations=10

# Run with debug logging (output to stdout)
.\runtime\python.exe -m pytest tests/unit -v -s
```

**Test Environment Variables** (set in `pytest.ini`):
- `QT_QPA_PLATFORM=offscreen` — Run UI tests without display
- `PYTEST_QT_API=pyside6` — Use PySide6 for Qt testing
- `PYTHONIOENCODING=utf-8` — Ensure UTF-8 handling

---

## Project Structure & Architecture

### Directory Layout

```
sakura/
├── main.py                           # Application entry point
├── app/                              # Main source code
│   ├── agent/                        # Agent runtime & decision logic
│   │   ├── runtime.py                # AgentRuntime: LLM calls + tool loops + memory
│   │   ├── actions.py                # Action/Event/Progress/Result data models
│   │   ├── tools/
│   │   │   ├── registry.py           # ToolRegistry: register/describe/execute tools
│   │   │   ├── permission_policy.py  # Permission & risk control
│   │   │   └── builtin/provider.py   # Built-in tools (memory, reminders, notes, etc.)
│   │   ├── mcp/                      # MCP Server integration
│   │   │   ├── bridge.py             # MCP client wrapper
│   │   │   ├── provider.py           # MCPToolProvider: load & execute MCP tools
│   │   │   └── web_search_server.py  # Built-in web search MCP server
│   │   ├── memory.py                 # MemoryStore: long-term memory with candidates
│   │   ├── reminders.py              # ReminderStore: scheduled reminders
│   │   ├── screen_observation.py     # Screen capture & observation orchestration
│   │   ├── screen_policy.py          # When/how to observe screen
│   │   └── proactive_care.py         # Autonomous check-in events
│   ├── core/                         # App assembly & contracts
│   │   ├── app_context.py            # AppContext: dependency container
│   │   ├── bootstrap.py              # Startup state loading, deferred service init
│   │   ├── chat_pipeline.py          # ChatPipeline: coordinate user messages, events, actions
│   │   ├── chat_worker.py            # Qt Worker thread for async chat processing
│   │   ├── app_builder.py            # AppBuilder: assemble all services
│   │   ├── contracts/                # Core interfaces
│   │   └── plugin_manager.py         # Plugin system (bridging layer)
│   ├── config/                       # Configuration management
│   │   ├── settings_service.py       # YAML config I/O (api.yaml, system_config.yaml)
│   │   ├── models.py                 # Config dataclasses (ApiSettings, DebugLogSettings)
│   │   ├── character_loader.py       # Load character.json, portraits, voice refs
│   │   └── migrations.py             # .env → YAML migration
│   ├── llm/                          # LLM API client & reply parsing
│   │   ├── api_client.py             # OpenAICompatibleClient (chat completions + tool calls)
│   │   ├── chat_reply.py             # Parse segmented JSON replies (ja/zh + tone + portrait)
│   │   ├── prompt_templates.py       # System prompts, event prompts, reply protocol
│   │   └── context_trimming.py       # Message context window management
│   ├── ui/                           # PySide6 UI components
│   │   ├── pet_window.py             # Main window: portrait, subtitle, tray icon
│   │   ├── settings_dialog.py        # Settings: API, character, TTS, debug
│   │   ├── history_window.py         # Chat history review
│   │   ├── portrait_controller.py    # Portrait expressions & scaling
│   │   ├── subtitle_controller.py    # Subtitle display & sync with audio
│   │   ├── tool_confirmation_panel.py # Tool execution confirmation UI
│   │   └── tray_menu.py              # Right-click tray menu
│   ├── voice/                        # TTS providers
│   │   ├── tts.py                    # GPT-SoVITS + Genie + Null providers
│   │   └── playback_controller.py    # Audio playback sync
│   ├── plugins/                      # Plugin system (new native design)
│   │   ├── manager.py                # PluginManager: discover/load/initialize plugins
│   │   ├── models.py                 # PluginManifest, ToolContribution, etc.
│   │   ├── capabilities.py           # PluginCapabilityRegistry
│   │   └── discovery.py              # Plugin discovery from plugins/ + config
│   ├── storage/                      # Local data persistence
│   │   ├── paths.py                  # StoragePaths: data/, memory/, chat_history/, etc.
│   │   ├── chat_history.py           # JSONL-based chat storage
│   │   └── visual_observation.py     # Screen observation records
│   └── orchestration/                # [Planned] Future orchestration layer
├── plugins/                          # Local plugins
│   └── playwright_browser/           # Browser automation plugin
├── characters/                       # Character packages (portraits, JSON, voice refs)
│   └── sakura/                       # Default character
├── data/                             # Runtime data (generated/configured at startup)
│   ├── config/
│   │   ├── api.yaml                  # LLM API config (base_url, api_key, model, timeout)
│   │   ├── system_config.yaml        # UI, memory curation, proactive care, MCP toggles
│   │   ├── plugins.yaml              # Plugin registry (entry, enabled, priority)
│   │   ├── mcp.yaml                  # MCP server configs (stdio/SSE servers)
│   │   └── characters.yaml           # Current character selection
│   ├── chat_history/                 # JSONL files per conversation
│   ├── memory/                       # Long-term memory (candidate + confirmed)
│   └── visual_observations/          # Screen observation records (JSONL)
├── tests/                            # Pytest test suite
│   ├── unit/                         # Fast, isolated unit tests
│   ├── integration/                  # Tests for multi-component flows
│   └── ui/                           # PySide6 UI tests (Qt-based)
├── docs/                             # Project documentation
│   ├── SAKURA_PLUGIN_SDK.md          # Plugin development guide
│   └── ARCHITECTURE.md               # [May exist] Architecture deep-dive
├── sdk/                              # [Deprecated] Old plugin SDK (use app/plugins/ instead)
├── scripts/                          # Install/start scripts
├── tools/mcp/                        # MCP server implementations
└── VERSION                           # Version string (e.g., 0.9.4-dev)
```

### Core Design Principles

#### 1. Agent Runtime & Tool Calling Loop

The heart of the system is `AgentRuntime` in `app/agent/runtime.py`:

```
User Message / Event / Confirmed Action
         ↓
  [ChatPipeline] (serialize input, add context)
         ↓
  [AgentRuntime.handle_*()]
         ↓
  Call LLM with native tool_calls protocol
         ↓
  While no final reply:
    - Parse tool_calls from LLM response
    - Execute each tool via ToolRegistry
    - Check tool permissions & request confirmation if needed
    - Collect results → pass back to LLM as tool role
         ↓
  Parse final reply (segmented JSON: ja/zh + tone + portrait)
         ↓
  [ChatReply] (segmented structure for UI sync)
         ↓
  Return to PetWindow
         ↓
  Display subtitles, switch expressions, play TTS
```

**Key Methods:**
- `handle_user_message()` — Process user input + context
- `handle_event()` — Process autonomous events (proactive care, reminders)
- `handle_confirmed_action()` — Execute tool after user confirmation
- `handle_cancelled_action()` — Handle user rejection

#### 2. Segmented Reply Protocol

The LLM always returns JSON segments. Each segment contains:

```json
{
  "ja_text": "原文",
  "zh_text": "中文",
  "tone": "neutral|happy|sad|angry|surprised",
  "portrait": "default|smile|thinking"
}
```

This allows UI to sync:
- **Subtitles** (ja or zh based on `system_config.yaml`)
- **Portrait expressions** (tone → portrait ID, supports GPT-SoVITS weight switching)
- **TTS playback** (tone drives voice reference selection)

If JSON is malformed, `AgentRuntime` attempts one repair call before failing.

#### 3. Tool Registry & Permission System

`ToolRegistry` (`app/agent/tools/registry.py`) is the single source of truth:

- **Register**: `register(tool)` — Define tool name, description, parameters, handler
- **Describe**: `describe_openai_tools()` — Export as OpenAI `tools` list for LLM
- **Execute**: `execute(tool_name, arguments)` — Call handler, catch errors
- **Permission Control**: `ToolPermissionPolicy` checks capability flags, risk levels, confirmation requirements

**Tool Sources:**
1. **Built-in**: `app/agent/tools/builtin/provider.py` (memory, reminders, notes, todos)
2. **MCP**: `app/agent/mcp/provider.py` (stdio/SSE servers registered in `data/config/mcp.yaml`)
3. **Plugins**: Loaded via `PluginManager`, contribute tools via `ToolContribution`

#### 4. Screen Observation Pipeline

Automatic and on-demand screen monitoring:

1. **Capture**: `ScreenPolicy` decides when to capture (autonomous checks, tool requests, events)
2. **Analyze**: Vision model extracts text + elements + summary
3. **Record**: `VisualObservationStore` persists to JSONL + vector DB for recall
4. **Inject**: Screen context added to LLM prompts for awareness

**Policy Controls** (in `system_config.yaml`):
- `proactive_care.enabled` — Enable autonomous check-ins
- `proactive_care.check_interval_minutes` — How often to check
- `autonomous_screen_observation_enabled` — Allow model to request screenshots

#### 5. Startup & Dependency Injection

**Lightweight Startup** (`bootstrap.py::load_startup_state`):
- Load config (YAML → `ApiSettings`, character selection)
- Load character package (`character.json`, portraits)
- Generate system prompt
- Display UI immediately (portrait visible)

**Deferred Startup** (Qt background worker):
- Initialize TTS provider (slow: download models, start servers)
- Load/initialize plugins
- Register MCP tools (connect to MCP servers)
- Load built-in + plugin tools into `ToolRegistry`
- Inject into `AppContext` once ready

This ensures UI appears instantly, heavy init doesn't block.

#### 6. Memory System

Long-term memory uses a candidate confirmation workflow:

1. **Observation Phase**: `AgentRuntime` detects memory-worthy moments
2. **Candidate Phase**: Proposes memory entry, stores in `data/memory/candidates.jsonl`
3. **UI Confirmation**: User reviews in settings, approves/rejects
4. **Confirmed Phase**: Moves to `data/memory/confirmed.jsonl`
5. **Recall**: On new messages, vector search (Qdrant) retrieves relevant memories as context

**Auto-Curation** (optional): `MemoryCurator` runs periodically to merge/consolidate memories.

---

## Configuration

### API Configuration (`data/config/api.yaml`)

```yaml
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "your_key_here"
  model: "gpt-4.1-mini"  # Must support vision + tool_calls
  timeout_seconds: 60

tts:
  enabled: false
  provider: gpt-sovits
  gpt_sovits:
    api_url: "http://127.0.0.1:9880/tts"
    ref_lang: ja
    text_lang: ja
```

### System Configuration (`data/config/system_config.yaml`)

```yaml
ui:
  subtitle_language: zh  # ja or zh
  portrait_scale_percent: 100

proactive_care:
  enabled: true
  check_interval_minutes: 20
  cooldown_minutes: 10

memory_curation:
  enabled: true

mcp:
  windows_enabled: false  # Enable Windows desktop tools

debug:
  enabled: false
```

### Character Package Structure

```
characters/sakura/
├── character.json          # Personality card, system prompt override
├── card.md                 # Markdown description
├── portraits/              # Expression variants
│   ├── default.png
│   ├── smile.png
│   └── ...
└── voice/                  # TTS reference audio
    ├── ref_audio.ogg
    └── ref_text.txt
```

---

## Common Development Tasks

### Adding a Built-in Tool

1. **Define tool function** in `app/agent/tools/builtin/provider.py`:
   ```python
   def create_builtin_tool_registry() -> ToolRegistry:
       registry = ToolRegistry()
       registry.register(Tool(
           name="my_tool",
           description="Does something",
           parameters={"type": "object", "properties": {...}, "required": [...]},
           handler=lambda **kwargs: {"result": "..."},
           group="default",
           risk="low"
       ))
       return registry
   ```

2. **Test** in `tests/unit/test_tool_registry.py` or integration tests

### Adding a Plugin

1. **Create plugin structure**:
   ```
   plugins/my_plugin/
   ├── plugin.py
   └── __init__.py
   ```

2. **Implement `PluginBase`** (see `docs/SAKURA_PLUGIN_SDK.md`):
   ```python
   from app.plugins.models import PluginBase, ToolContribution
   
   class MyPlugin(PluginBase):
       @property
       def plugin_id(self) -> str:
           return "my_plugin"
       
       def initialize(self, register, plugin_root, host):
           register.register_tool(ToolContribution(
               name="my_tool",
               description="...",
               parameters={...},
               handler=self._handler
           ))
       
       def _handler(self, **kwargs):
           return {"result": "..."}
   ```

3. **Register in `data/config/plugins.yaml`**:
   ```yaml
   - entry: plugins.my_plugin.plugin:MyPlugin
     enabled: true
     priority: 100
   ```

### Modifying System Prompts or Reply Protocol

All prompt templates are in `app/llm/prompt_templates.py` and `app/llm/prompts/`:

- **System prompt**: `build_system_prompt()` — Character personality, tool list, memory context
- **Reply protocol**: `build_agent_reply_protocol()` — JSON structure for responses
- **Event prompts**: `build_event_system_prompt()` — Autonomous check-in prompts

Changes here affect all Agent decisions. Test with `tests/unit/test_prompt_templates.py`.

### Debugging Agent Behavior

Enable debug logging in `system_config.yaml`:

```yaml
debug:
  enabled: true
  body_enabled: true    # Log full API request/response bodies
  file_enabled: true    # Write logs to file
```

Logs appear in terminal output and (if `file_enabled`) in `data/debug.log`.

Key debug points:
- `app/core/debug_log.py` — Centralized logging
- `ChatWorker` messages show input context
- `AgentRuntime` logs each tool call and decision
- `api_client.py` logs LLM requests/responses

---

## Testing Guidelines

### Test Organization

- **`tests/unit/`** — Fast, isolated tests (no LLM calls, no UI, ~1-10s each)
  - `test_agent_runtime.py` — Core runtime logic
  - `test_tool_registry.py` — Tool system
  - `test_api_client.py` — LLM client
  - `test_config.py` — Configuration loading
  - `test_memory_curator.py` — Memory system
  - etc.

- **`tests/integration/`** — Multi-component flows (~10-60s each)
  - `test_chat_pipeline.py` — Full chat flow (user → runtime → reply)
  - `test_agent_core.py` — Agent with tools

- **`tests/ui/`** — PySide6 UI tests (requires Qt, slower)
  - `test_pet_window.py` — Main window behavior
  - `test_history_window.py` — History review

### Test Markers

```python
@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.requires_llm        # Needs real API (slow, may fail)
@pytest.mark.requires_network    # Needs internet
```

### Mocking LLM Responses

Most tests use `pytest-mock` with fixture-based mocks:

```python
def test_something(mocker):
    mock_response = {"choices": [{"message": {"content": "..."}}]}
    mocker.patch(
        "app.llm.api_client.OpenAICompatibleClient.chat_completion",
        return_value=mock_response
    )
```

### Running Tests Before Commit

```powershell
# Quick smoke test (unit only)
.\runtime\python.exe -m pytest tests/unit -x

# Full test suite (all layers)
.\runtime\python.exe -m pytest

# With coverage
.\runtime\python.exe -m pytest --cov=app tests/
```

---

## Important Notes for Editing

### Architecture Constraints

1. **Do not bypass `ToolRegistry`** — All tool access must go through it for permission/risk control
2. **All replies must be segmented JSON** — UI sync depends on `{ja_text, zh_text, tone, portrait}`
3. **Async model**: Use Qt `QThread` + signals for long-running operations (LLM calls, plugin init)
4. **Config in YAML, not code** — API keys, model names, feature flags belong in `data/config/`, not hardcoded

### Files to Avoid Modifying Unless Necessary

- `sdk/` — Deprecated, use `app/plugins/` instead
- `tools/mcp/`, `third_party/` — External code; coordinate changes
- `characters/sakura/` — Default character assets; treat as immutable

### Safe Areas for New Code

- `app/agent/` — Add tools, memory logic, event handlers
- `app/plugins/` — Plugin implementations
- `app/config/` — Config models (keep migrations in mind)
- `app/core/` — Service composition, app lifecycle
- `tests/` — All test additions welcome

### Git Workflow

- **Branches**: `main`（发布用）/ `dev`（开发主干）/ `feat/xxx`（新功能）/ `fix/xxx`（修复）
- 日常开发从 `dev` 分出 `feat/xxx` 或 `fix/xxx` 分支，完成后合并回 `dev`
- 发布时将 `dev` 合并进 `main`，不直接在 `main` 上开发
- **Commits**: 使用常规类型前缀：`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`
- **Tests**: 推送前确保单元测试和集成测试通过
- 不对 `main` / `dev` 做 force push

---

## Performance Notes

- **LLM Calls**: Default timeout 60s; adjust in `api.yaml` if needed
- **Screen Observation**: Async, but can be expensive (vision model costs); use policies to limit
- **Memory Recall**: Vector search is fast (<100ms), but consolidation can be slow
- **Plugin Init**: Happens in background after UI shows; monitor `app/core/bootstrap.py::build_deferred_services`
- **UI Responsiveness**: All blocking operations run in `ChatWorker` thread; keep main thread free

---

## Useful References

- **README.md** / **README.en.md** — User-facing overview and tutorials
- **AGENTS.md** — Constraints on AI agents modifying this repo
- **docs/SAKURA_PLUGIN_SDK.md** — Plugin API details
- **tests/conftest.py** — Pytest fixtures and test setup
- **pytest.ini** — Test runner configuration
