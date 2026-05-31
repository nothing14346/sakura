from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable

from app.agent.actions import AgentAction, AgentEvent, AgentProgress, AgentResult, PendingToolAction
from app.agent.memory import MemoryStore
from app.agent.screen_tools import (
    OBSERVE_SCREEN_TOOL_NAME,
    SCREEN_OBSERVATION_CAPABILITY,
    SCREEN_OBSERVATION_DISABLED_ERROR,
    SCREEN_OBSERVATION_REQUEST_ACTION,
)
from app.agent.tool_registry import ToolExecutionResult, ToolRegistry
from app.api_client import (
    ApiRequestError,
    ChatMessage,
    OpenAICompatibleClient,
    is_vision_unsupported_error,
    messages_contain_image,
)
from app.chat_reply import ChatReply, parse_chat_reply
from app.debug_log import debug_log, summarize_messages
from app.prompt_templates import (
    build_agent_reply_protocol,
    build_context_acquisition_strategy,
    build_event_reply_protocol,
    build_proactive_rules,
)
from app.screen_observation import (
    MANUAL_SCREEN_OBSERVATION_HISTORY_MARKER,
    SCREEN_OBSERVATION_HISTORY_MARKER,
)


MAX_AGENT_STEPS_PER_TURN = 4
MAX_TOOL_CALLS_PER_STEP = 3
MAX_TOOL_CALLS_PER_TURN = 8
MAX_TOOL_RESULT_CHARS = 6000
ProgressCallback = Callable[[AgentProgress], None]


class AgentRuntime:
    """封装聊天决策链路，为后续工具调用和长期记忆留下扩展点。"""

    def __init__(
        self,
        api_client: OpenAICompatibleClient,
        system_prompt: str,
        reply_tones: list[str] | None = None,
        reply_portraits: list[str] | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.api_client = api_client
        self.system_prompt = system_prompt
        self.reply_tones = [*reply_tones] if reply_tones is not None else []
        self.reply_portraits = [*reply_portraits] if reply_portraits is not None else []
        self.tools = tools or ToolRegistry()
        self.memory = memory or MemoryStore()
        self.model_vision_enabled = True
        self.autonomous_screen_observation_enabled = False

    def update_character(
        self,
        system_prompt: str,
        reply_tones: list[str] | None = None,
        reply_portraits: list[str] | None = None,
    ) -> None:
        """角色切换后同步系统提示词、可用语气和可用立绘列表。"""
        self.system_prompt = system_prompt
        self.reply_tones = [*reply_tones] if reply_tones is not None else []
        self.reply_portraits = [*reply_portraits] if reply_portraits is not None else []

    def set_model_vision_enabled(self, enabled: bool) -> None:
        """允许模型在需要时请求一次当前屏幕截图。"""
        self.model_vision_enabled = enabled

    def set_autonomous_screen_observation_enabled(self, enabled: bool) -> None:
        """允许模型在对话或主动事件中自主决定是否观察屏幕。"""
        self.autonomous_screen_observation_enabled = enabled

    def handle_user_message(
        self,
        messages: list[ChatMessage],
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        turn_started_at = time.perf_counter()
        allow_screen_observation = (
            self.model_vision_enabled
            and self.autonomous_screen_observation_enabled
            and not messages_contain_image(messages)
            and _should_offer_screen_observation(messages)
        )
        debug_log(
            "AgentRuntime",
            "开始处理用户消息",
            {
                "message_count": len(messages),
                "allow_screen_observation": allow_screen_observation,
                "model_vision_enabled": self.model_vision_enabled,
                "autonomous_screen_observation_enabled": self.autonomous_screen_observation_enabled,
                "messages": summarize_messages(messages),
            },
        )
        return self._run_tool_loop(
            messages,
            allow_screen_observation=allow_screen_observation,
            turn_started_at=turn_started_at,
            vision_unsupported_reply=_build_vision_unsupported_reply(),
            progress_callback=progress_callback,
        )

    def _run_tool_loop(
        self,
        messages: list[ChatMessage],
        *,
        allow_screen_observation: bool,
        turn_started_at: float,
        planning_extra_instructions: str = "",
        initial_actions: list[AgentAction] | None = None,
        vision_unsupported_reply: ChatReply | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        """执行受限 Agent 循环：规划工具、执行、回填结果，并允许继续规划。"""
        working_messages: list[ChatMessage] = [*messages]
        execution_results: list[ToolExecutionResult] = []
        emitted_actions: list[AgentAction] = [*(initial_actions or [])]
        total_tool_calls = 0
        for step_index in range(MAX_AGENT_STEPS_PER_TURN):
            try:
                planning_started_at = time.perf_counter()
                model_content = self.api_client.complete_raw(
                    self._build_tool_planning_prompt(
                        allow_screen_observation=allow_screen_observation,
                        step_index=step_index,
                        remaining_steps=MAX_AGENT_STEPS_PER_TURN - step_index - 1,
                        extra_instructions=planning_extra_instructions,
                    ),
                    working_messages,
                    temperature=0.8,
                )
            except ApiRequestError as exc:
                if messages_contain_image(working_messages) and is_vision_unsupported_error(exc):
                    debug_log("AgentRuntime", "视觉输入不受支持，返回兜底回复", {"error": str(exc)})
                    return AgentResult(
                        reply=vision_unsupported_reply or _build_vision_unsupported_reply(),
                        actions=emitted_actions,
                    )
                raise
            debug_log(
                "AgentRuntime",
                "工具规划模型返回",
                {
                    "step_index": step_index,
                    "content": model_content,
                    "planning_elapsed_ms": int((time.perf_counter() - planning_started_at) * 1000),
                },
            )
            agent_data = _load_json_object(model_content)
            if agent_data is None:
                debug_log(
                    "AgentRuntime",
                    "模型返回非 JSON，按普通回复解析",
                    {
                        "step_index": step_index,
                        "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                    },
                )
                return AgentResult(
                    reply=parse_chat_reply(model_content),
                    actions=emitted_actions,
                )

            tool_calls = _parse_tool_calls(agent_data.get("tool_calls"))
            debug_log(
                "AgentRuntime",
                "工具规划解析完成",
                {
                    "step_index": step_index,
                    "has_reply": isinstance(agent_data.get("reply"), dict),
                    "tool_calls": tool_calls,
                    "total_tool_calls": total_tool_calls,
                },
            )
            if not tool_calls:
                debug_log(
                    "AgentRuntime",
                    "多步循环完成，返回模型回复",
                    {
                        "step_index": step_index,
                        "tool_result_count": len(execution_results),
                        "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                    },
                )
                return AgentResult(
                    reply=_parse_agent_reply(agent_data, model_content),
                    actions=emitted_actions,
                )

            _emit_progress(
                progress_callback,
                agent_data,
                stage="tool_planning",
                metadata={
                    "step_index": step_index,
                    "tool_names": [call["name"] for call in tool_calls],
                    "tool_call_count": len(tool_calls),
                },
            )
            step_results: list[ToolExecutionResult] = []
            pending_actions: list[PendingToolAction] = []
            tools_started_at = time.perf_counter()
            allowed_calls = min(
                len(tool_calls),
                MAX_TOOL_CALLS_PER_STEP,
                max(0, MAX_TOOL_CALLS_PER_TURN - total_tool_calls),
            )
            for call in tool_calls[:allowed_calls]:
                total_tool_calls += 1
                debug_log("AgentRuntime", "准备工具调用", {"step_index": step_index, **call})
                prepared = self.tools.prepare_or_execute(
                    call["name"],
                    call["arguments"],
                    call.get("reason", ""),
                )
                if isinstance(prepared, PendingToolAction):
                    debug_log("AgentRuntime", "工具调用等待用户确认", prepared.to_dict())
                    pending_actions.append(prepared)
                    continue

                if _is_screen_observation_request(prepared):
                    if allow_screen_observation:
                        screen_action = AgentAction(
                            type=SCREEN_OBSERVATION_REQUEST_ACTION,
                            payload={"reason": call.get("reason", "")},
                        )
                        debug_log(
                            "AgentRuntime",
                            "请求屏幕观察 follow-up",
                            {
                                "step_index": step_index,
                                "reason": call.get("reason", ""),
                                "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                            },
                        )
                        return AgentResult(
                            reply=_build_screen_observation_request_reply(),
                            actions=[*emitted_actions, screen_action],
                        )
                    prepared = ToolExecutionResult(
                        tool_name=OBSERVE_SCREEN_TOOL_NAME,
                        success=False,
                        content="",
                        error=SCREEN_OBSERVATION_DISABLED_ERROR,
                    )

                debug_log("AgentRuntime", "工具调用完成", _redact_tool_result_for_model(prepared))
                step_results.append(prepared)
                execution_results.append(prepared)
                emitted_actions.append(
                    AgentAction(
                        type="tool_call",
                        payload=_redact_tool_result_for_model(prepared),
                    )
                )

            skipped_calls = len(tool_calls) - allowed_calls
            if skipped_calls > 0:
                limit_error = (
                    f"本步骤最多执行 {MAX_TOOL_CALLS_PER_STEP} 个工具调用，"
                    f"整轮最多执行 {MAX_TOOL_CALLS_PER_TURN} 个工具调用，"
                    f"已跳过 {skipped_calls} 个后续调用。"
                )
                limit_result = ToolExecutionResult(
                    tool_name="runtime",
                    success=False,
                    content="",
                    error=limit_error,
                )
                debug_log(
                    "AgentRuntime",
                    "工具调用数量超过上限",
                    {
                        "step_index": step_index,
                        "requested": len(tool_calls),
                        "allowed": allowed_calls,
                        "total_tool_calls": total_tool_calls,
                        "step_limit": MAX_TOOL_CALLS_PER_STEP,
                        "turn_limit": MAX_TOOL_CALLS_PER_TURN,
                    },
                )
                step_results.append(limit_result)
                execution_results.append(limit_result)
                emitted_actions.append(
                    AgentAction(
                        type="tool_call",
                        payload=_redact_tool_result_for_model(limit_result),
                    )
                )

            if pending_actions:
                debug_log(
                    "AgentRuntime",
                    "返回待确认动作",
                    {
                        "step_index": step_index,
                        "pending_actions": [action.to_dict() for action in pending_actions],
                        "tools_elapsed_ms": int((time.perf_counter() - tools_started_at) * 1000),
                        "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                    },
                )
                return AgentResult(
                    reply=_build_pending_action_reply(pending_actions),
                    actions=[
                        *emitted_actions,
                        *[
                            AgentAction(
                                type="pending_action",
                                payload=action.to_dict(),
                            )
                            for action in pending_actions
                        ],
                    ],
                )

            if not step_results:
                break

            working_messages.extend(
                [
                    {"role": "assistant", "content": model_content},
                    _build_tool_results_message(
                        step_results,
                        include_images=self.model_vision_enabled,
                    ),
                ]
            )
            if total_tool_calls >= MAX_TOOL_CALLS_PER_TURN:
                break

        try:
            final_started_at = time.perf_counter()
            final_reply = self.api_client.chat(
                self._build_final_reply_prompt(),
                working_messages,
                self.reply_tones,
                self.reply_portraits,
            )
        except Exception as exc:
            print(f"[AgentRuntime] 工具结果总结失败，使用本地兜底回复：{exc}")
            debug_log("AgentRuntime", "工具结果总结失败，使用本地兜底回复", {"error": str(exc)})
            final_reply = _build_fallback_tool_reply(execution_results)
        debug_log(
            "AgentRuntime",
            "最终回复生成完成",
            {
                "segments": len(final_reply.segments),
                "actions": [_redact_tool_result_for_model(result) for result in execution_results],
                "final_reply_elapsed_ms": int((time.perf_counter() - final_started_at) * 1000),
                "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
            },
        )
        return AgentResult(
            reply=final_reply,
            actions=emitted_actions,
        )

    def handle_confirmed_action(
        self,
        action: PendingToolAction,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        _ = progress_callback
        debug_log("AgentRuntime", "执行已确认动作", action.to_dict())
        result = self.tools.execute(action.tool_name, action.arguments)
        try:
            reply = self.api_client.chat(
                self._build_final_reply_prompt(),
                [
                    _build_tool_results_message(
                        [result],
                        include_images=self.model_vision_enabled,
                    )
                ],
                self.reply_tones,
                self.reply_portraits,
            )
        except Exception as exc:
            print(f"[AgentRuntime] 确认动作总结失败，使用本地兜底回复：{exc}")
            debug_log("AgentRuntime", "确认动作总结失败，使用本地兜底回复", {"error": str(exc)})
            reply = _build_fallback_tool_reply([result])
        debug_log(
            "AgentRuntime",
            "已确认动作处理完成",
            {
                "result": _redact_tool_result_for_model(result),
                "segments": len(reply.segments),
            },
        )
        return AgentResult(
            reply=reply,
            actions=[
                AgentAction(
                    type="tool_call",
                    payload=_redact_tool_result_for_model(result),
                )
            ],
        )

    def handle_cancelled_action(self, action: PendingToolAction) -> AgentResult:
        debug_log("AgentRuntime", "用户取消待确认动作", action.to_dict())
        return AgentResult(
            reply=parse_chat_reply(
                json.dumps(
                    {
                        "segments": [
                            {
                                "ja": "わかった。実行しないでおくね。",
                                "zh": "知道了。我不会执行这个动作。",
                                "tone": "中性",
                                "portrait": "站立待机",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            ),
            actions=[
                AgentAction(
                    type="cancelled_action",
                    payload=action.to_dict(),
                )
            ],
        )

    def handle_event(
        self,
        event: AgentEvent,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        if event.type not in {"reminder_due", "proactive_check"}:
            return AgentResult(reply=parse_chat_reply("未対応のイベントだよ。"))

        debug_log("AgentRuntime", "处理主动事件", {"event": {"type": event.type, "payload": event.payload}})
        event_messages = _build_event_messages(event)
        event_action = AgentAction(
            type="event",
            payload={
                "event_type": event.type,
                "event_payload": event.payload,
            },
        )
        if event.type == "proactive_check":
            screen_context_allowed = bool(event.payload.get("screen_context_allowed"))
            allow_screen_observation = (
                screen_context_allowed
                and not messages_contain_image(event_messages)
            )
            return self._run_tool_loop(
                event_messages,
                allow_screen_observation=allow_screen_observation,
                turn_started_at=time.perf_counter(),
                planning_extra_instructions=_build_proactive_tool_loop_rules(),
                initial_actions=[event_action],
                vision_unsupported_reply=_build_proactive_vision_unsupported_reply(),
                progress_callback=progress_callback,
            )

        try:
            reply = self.api_client.chat(
                self._build_event_reply_prompt(event.type),
                event_messages,
                self.reply_tones,
                self.reply_portraits,
            )
        except ApiRequestError as exc:
            if messages_contain_image(event_messages) and is_vision_unsupported_error(exc):
                debug_log("AgentRuntime", "主动事件视觉输入不受支持，返回兜底回复", {"error": str(exc)})
                return AgentResult(reply=_build_proactive_vision_unsupported_reply())
            raise
        return AgentResult(
            reply=reply,
            actions=[event_action],
        )

    def _build_tool_planning_prompt(
        self,
        allow_screen_observation: bool = False,
        step_index: int = 0,
        remaining_steps: int = MAX_AGENT_STEPS_PER_TURN - 1,
        extra_instructions: str = "",
    ) -> str:
        allowed_capabilities = {SCREEN_OBSERVATION_CAPABILITY} if allow_screen_observation else set()
        tools = self.tools.describe_tools(allowed_capabilities=allowed_capabilities)
        tool_descriptions = json.dumps(
            tools,
            ensure_ascii=False,
            indent=2,
        )
        memory_summary = self._memory_summary()
        current_time = datetime.now().astimezone().isoformat(timespec="seconds")
        reply_protocol = build_agent_reply_protocol(self.reply_tones, self.reply_portraits)
        context_strategy = build_context_acquisition_strategy(
            allow_screen_observation=allow_screen_observation
        )
        screen_observation_rule = (
            "- 屏幕理解只通过 observe_screen 请求，由 Sakura 现有截图流程处理；不要请求 Windows-MCP 的 Screenshot 或 Snapshot。"
            if allow_screen_observation
            else "- 当前没有可用的屏幕观察工具；不要请求 Windows-MCP 的 Screenshot 或 Snapshot，也不要臆造当前屏幕。"
        )
        return f"""
{self.system_prompt.strip()}

你现在可以作为桌面陪伴型 Agent 判断是否需要调用内部工具。
如果需要工具，返回 reply 和 tool_calls；如果不需要工具，tool_calls 返回空数组或省略。
输出必须是单个 JSON object；不要添加 Markdown、代码块、反引号、工具名伪代码或 JSON 外的解释文字。

长期记忆摘要：
{memory_summary}

当前本地时间：
{current_time}

当前 Agent 循环：
- 这是第 {step_index + 1} 步，之后最多还可以继续 {remaining_steps} 步。
- 你可以根据已有工具结果继续请求下一批工具；信息足够或已经完成时，tool_calls 必须为空，并在 reply 中给最终答复。
- 每步最多请求 {MAX_TOOL_CALLS_PER_STEP} 个工具，整轮最多 {MAX_TOOL_CALLS_PER_TURN} 个工具；不要为了凑数量而调用工具。
- 不要重复调用刚失败且参数相同的工具；如果受限、需要确认或信息不足，请停止循环并说明当前状态。

可用工具：
{tool_descriptions}

{reply_protocol}

{context_strategy}

工具要求：
- 如果需要调用工具，reply 只写执行前可以直接说给用户听的短句，例如“我先查一下”“我看一下屏幕”；不要提前给最终结论。
- 如果工具可以帮助完成用户请求，优先用 tool_calls 表达要执行的动作。
- 不要臆造工具名；只能使用上面列出的工具。
- requires_confirmation 为 true 的工具只会在用户确认后执行；你仍然可以发起 tool_calls，但必须说明原因。
- 浏览器内部任务优先使用 browser__ 前缀的 Playwright MCP 工具；先用 browser__browser_snapshot 获取页面结构，再基于真实 target 调用点击、输入、表单、等待等工具。
- 桌面窗口、应用切换、鼠标坐标、快捷键等浏览器外部任务才使用 windows__ 前缀的 Windows-MCP 工具；不要用 windows__Click/Move/Type 操作普通网页内部元素。
{screen_observation_rule}
- 如果 browser__ 工具不可用，读取 Sakura 受控浏览器内容时再使用 browser_get_content；需要打开受控浏览器网页时使用 browser_open_url。
- 需要网页交互时，只能基于当前页面真实内容选择工具，不要臆造 selector、target 或页面内容。
{extra_instructions.strip()}
- 用户说“几分钟后/几秒后/一会儿后”等相对提醒时，add_reminder 必须使用 delay_minutes 或 delay_seconds，不要自己换算 trigger_at。
- 只有用户给出明确日期或钟点时，add_reminder 才使用 trigger_at。
- 长期记忆由后台整理器自动维护；你不要尝试写入记忆。
- 只有用户明确要求忘掉信息时，才使用 forget_memory。
""".strip()
    def _build_final_reply_prompt(self) -> str:
        return f"""
{self.system_prompt.strip()}

你会收到上一轮工具调用结果。请基于这些结果给用户最终回复。
不要再次请求工具，不要提及内部 JSON、工具协议或实现细节。
""".strip()

    def _build_event_reply_prompt(self, event_type: str = "reminder_due") -> str:
        proactive_rules = ""
        example_tone = "提醒"
        if event_type == "proactive_check":
            example_tone = "中性"
            proactive_rules = build_proactive_rules()
        reply_protocol = build_event_reply_protocol(
            self.reply_tones,
            self.reply_portraits,
            example_tone=example_tone,
        )
        return f"""
{self.system_prompt.strip()}

你正在处理 Sakura 桌宠的主动事件。请用角色语气自然搭话、提问或提醒用户。
{reply_protocol}
- 不要提及内部事件类型、JSON 或工具实现。
{proactive_rules}
""".strip()

    def _memory_summary(self) -> str:
        try:
            return self.memory.summary()
        except Exception as exc:
            return f"长期记忆读取失败：{exc}"


def _load_json_object(content: str) -> dict[str, Any] | None:
    text = _strip_code_fence(content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data

    fallback: dict[str, Any] | None = None
    for candidate in _iter_json_object_candidates(content):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if fallback is None:
            fallback = data
        if _looks_like_agent_payload(data):
            return data
    return fallback


def _looks_like_agent_payload(data: dict[str, Any]) -> bool:
    return "reply" in data or "tool_calls" in data


def _iter_json_object_candidates(content: str):
    """从混杂模型输出中抽取可能的 JSON object，避免泄露规划协议到前台。"""
    text = _strip_code_fence(content.strip())
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _strip_code_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _parse_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments", {})
        reason = item.get("reason", "")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(reason, str):
            reason = ""
        tool_calls.append({"name": name.strip(), "arguments": arguments, "reason": reason.strip()})
    return tool_calls


def _parse_agent_reply(agent_data: dict[str, Any], fallback_content: str) -> ChatReply:
    reply_data = agent_data.get("reply")
    if isinstance(reply_data, dict):
        return parse_chat_reply(json.dumps(reply_data, ensure_ascii=False))
    return parse_chat_reply(fallback_content)


def _parse_progress_reply(agent_data: dict[str, Any]) -> ChatReply | None:
    reply_data = agent_data.get("reply")
    if not isinstance(reply_data, dict):
        return None
    reply = parse_chat_reply(json.dumps(reply_data, ensure_ascii=False))
    return reply if reply.text.strip() else None


def _emit_progress(
    progress_callback: ProgressCallback | None,
    agent_data: dict[str, Any],
    *,
    stage: str,
    metadata: dict[str, Any],
) -> None:
    if progress_callback is None:
        return
    reply = _parse_progress_reply(agent_data)
    if reply is None:
        return
    try:
        progress_callback(AgentProgress(reply=reply, stage=stage, metadata=metadata))
    except Exception as exc:
        debug_log("AgentRuntime", "中间回复回调失败，已忽略", {"error": str(exc), "stage": stage})


def _is_screen_observation_request(result: ToolExecutionResult) -> bool:
    if result.tool_name != OBSERVE_SCREEN_TOOL_NAME or not result.success:
        return False
    if not isinstance(result.content, dict):
        return False
    return result.content.get("action") == SCREEN_OBSERVATION_REQUEST_ACTION


def _should_offer_screen_observation(messages: list[ChatMessage]) -> bool:
    """只在当前轮仍有可关联用户消息时开放自主屏幕观察。"""
    text = _latest_user_text(messages)
    if text is None:
        return False
    return (
        SCREEN_OBSERVATION_HISTORY_MARKER not in text
        and MANUAL_SCREEN_OBSERVATION_HISTORY_MARKER not in text
    )


def _latest_user_text(messages: list[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(parts)
        return ""
    return None


def _build_tool_results_message(
    results: list[ToolExecutionResult],
    include_images: bool = False,
) -> ChatMessage:
    text = _format_tool_results_for_model(results)
    images = _extract_tool_result_images(results) if include_images else []
    if not images:
        return {"role": "user", "content": text}

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "low",
            },
        }
        for image_url in images
    )
    return {"role": "user", "content": content}


def _format_tool_results_for_model(results: list[ToolExecutionResult]) -> str:
    return (
        "工具执行结果如下，请据此给用户最终回复。"
        "如果工具结果标记已附加浏览器截图，请结合截图兜底判断页面内容，不要臆造看不到的信息：\n"
        + json.dumps(
            [_redact_tool_result_for_model(result) for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )


def _redact_tool_result_for_model(result: ToolExecutionResult) -> dict[str, Any]:
    data = result.to_dict()
    content = data.get("content")
    if isinstance(content, str):
        data["content"] = _truncate_text_for_model(content, MAX_TOOL_RESULT_CHARS)
        return data
    if not isinstance(content, dict):
        return data

    redacted = dict(content)
    if redacted.pop("screenshot_data_url", None):
        redacted["screenshot_attached"] = True
    data["content"] = _truncate_value_for_model(redacted, MAX_TOOL_RESULT_CHARS)
    return data


def _truncate_value_for_model(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return value
    head_chars = max(1, max_chars // 2)
    tail_chars = max(0, max_chars - head_chars)
    return {
        "truncated": True,
        "original_chars": len(text),
        "omitted_chars": max(0, len(text) - head_chars - tail_chars),
        "head": text[:head_chars],
        "tail": text[-tail_chars:] if tail_chars else "",
    }


def _truncate_text_for_model(text: str, max_chars: int) -> str | dict[str, Any]:
    if len(text) <= max_chars:
        return text
    head_chars = max(1, max_chars // 2)
    tail_chars = max(0, max_chars - head_chars)
    return {
        "truncated": True,
        "original_chars": len(text),
        "omitted_chars": max(0, len(text) - head_chars - tail_chars),
        "head": text[:head_chars],
        "tail": text[-tail_chars:] if tail_chars else "",
    }


def _extract_tool_result_images(results: list[ToolExecutionResult]) -> list[str]:
    images: list[str] = []
    for result in results:
        if not isinstance(result.content, dict):
            continue
        image_url = result.content.get("screenshot_data_url")
        if isinstance(image_url, str) and image_url.startswith("data:image/"):
            images.append(image_url)
    return images[:1]


def _build_pending_action_reply(actions: list[PendingToolAction]) -> ChatReply:
    if len(actions) == 1:
        action = actions[0]
        text = _describe_pending_action(action)
        return parse_chat_reply(
            json.dumps(
                {
                    "segments": [
                        {
                            "ja": "実行する前に確認させて。",
                            "zh": f"执行前需要你确认：{text}",
                            "tone": "提醒",
                            "portrait": "伸手命令",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "いくつか確認が必要な操作があるよ。",
                        "zh": f"有 {len(actions)} 个动作需要你确认，我会先处理第一个。",
                        "tone": "提醒",
                        "portrait": "伸手命令",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _describe_pending_action(action: PendingToolAction) -> str:
    if action.tool_name == "open_url":
        return f"打开网页 {action.arguments.get('url', '')}"
    if action.tool_name == "browser_open_url":
        return f"在受控浏览器中打开网页 {action.arguments.get('url', '')}"
    if action.tool_name == "browser_scroll":
        direction = action.arguments.get("direction", "")
        amount = action.arguments.get("amount", "")
        return f"滚动受控浏览器页面 {direction} {amount}"
    if action.tool_name == "browser_click":
        return f"点击受控浏览器页面元素 {action.arguments.get('selector', '')}"
    if action.tool_name == "open_local_folder":
        return f"打开文件夹 {action.arguments.get('path', '')}"
    if action.tool_name.startswith("browser__"):
        return f"执行浏览器 MCP 操作 {action.tool_name.removeprefix('browser__')}"
    if action.tool_name.startswith("windows__"):
        return f"执行 Windows 桌面 MCP 操作 {action.tool_name.removeprefix('windows__')}"
    return f"执行 {action.tool_name}"


def _build_screen_observation_request_reply() -> ChatReply:
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "画面を確認してから答えるね。",
                        "zh": "我先看一下当前画面再回答。",
                        "tone": "提醒",
                        "portrait": "伸手命令",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _build_fallback_tool_reply(results: list[ToolExecutionResult]) -> ChatReply:
    if not results:
        return parse_chat_reply("ツール結果の確認に失敗したよ。")

    succeeded = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    if succeeded and not failed:
        summary = _summarize_tool_results(succeeded)
        return parse_chat_reply(
            json.dumps(
                {
                    "segments": [
                        {
                            "ja": f"処理は終わったよ。{summary}",
                            "zh": f"已经处理好了。{summary}",
                            "tone": "提醒",
                            "portrait": "自信拍胸",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

    error_text = "；".join(
        f"{result.tool_name}: {result.error or '执行失败'}"
        for result in failed
    )
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "処理中に問題が起きたみたい。設定かネットワークを確認して。",
                        "zh": f"工具执行时出了点问题：{error_text}",
                        "tone": "困惑",
                        "portrait": "张嘴疑问",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _build_vision_unsupported_reply() -> ChatReply:
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "今のモデルでは画像を見られないみたい。画面の内容は勝手に想像しないでおくね。",
                        "zh": "当前模型或接口似乎不支持图片输入。我不会猜屏幕内容，请换成支持视觉的模型后再试。",
                        "tone": "困惑",
                        "portrait": "张嘴疑问",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _summarize_tool_results(results: list[ToolExecutionResult]) -> str:
    parts: list[str] = []
    for result in results:
        if isinstance(result.content, dict):
            if isinstance(result.content.get("reminder"), dict):
                reminder = result.content["reminder"]
                text = reminder.get("text", "")
                trigger_at = reminder.get("trigger_at", "")
                parts.append(f"提醒「{text}」已设置在 {trigger_at}。")
            elif isinstance(result.content.get("task"), dict):
                task = result.content["task"]
                parts.append(f"待办「{task.get('text', '')}」已更新。")
            elif isinstance(result.content.get("memory"), dict):
                memory = result.content["memory"]
                parts.append(f"记忆「{memory.get('content', '')}」已更新。")
            elif result.tool_name == "open_url":
                parts.append(f"网页已打开：{result.content.get('url', '')}。")
            elif result.tool_name == "browser_open_url":
                parts.append(f"受控浏览器已打开：{result.content.get('url', '')}。")
            elif result.tool_name == "browser_get_content":
                title = result.content.get("title", "")
                text = result.content.get("text", "")
                parts.append(f"网页内容已读取：{title}。{str(text)[:120]}")
            elif result.tool_name == "browser_scroll":
                parts.append(f"页面已滚动到 Y={result.content.get('scroll_y', '')}。")
            elif result.tool_name == "browser_click":
                parts.append(f"页面元素已点击：{result.content.get('selector', '')}。")
            elif result.tool_name == "browser_get_state":
                parts.append(
                    f"当前网页：{result.content.get('title', '')}，"
                    f"滚动位置 Y={result.content.get('scroll_y', '')}。"
                )
            elif result.tool_name == "open_local_folder":
                parts.append(f"文件夹已打开：{result.content.get('path', '')}。")
            elif result.tool_name == "read_note":
                parts.append(f"笔记「{result.content.get('name', '')}」已读取。")
            elif result.tool_name == "write_note":
                parts.append(f"笔记「{result.content.get('name', '')}」已保存。")
            else:
                parts.append(f"{result.tool_name} 已完成。")
        else:
            parts.append(f"{result.tool_name} 已完成。")
    return " ".join(part for part in parts if part).strip()


def _build_event_messages(event: AgentEvent) -> list[ChatMessage]:
    text = _format_event_for_model(event)
    image_parts = _build_event_screen_context_image_parts(event.payload)
    if not image_parts:
        return [{"role": "user", "content": text}]

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
                *image_parts,
            ],
        }
    ]


def _build_event_screen_context_image_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    screen_contexts = payload.get("screen_contexts")
    image_parts: list[dict[str, Any]] = []
    if isinstance(screen_contexts, list):
        for screen_context in screen_contexts:
            if isinstance(screen_context, dict):
                image_part = _build_screen_context_image_part(screen_context)
                if image_part is not None:
                    image_parts.append(image_part)
    if image_parts:
        return image_parts

    screen_context = payload.get("screen_context")
    if isinstance(screen_context, dict):
        image_part = _build_screen_context_image_part(screen_context)
        if image_part is not None:
            return [image_part]
    return []


def _build_screen_context_image_part(screen_context: dict[str, Any]) -> dict[str, Any] | None:
    data_url = screen_context.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": data_url,
            "detail": "low",
        },
    }


def _format_event_for_model(event: AgentEvent) -> str:
    instruction = (
        "主动事件如下，请基于屏幕或事件内容生成要直接说给用户听的自然搭话："
        if event.type == "proactive_check"
        else "主动事件如下，请生成要直接说给用户听的提醒："
    )
    return instruction + "\n" + json.dumps(
        _redact_event_for_model(event),
        ensure_ascii=False,
        indent=2,
    )


def _redact_event_for_model(event: AgentEvent) -> dict[str, Any]:
    payload = dict(event.payload)
    screen_context = payload.get("screen_context")
    if isinstance(screen_context, dict):
        payload["screen_context"] = _redact_screen_context_for_model(screen_context)
    screen_contexts = payload.get("screen_contexts")
    if isinstance(screen_contexts, list):
        payload["screen_contexts"] = [
            _redact_screen_context_for_model(screen_context)
            if isinstance(screen_context, dict)
            else screen_context
            for screen_context in screen_contexts
        ]
    return {
        "type": event.type,
        "payload": payload,
    }


def _redact_screen_context_for_model(screen_context: dict[str, Any]) -> dict[str, Any]:
    redacted_context = dict(screen_context)
    if redacted_context.pop("data_url", None):
        redacted_context["image_attached"] = True
    return redacted_context


def _build_proactive_vision_unsupported_reply() -> ChatReply:
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "今のモデルでは画面までは見られないみたい。勝手に想像しないで、少しだけ休憩の合図にしておくね。",
                        "zh": "当前模型似乎还不能看屏幕。我不会乱猜，就先轻轻提醒你休息一下。",
                        "tone": "提醒",
                        "portrait": "伸手命令",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _build_proactive_tool_loop_rules() -> str:
    return "\n".join(
        [
            "- 这是主动检查事件，不是用户直接发来的请求；整体保持低打扰。",
            "- 请用角色语气自然搭话、提问或提醒用户。",
            build_proactive_rules(include_tool_rules=True),
        ]
    )
