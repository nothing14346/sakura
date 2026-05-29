from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agent.browser_tools import BrowserToolExecutor, BrowserTools
from app.agent.desktop_tools import NotesStore, open_local_folder, open_url
from app.agent.memory import MemoryStore
from app.agent.reminders import ReminderStore
from app.agent.tool_registry import Tool, ToolRegistry


def create_builtin_tool_registry(
    base_dir: Path,
    memory: MemoryStore | None = None,
    reminders: ReminderStore | None = None,
    browser_executor: BrowserToolExecutor | None = None,
) -> ToolRegistry:
    store = TodoStore(base_dir / "data" / "tasks.json")
    notes = NotesStore(base_dir / "data" / "notes")
    memory = memory or MemoryStore(base_dir / "data" / "memory.json")
    reminders = reminders or ReminderStore(base_dir / "data" / "reminders.json")
    browser = BrowserTools(browser_executor)
    return ToolRegistry(
        [
            Tool(
                name="get_current_time",
                description="获取当前本机时间和时区。",
                parameters={},
                handler=lambda _arguments: get_current_time(),
            ),
            Tool(
                name="add_todo",
                description="新增一条待办事项。",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "待办内容。"},
                    },
                    "required": ["text"],
                },
                handler=store.add_todo,
            ),
            Tool(
                name="list_todos",
                description="列出所有未完成待办事项。",
                parameters={},
                handler=store.list_todos,
            ),
            Tool(
                name="complete_todo",
                description="按 id 标记一条待办事项为完成。",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "待办 id。"},
                    },
                    "required": ["id"],
                },
                handler=store.complete_todo,
            ),
            Tool(
                name="add_reminder",
                description="创建一次性提醒。用户说“几分钟后/几秒后”这类相对时间时，必须优先使用 delay_seconds 或 delay_minutes，让程序计算触发时间；只有用户给出明确日期时间时才使用 trigger_at。repeat 第一版只支持 null 或省略。",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "提醒内容。"},
                        "trigger_at": {
                            "type": "string",
                            "description": "明确的提醒时间，本地时区 ISO 字符串。相对时间不要使用这个字段。",
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "从现在开始延迟多少秒触发。适合“30 秒后”等相对提醒。",
                        },
                        "delay_minutes": {
                            "type": "number",
                            "description": "从现在开始延迟多少分钟触发。适合“3 分钟后”等相对提醒。",
                        },
                        "repeat": {
                            "type": ["null"],
                            "description": "第一版只支持 null。",
                        },
                    },
                    "required": ["text"],
                },
                handler=reminders.add_reminder,
            ),
            Tool(
                name="list_reminders",
                description="列出未完成且未取消的一次性提醒。",
                parameters={},
                handler=reminders.list_reminders,
            ),
            Tool(
                name="cancel_reminder",
                description="按 id 取消一条未完成提醒。",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "提醒 id。"},
                    },
                    "required": ["id"],
                },
                handler=reminders.cancel_reminder,
            ),
            Tool(
                name="read_note",
                description="读取 data/notes/ 下的文本笔记。只能读取笔记名，不能读取任意路径。",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "笔记名，可省略 .txt 后缀。"},
                    },
                    "required": ["name"],
                },
                handler=notes.read_note,
            ),
            Tool(
                name="write_note",
                description="写入 data/notes/ 下的文本笔记。只能写入笔记名，不能写入任意路径。",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "笔记名，可省略 .txt 后缀。"},
                        "content": {"type": "string", "description": "笔记内容。"},
                    },
                    "required": ["name", "content"],
                },
                handler=notes.write_note,
            ),
            Tool(
                name="open_url",
                description="打开 http 或 https 网页。该工具会离开聊天窗口，需要用户确认后才能执行。",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "要打开的 http/https URL。"},
                    },
                    "required": ["url"],
                },
                handler=open_url,
                requires_confirmation=True,
            ),
            Tool(
                name="browser_open_url",
                description="在 Sakura 托管的受控浏览器窗口中打开 http 或 https 网页。需要用户确认后才能执行。",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "要打开的 http/https URL。"},
                    },
                    "required": ["url"],
                },
                handler=browser.open_url,
                requires_confirmation=True,
            ),
            Tool(
                name="browser_get_content",
                description="读取 Sakura 受控浏览器当前页面的 URL、标题、正文文本和主要链接。需要网页内容时优先使用该工具。",
                parameters={
                    "type": "object",
                    "properties": {
                        "max_chars": {
                            "type": "number",
                            "description": "正文最大返回字符数，默认 6000，最高 20000。",
                        },
                    },
                },
                handler=browser.get_content,
            ),
            Tool(
                name="browser_scroll",
                description="滚动 Sakura 受控浏览器当前页面。该工具会改变网页状态，需要用户确认后才能执行。",
                parameters={
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "description": "滚动方向，只能是 up 或 down。",
                        },
                        "amount": {
                            "type": "number",
                            "description": "滚动像素数，默认 800，最高 5000。",
                        },
                    },
                    "required": ["direction"],
                },
                handler=browser.scroll,
                requires_confirmation=True,
            ),
            Tool(
                name="browser_click",
                description="在 Sakura 受控浏览器当前页面点击第一个匹配 CSS selector 的元素。该工具会改变网页状态，需要用户确认后才能执行。",
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector，例如 button.submit 或 a[href*='login']。",
                        },
                    },
                    "required": ["selector"],
                },
                handler=browser.click,
                requires_confirmation=True,
            ),
            Tool(
                name="browser_get_state",
                description="获取 Sakura 受控浏览器当前页面 URL、标题、加载状态、滚动位置和页面尺寸。",
                parameters={},
                handler=browser.get_state,
            ),
            Tool(
                name="open_local_folder",
                description="打开已存在的本地文件夹。该工具会访问桌面环境，需要用户确认后才能执行。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要打开的本地文件夹路径。"},
                    },
                    "required": ["path"],
                },
                handler=open_local_folder,
                requires_confirmation=True,
            ),
            Tool(
                name="search_memory",
                description="搜索已确认长期记忆。适合在用户询问偏好、习惯、项目状态等已知信息时使用。",
                parameters={
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "搜索关键词，可为空。"},
                        "category": {
                            "type": "string",
                            "description": "可选分类：preference、project、habit、fact。",
                        },
                    },
                },
                handler=memory.search_memory,
            ),
            Tool(
                name="propose_memory_update",
                description="提出一条候选长期记忆。只在用户明确要求记住长期偏好、习惯、项目状态或事实时使用；不会直接写入正式记忆。",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "分类：preference、project、habit、fact。",
                        },
                        "content": {"type": "string", "description": "要记住的内容。"},
                        "reason": {"type": "string", "description": "为什么这条信息值得长期记住。"},
                    },
                    "required": ["category", "content"],
                },
                handler=memory.propose_memory_update,
            ),
            Tool(
                name="confirm_memory_update",
                description="在用户明确确认后，将候选记忆写入正式长期记忆。",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "候选记忆 id。"},
                    },
                    "required": ["id"],
                },
                handler=memory.confirm_memory_update,
            ),
            Tool(
                name="forget_memory",
                description="在用户明确要求忘记某条信息时，按 id 删除正式长期记忆。",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "正式记忆 id。"},
                    },
                    "required": ["id"],
                },
                handler=memory.forget_memory,
            ),
        ]
    )


def get_current_time() -> dict[str, str]:
    now = datetime.now().astimezone()
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": now.tzname() or "",
    }


class TodoStore:
    """以 JSON 文件保存轻量待办，供内部工具使用。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def add_todo(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = _required_text(arguments, "text")
        data = self._load()
        task = {
            "id": uuid.uuid4().hex[:8],
            "text": text,
            "created_at": _now_iso(),
            "completed_at": None,
        }
        data["tasks"].append(task)
        self._save(data)
        return {"task": task}

    def list_todos(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        data = self._load()
        tasks = [task for task in data["tasks"] if task.get("completed_at") is None]
        return {"tasks": tasks}

    def complete_todo(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task_id = _required_text(arguments, "id")
        data = self._load()
        for task in data["tasks"]:
            if task.get("id") == task_id:
                if task.get("completed_at") is None:
                    task["completed_at"] = _now_iso()
                    self._save(data)
                return {"task": task}
        raise ValueError(f"未找到待办：{task_id}")

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {"tasks": []}

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"待办文件不是有效 JSON：{self.path}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
            raise ValueError("待办文件格式无效，顶层必须是包含 tasks 列表的对象。")
        tasks = [task for task in data["tasks"] if isinstance(task, dict)]
        return {"tasks": tasks}

    def _save(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _required_text(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少必填参数：{key}")
    return value.strip()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
