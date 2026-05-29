from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlparse


DEFAULT_CONTENT_MAX_CHARS = 6000
MAX_CONTENT_CHARS = 20000
MAX_LINKS = 20


class BrowserToolExecutor(Protocol):
    """由 UI 层实现的受控浏览器执行器。"""

    def execute_browser_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行一个浏览器工具，并返回可 JSON 化的结果。"""


class BrowserTools:
    """Agent 内置浏览器工具的参数校验和结果整形。"""

    def __init__(self, executor: BrowserToolExecutor | None = None) -> None:
        self.executor = executor

    def open_url(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = _required_text(arguments, "url")
        _validate_http_url(url)
        return self._execute("browser_open_url", {"url": url})

    def get_content(self, arguments: dict[str, Any]) -> dict[str, Any]:
        max_chars = _optional_positive_int(arguments, "max_chars", DEFAULT_CONTENT_MAX_CHARS)
        max_chars = min(max_chars, MAX_CONTENT_CHARS)
        result = self._execute("browser_get_content", {"max_chars": max_chars})

        text = result.get("text")
        if isinstance(text, str) and len(text) > max_chars:
            result["text"] = text[:max_chars]

        links = result.get("links")
        if isinstance(links, list):
            result["links"] = [
                link
                for link in links[:MAX_LINKS]
                if isinstance(link, dict)
                and isinstance(link.get("href"), str)
                and isinstance(link.get("text", ""), str)
            ]
        return result

    def scroll(self, arguments: dict[str, Any]) -> dict[str, Any]:
        direction = _required_text(arguments, "direction").lower()
        if direction not in {"up", "down"}:
            raise ValueError("direction 只支持 up 或 down。")
        amount = _optional_positive_int(arguments, "amount", 800)
        return self._execute(
            "browser_scroll",
            {
                "direction": direction,
                "amount": min(amount, 5000),
            },
        )

    def click(self, arguments: dict[str, Any]) -> dict[str, Any]:
        selector = _required_text(arguments, "selector")
        return self._execute("browser_click", {"selector": selector})

    def get_state(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _ = arguments
        return self._execute("browser_get_state", {})

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.executor is None:
            raise RuntimeError("受控浏览器尚未初始化。请在桌宠窗口中使用浏览器工具。")
        result = self.executor.execute_browser_tool(name, arguments)
        if not isinstance(result, dict):
            raise RuntimeError("受控浏览器返回了无效结果。")
        return result


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL 只支持 http:// 或 https://。")


def _required_text(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少必填参数：{key}")
    return value.strip()


def _optional_positive_int(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是正数。")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{key} 必须大于 0。")
    return number
