from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot


BROWSER_TOOL_TIMEOUT_SECONDS = 25
BROWSER_SCREENSHOT_MAX_EDGE = 1280
BROWSER_SCREENSHOT_JPEG_QUALITY = 70


@dataclass
class _BrowserToolRequest:
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str = ""


class BrowserToolBridge(QObject):
    """把后台工具调用安全转发到 Qt 主线程中的浏览器控制器。"""

    requested = Signal(str, dict, object)

    def __init__(self, controller: "BrowserController", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.requested.connect(self._handle_request)

    def execute_browser_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if QThread.currentThread() == self.thread():
            raise RuntimeError("浏览器工具需要从后台工作线程调用，不能阻塞 UI 主线程。")

        request = _BrowserToolRequest()
        self.requested.emit(name, arguments, request)
        if not request.done.wait(BROWSER_TOOL_TIMEOUT_SECONDS):
            raise TimeoutError("浏览器操作超时。")
        if request.error:
            raise RuntimeError(request.error)
        if request.result is None:
            raise RuntimeError("浏览器操作没有返回结果。")
        return request.result

    @Slot(str, dict, object)
    def _handle_request(self, name: str, arguments: dict[str, Any], request: object) -> None:
        if not isinstance(request, _BrowserToolRequest):
            return

        def complete(result: dict[str, Any] | None = None, error: str = "") -> None:
            request.result = result
            request.error = error
            request.done.set()

        self.controller.execute(name, arguments, complete)


class BrowserController(QObject):
    """Sakura 托管的单窗口浏览器控制器。"""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._view: Any | None = None
        self._loading = False
        self._has_page = False

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        complete: Callable[[dict[str, Any] | None, str], None],
    ) -> None:
        try:
            if name == "browser_open_url":
                self.open_url(str(arguments.get("url", "")), complete)
            elif name == "browser_get_content":
                self.get_content(int(arguments.get("max_chars", 6000)), complete)
            elif name == "browser_scroll":
                self.scroll(
                    str(arguments.get("direction", "")),
                    int(arguments.get("amount", 800)),
                    complete,
                )
            elif name == "browser_click":
                self.click(str(arguments.get("selector", "")), complete)
            elif name == "browser_get_state":
                self.get_state(complete)
            else:
                complete(None, f"未知浏览器工具：{name}")
        except Exception as exc:
            complete(None, str(exc))

    def open_url(
        self,
        url: str,
        complete: Callable[[dict[str, Any] | None, str], None],
    ) -> None:
        view = self._ensure_view()
        self._loading = True

        def on_finished(ok: bool) -> None:
            try:
                view.loadFinished.disconnect(on_finished)
            except (RuntimeError, TypeError):
                pass
            self._loading = False
            self._has_page = ok
            if not ok:
                complete(None, f"网页加载失败：{url}")
                return

            open_result = {
                "url": view.url().toString(),
                "title": view.title(),
                "opened": True,
                "loaded": True,
            }

            def with_content(content: dict[str, Any] | None, error: str) -> None:
                if isinstance(content, dict):
                    open_result.update(content)
                    open_result["opened"] = True
                    open_result["loaded"] = True
                if error:
                    open_result["content_warning"] = error
                complete(open_result, "")

            self.get_content(3000, with_content)

        view.loadFinished.connect(on_finished)
        view.setWindowTitle("Sakura 受控浏览器")
        view.resize(1120, 760)
        view.show()
        view.raise_()
        view.load(QUrl(url))

    def get_content(
        self,
        max_chars: int,
        complete: Callable[[dict[str, Any] | None, str], None],
    ) -> None:
        view = self._require_loaded_view()
        script = f"""
JSON.stringify((() => {{
  const bodyText = document.body ? document.body.innerText : "";
  const text = bodyText.replace(/[ \\t]+/g, " ").replace(/\\n{{3,}}/g, "\\n\\n").trim();
  const links = Array.from(document.querySelectorAll("a[href]"))
    .map((link) => ({{
      text: (link.innerText || link.getAttribute("aria-label") || "").trim().slice(0, 160),
      href: link.href
    }}))
    .filter((link) => link.href)
    .slice(0, 20);
  return {{
    url: location.href,
    title: document.title,
    text: text.slice(0, {max_chars}),
    links
  }};
}})())
"""

        def with_plain_text(result: dict[str, Any] | None, error: str) -> None:
            if error and result is None:
                result = {
                    "url": view.url().toString(),
                    "title": view.title(),
                    "text": "",
                    "links": [],
                    "extract_warning": error,
                }

            def on_plain_text(value: str) -> None:
                content = result or {
                    "url": view.url().toString(),
                    "title": view.title(),
                    "text": "",
                    "links": [],
                }
                fallback_text = _normalize_page_text(value)
                current_text = str(content.get("text") or "").strip()
                if fallback_text and len(fallback_text) > len(current_text):
                    content["text"] = fallback_text[:max_chars]
                content.setdefault("url", view.url().toString())
                content.setdefault("title", view.title())
                content.setdefault("links", [])
                content["text_length"] = len(str(content.get("text") or ""))
                links = content.get("links")
                content["links_count"] = len(links) if isinstance(links, list) else 0
                if content["text_length"] == 0:
                    screenshot = _capture_view_screenshot_data_url(view)
                    if screenshot:
                        content["screenshot_data_url"] = screenshot
                        content["screenshot_fallback"] = True
                complete(content, "")

            view.page().toPlainText(on_plain_text)

        self._run_js_dict(view, script, with_plain_text)

    def scroll(
        self,
        direction: str,
        amount: int,
        complete: Callable[[dict[str, Any] | None, str], None],
    ) -> None:
        view = self._require_loaded_view()
        delta = -amount if direction == "up" else amount
        script = f"""
JSON.stringify((() => {{
  window.scrollBy(0, {delta});
  return {{
    url: location.href,
    title: document.title,
    scroll_x: window.scrollX,
    scroll_y: window.scrollY,
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight,
    page_width: Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0),
    page_height: Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)
  }};
}})())
"""
        self._run_js_dict(view, script, complete)

    def click(
        self,
        selector: str,
        complete: Callable[[dict[str, Any] | None, str], None],
    ) -> None:
        view = self._require_loaded_view()
        encoded_selector = json.dumps(selector)
        script = f"""
JSON.stringify((() => {{
  let element = null;
  try {{
    element = document.querySelector({encoded_selector});
  }} catch (error) {{
    return {{ ok: false, error: `CSS selector 无效：${{error.message}}` }};
  }}
  if (!element) {{
    return {{ ok: false, error: "未找到匹配 selector 的元素。" }};
  }}
  element.scrollIntoView({{ block: "center", inline: "center" }});
  element.click();
  return {{
    ok: true,
    url: location.href,
    title: document.title,
    selector: {encoded_selector},
    clicked_text: (element.innerText || element.getAttribute("aria-label") || "").trim().slice(0, 160)
  }};
}})())
"""

        def after_click(result: dict[str, Any] | None, error: str) -> None:
            if error:
                complete(None, error)
                return
            if not isinstance(result, dict) or not result.get("ok"):
                message = result.get("error") if isinstance(result, dict) else "点击失败。"
                complete(None, str(message or "点击失败。"))
                return
            complete(result, "")

        self._run_js_dict(view, script, after_click)

    def get_state(self, complete: Callable[[dict[str, Any] | None, str], None]) -> None:
        view = self._require_loaded_view()
        script = """
JSON.stringify((() => ({
  url: location.href,
  title: document.title,
  ready_state: document.readyState,
  scroll_x: window.scrollX,
  scroll_y: window.scrollY,
  viewport_width: window.innerWidth,
  viewport_height: window.innerHeight,
  page_width: Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0),
  page_height: Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)
}))())
"""

        def with_loading(result: dict[str, Any] | None, error: str) -> None:
            if error:
                complete(None, error)
                return
            result = result or {}
            result["loading"] = self._loading
            complete(result, "")

        self._run_js_dict(view, script, with_loading)

    def _ensure_view(self) -> Any:
        if self._view is not None:
            return self._view
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except ImportError as exc:
            raise RuntimeError(
                "当前环境缺少 PySide6.QtWebEngineWidgets。请安装 PySide6-Addons 后再使用受控浏览器。"
            ) from exc

        view = QWebEngineView()
        view.destroyed.connect(self._clear_view)
        self._view = view
        return view

    def _require_loaded_view(self) -> Any:
        view = self._ensure_view()
        if not self._has_page or not view.url().isValid() or not view.url().toString():
            raise RuntimeError("受控浏览器尚未打开网页。")
        return view

    def _run_js_dict(
        self,
        view: Any,
        script: str,
        complete: Callable[[dict[str, Any] | None, str], None],
    ) -> None:
        def on_result(value: Any) -> None:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as exc:
                    complete(None, f"网页脚本返回的 JSON 无法解析：{exc}")
                    return
            if not isinstance(value, dict):
                complete(None, f"网页脚本没有返回有效对象：{type(value).__name__}")
                return
            complete(value, "")

        view.page().runJavaScript(script, on_result)

    @Slot()
    def _clear_view(self) -> None:
        self._view = None
        self._has_page = False
        self._loading = False


def _normalize_page_text(text: str) -> str:
    """压缩网页纯文本中的多余空白，避免返回整页空行。"""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    compact_lines = [line for line in lines if line]
    return "\n".join(compact_lines)


def _capture_view_screenshot_data_url(view: Any) -> str:
    """截取浏览器当前视口，用于 DOM 文本为空时给视觉模型兜底。"""
    from PySide6.QtCore import QBuffer, QIODevice, Qt

    pixmap = view.grab()
    if pixmap.isNull():
        return ""

    longest_edge = max(pixmap.width(), pixmap.height())
    if longest_edge > BROWSER_SCREENSHOT_MAX_EDGE:
        pixmap = pixmap.scaled(
            BROWSER_SCREENSHOT_MAX_EDGE,
            BROWSER_SCREENSHOT_MAX_EDGE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pixmap.toImage().save(buffer, "JPEG", BROWSER_SCREENSHOT_JPEG_QUALITY):
        return ""

    import base64

    encoded = base64.b64encode(bytes(buffer.data())).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
