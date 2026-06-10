from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.ui.window_backdrop import WindowBackdrop, create_window_backdrop


class AcrylicCardWindow(QWidget):
    """承载一块内容（对话气泡 / 输入栏）的独立无边框磨砂卡片窗口。

    窗口自身透明，由 WindowBackdrop 在背后施加系统级模糊（Windows 亚克力），
    圆角与卡片底色由内容控件的 QSS（border-radius/半透明背景）提供。
    不进任务栏；可选不抢焦点（气泡卡片不抢、输入栏卡片需可激活以便打字）。
    拖拽沿用内容控件已安装的事件过滤器转发给宿主 PetWindow，本类不另做转发。
    """

    def __init__(
        self,
        content: QWidget,
        *,
        activatable: bool,
        backdrop: WindowBackdrop | None = None,
        background_layer: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._backdrop = backdrop if backdrop is not None else create_window_backdrop()
        self._tint = QColor(255, 255, 255, 40)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if not activatable:
            # 气泡卡片：显示/点击不抢主窗口激活态，避免误触发托盘恢复等激活逻辑。
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(content)  # 将既有控件 reparent 进卡片窗口
        self.setLayout(layout)
        self._content = content

        # 可选软件模糊背景层：不进 layout，手动铺满并置于最底，由内容控件的透明背景透出。
        # 用于输入栏的软件截图模糊（替代 DWM 亚克力以实现大圆角），见 InputBlurBackground。
        self._background_layer = background_layer
        if background_layer is not None:
            background_layer.setParent(self)
            background_layer.lower()

    @property
    def content(self) -> QWidget:
        return self._content

    def set_theme(self, stylesheet: str, tint: QColor) -> None:
        """切主题时刷新卡片样式并按新底色重应用背景模糊（幂等）。"""
        self.setStyleSheet(stylesheet)
        self._tint = QColor(tint)
        if self.isVisible():
            self._backdrop.apply(self, self._tint)

    def supports_native_blur(self) -> bool:
        return self._backdrop.supports_native_blur()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        # 背景层始终铺满卡片窗口并保持在最底，跟随卡片尺寸变化。
        if self._background_layer is not None:
            self._background_layer.setGeometry(self.rect())
            self._background_layer.lower()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        # winId 在 show 之后才有效，此时施加亚克力最稳。圆角由 backdrop 内部的 DWM 原生圆角负责。
        self._backdrop.apply(self, self._tint)

    def hideEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().hideEvent(event)
        # macOS 上 hide() 销毁 native NSWindow，Content Swap 的 effect_view
        # 也被释放（PyObjC 引用虽在但底层 NSView 已 dead）。remove 清理引用，
        # 确保下次 showEvent 中 apply 能重建。
        self._backdrop.remove(self)
