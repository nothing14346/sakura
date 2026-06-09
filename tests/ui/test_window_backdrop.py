from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QColor  # noqa: E402

from app.ui import window_backdrop as wb  # noqa: E402


def _qt_app_or_skip():  # type: ignore[no-untyped-def]
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    return qtwidgets.QApplication.instance() or qtwidgets.QApplication([])


def test_gradient_color_packs_abgr() -> None:
    # QColor(r,g,b,a) → 亚克力 GradientColor 0xAABBGGRR。
    assert wb._gradient_color(QColor(0x12, 0x34, 0x56, 0x78)) == 0x78563412


def test_create_window_backdrop_acrylic_on_win11(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(wb.sys, "platform", "win32")
    monkeypatch.setattr(wb, "_windows_build", lambda: 22631)
    backdrop = wb.create_window_backdrop()
    assert isinstance(backdrop, wb.WindowsAcrylicBackdrop)
    assert backdrop.supports_native_blur() is True


def test_create_window_backdrop_acrylic_on_win10(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(wb.sys, "platform", "win32")
    monkeypatch.setattr(wb, "_windows_build", lambda: 19045)
    assert isinstance(wb.create_window_backdrop(), wb.WindowsAcrylicBackdrop)


def test_create_window_backdrop_fallback_on_old_windows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(wb.sys, "platform", "win32")
    monkeypatch.setattr(wb, "_windows_build", lambda: 9600)  # Win8.1
    backdrop = wb.create_window_backdrop()
    assert isinstance(backdrop, wb.FallbackTintBackdrop)
    assert backdrop.supports_native_blur() is False


def test_create_window_backdrop_fallback_on_non_windows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(wb.sys, "platform", "linux")
    assert isinstance(wb.create_window_backdrop(), wb.FallbackTintBackdrop)


def test_backdrops_apply_remove_do_not_raise() -> None:
    from PySide6.QtWidgets import QWidget

    _qt_app_or_skip()
    widget = QWidget()
    widget.resize(120, 52)
    tint = QColor(255, 255, 255, 40)

    # 任一 backdrop 的 apply/remove 都不应抛异常（失败时内部 try/except 降级）。
    for backdrop in (
        wb.FallbackTintBackdrop(),
        wb.WindowsAcrylicBackdrop(rounded=False),
        wb.WindowsAcrylicBackdrop(rounded=True),
    ):
        backdrop.apply(widget, tint)
        backdrop.remove(widget)

    widget.deleteLater()


class _RecordingBackdrop:
    def __init__(self) -> None:
        self.applied = 0
        self.removed = 0

    def apply(self, window, tint) -> None:  # type: ignore[no-untyped-def]
        self.applied += 1

    def remove(self, window) -> None:  # type: ignore[no-untyped-def]
        self.removed += 1

    def supports_native_blur(self) -> bool:
        return False


def test_acrylic_card_window_flags_reparent_and_backdrop() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    app = _qt_app_or_skip()
    from app.ui.acrylic_card_window import AcrylicCardWindow

    content = QWidget()
    backdrop = _RecordingBackdrop()
    card = AcrylicCardWindow(content, activatable=False, backdrop=backdrop)

    flags = card.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.Tool
    assert card.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    # 内容控件被 reparent 进卡片窗口。
    assert content.parentWidget() is card

    card.set_theme("/* qss */", QColor(255, 255, 255, 40))  # 未显示时不应崩
    card.show()
    app.processEvents()
    # showEvent 后应施加一次背景模糊。
    assert backdrop.applied >= 1

    card.deleteLater()
