from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from app.core.debug_log import debug_log


@runtime_checkable
class WindowBackdrop(Protocol):
    """跨平台窗口背景模糊能力接口。

    apply 把系统级背景模糊（如 Windows 亚克力）施加到一个**已显示**的顶层窗口，
    让窗口透明区透出并模糊背后的真实桌面。不支持的平台用降级实现，保证调用统一、不报错。
    """

    def apply(self, window: QWidget, tint: QColor) -> None: ...

    def remove(self, window: QWidget) -> None: ...

    def supports_native_blur(self) -> bool: ...


def create_window_backdrop() -> WindowBackdrop:
    """按当前平台与系统版本探测，返回最合适的背景模糊实现。"""
    if sys.platform == "win32":
        build = _windows_build()
        if build >= 17134:  # Windows 10 1803+ 起支持亚克力
            return WindowsAcrylicBackdrop(rounded=build >= 22000)
    # Mac/Linux/旧 Windows 本次只做降级占位，原生模糊以后再接。
    return FallbackTintBackdrop()


def _windows_build() -> int:
    try:
        return int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
    except Exception:
        return 0


class FallbackTintBackdrop:
    """无系统级模糊的平台（Mac/Linux/旧 Windows）降级占位。

    不做真模糊：卡片自身的半透明 QSS 背景即降级效果，这里只作为接口占位，
    apply/remove 为空操作，保证上层调用统一、不报错。
    """

    def apply(self, window: QWidget, tint: QColor) -> None:
        del window, tint

    def remove(self, window: QWidget) -> None:
        del window

    def supports_native_blur(self) -> bool:
        return False


class WindowsAcrylicBackdrop:
    """Windows 亚克力背景模糊（DWM 合成器实时模糊窗口背后的真实桌面）。

    主路径：user32.SetWindowCompositionAttribute + ACCENT_ENABLE_ACRYLICBLURBEHIND，
    Win10 1803+ / Win11 通用；Win11 额外用 DwmSetWindowAttribute 设原生圆角。
    任何调用失败都静默降级（不影响窗口正常显示）。
    """

    _WCA_ACCENT_POLICY = 19
    _ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
    _ACCENT_DISABLED = 0
    _DWMWA_WINDOW_CORNER_PREFERENCE = 33
    _DWMWCP_ROUND = 2

    def __init__(self, *, rounded: bool) -> None:
        self._rounded = rounded

    def apply(self, window: QWidget, tint: QColor) -> None:
        # 亚克力是 DWM 窗口级合成，无视 Qt setMask/SetWindowRgn，圆角只能交给 DWM 原生圆角。
        try:
            hwnd = int(window.winId())
            self._set_accent(hwnd, self._ACCENT_ENABLE_ACRYLICBLURBEHIND, tint)
            if self._rounded:
                self._set_round_corners(hwnd)
        except Exception as exc:  # noqa: BLE001
            debug_log("UI", "Windows 亚克力背景应用失败，降级为半透明", {"error": str(exc)})

    def remove(self, window: QWidget) -> None:
        try:
            hwnd = int(window.winId())
            self._set_accent(hwnd, self._ACCENT_DISABLED, QColor(0, 0, 0, 0))
        except Exception as exc:  # noqa: BLE001
            debug_log("UI", "Windows 亚克力背景移除失败", {"error": str(exc)})

    def supports_native_blur(self) -> bool:
        return True

    def _set_accent(self, hwnd: int, accent_state: int, tint: QColor) -> None:
        import ctypes

        class ACCENT_POLICY(ctypes.Structure):
            _fields_ = [
                ("AccentState", ctypes.c_int),
                ("AccentFlags", ctypes.c_int),
                ("GradientColor", ctypes.c_uint),
                ("AnimationId", ctypes.c_int),
            ]

        class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
            _fields_ = [
                ("Attribute", ctypes.c_int),
                ("Data", ctypes.POINTER(ACCENT_POLICY)),
                ("SizeOfData", ctypes.c_size_t),
            ]

        accent = ACCENT_POLICY()
        accent.AccentState = accent_state
        accent.AccentFlags = 0
        accent.GradientColor = _gradient_color(tint)
        accent.AnimationId = 0

        data = WINDOWCOMPOSITIONATTRIBDATA()
        data.Attribute = self._WCA_ACCENT_POLICY
        data.SizeOfData = ctypes.sizeof(accent)
        data.Data = ctypes.pointer(accent)

        ctypes.windll.user32.SetWindowCompositionAttribute(hwnd, ctypes.pointer(data))

    def _set_round_corners(self, hwnd: int) -> None:
        import ctypes

        preference = ctypes.c_int(self._DWMWCP_ROUND)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            self._DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(preference),
            ctypes.sizeof(preference),
        )


def _gradient_color(tint: QColor) -> int:
    """QColor → 亚克力 GradientColor 的 0xAABBGGRR 整数（磨砂底色 + alpha）。"""
    return (
        (tint.alpha() << 24)
        | (tint.blue() << 16)
        | (tint.green() << 8)
        | tint.red()
    )
