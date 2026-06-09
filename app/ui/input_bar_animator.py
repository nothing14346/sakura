from __future__ import annotations

from typing import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QSequentialAnimationGroup,
    QTimer,
)
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

# hover 状态轮询间隔：靠回调判断光标是否落在桌宠任一窗口内。
HOVER_POLL_INTERVAL_MS = 180
# 浮现/收起淡入淡出时长，整体保持克制轻盈。
HOVER_ANIM_DURATION_MS = 220
# 发送反馈：输入栏内容做一次轻微"暗-亮"脉冲，每半程时长与最暗透明度。
SEND_PULSE_HALF_MS = 90
SEND_PULSE_DIP_OPACITY = 0.82


class InputBarAnimator(QObject):
    """输入栏卡片 hover 浮现/收起 + 发送反馈脉冲控制器。

    可见性 = hover_active OR pinned。pinned 由「输入框有焦点 / 输入框有文本 / 待确认动作面板」
    决定（不含桌宠讲话），保证用户正在输入或有未完成交互时输入栏不会被收起。

    浮现/收起对**整个输入栏卡片窗口**做 windowOpacity 淡入淡出——连同亚克力背景一起淡，
    避免内容淡完后亚克力背景才突然消失的生硬感。发送脉冲只作用输入栏内容自身的 opacity effect，
    与窗口级淡入淡出互不干扰。
    """

    def __init__(
        self,
        input_bar: QWidget,
        input_window: QWidget,
        is_pinned: Callable[[], bool],
        is_hover_active: Callable[[], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._input_bar = input_bar
        self._input_window = input_window
        self._is_pinned = is_pinned
        self._is_hover_active = is_hover_active

        # 仅用于发送脉冲：让输入栏内容做一次"暗-亮"，与窗口级 hover 淡入淡出分离。
        self._bar_effect = QGraphicsOpacityEffect(input_bar)
        self._bar_effect.setOpacity(1.0)
        input_bar.setGraphicsEffect(self._bar_effect)

        self._hover = False
        self._shown = False
        self._started = False
        self._anim: QPropertyAnimation | None = None
        self._send_anim: QSequentialAnimationGroup | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(HOVER_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._on_poll)

    # --- 对外接口 -----------------------------------------------------------
    def start(self) -> None:
        """窗口显示后启动 hover 轮询，并将输入栏落到当前静止态。"""
        if self._started:
            return
        self._started = True
        self._apply_resting_state()
        self._poll_timer.start()

    def sync(self) -> None:
        """外部 pinned 状态（如待确认动作出现）变化时，立即重算可见性。"""
        self._sync()

    # --- 内部逻辑 -----------------------------------------------------------
    def _on_poll(self) -> None:
        self._hover = bool(self._is_hover_active())
        self._sync()

    def _target_visible(self) -> bool:
        return self._hover or bool(self._is_pinned())

    def _sync(self) -> None:
        target = self._target_visible()
        if target == self._shown:
            return
        self._shown = target
        self._animate(target)

    def _animate(self, show: bool) -> None:
        if self._anim is not None:
            self._anim.stop()
            self._anim.deleteLater()
            self._anim = None
        if show:
            self._input_window.show()
        anim = QPropertyAnimation(self._input_window, b"windowOpacity")
        anim.setDuration(HOVER_ANIM_DURATION_MS)
        anim.setEndValue(1.0 if show else 0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        if not show:
            anim.finished.connect(self._on_hide_finished)
        anim.start()
        self._anim = anim

    def _on_hide_finished(self) -> None:
        # 整窗淡出结束后再隐藏，避免亚克力背景残留（若期间又变可见则跳过）。
        if not self._shown:
            self._input_window.hide()

    def _apply_resting_state(self) -> None:
        if self._anim is not None:
            self._anim.stop()
            self._anim.deleteLater()
            self._anim = None
        self._shown = self._target_visible()
        if self._shown:
            self._input_window.setWindowOpacity(1.0)
            self._input_window.show()
        else:
            self._input_window.setWindowOpacity(0.0)
            self._input_window.hide()

    def play_send_feedback(self) -> None:
        """发送时让输入栏内容做一次轻微"暗-亮"脉冲作为反馈。"""
        if self._send_anim is not None:
            self._send_anim.stop()
            self._send_anim.deleteLater()
        # dim 不设 startValue，从当前透明度起步，避免突跳。
        dim = QPropertyAnimation(self._bar_effect, b"opacity")
        dim.setDuration(SEND_PULSE_HALF_MS)
        dim.setEndValue(SEND_PULSE_DIP_OPACITY)
        dim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        restore = QPropertyAnimation(self._bar_effect, b"opacity")
        restore.setDuration(SEND_PULSE_HALF_MS)
        restore.setStartValue(SEND_PULSE_DIP_OPACITY)
        restore.setEndValue(1.0)
        restore.setEasingCurve(QEasingCurve.Type.InOutQuad)
        group = QSequentialAnimationGroup(self)
        group.addAnimation(dim)
        group.addAnimation(restore)
        group.start()
        self._send_anim = group
