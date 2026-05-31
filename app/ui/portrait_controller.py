from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEasingCurve, QObject, QParallelAnimationGroup, QPropertyAnimation, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QMessageBox, QWidget

from app.character_loader import CharacterProfile
from app.chat_reply import ChatSegment
from app.portrait_utils import should_crossfade_portrait


PORTRAIT_TRANSITION_MS = 1400


class PortraitController(QObject):
    """负责立绘资源缓存、分段表情切换和淡入淡出动画。"""

    def __init__(
        self,
        *,
        profile: CharacterProfile,
        parent_widget: QWidget,
        main_label: QLabel,
        transition_label: QLabel,
        main_opacity_effect: QGraphicsOpacityEffect,
        transition_opacity_effect: QGraphicsOpacityEffect,
        stage_size: tuple[int, int],
        relayout: Callable[[], None],
        raise_foreground: Callable[[], None],
        on_portrait_changed: Callable[[QPixmap], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.parent_widget = parent_widget
        self.main_label = main_label
        self.transition_label = transition_label
        self.main_opacity_effect = main_opacity_effect
        self.transition_opacity_effect = transition_opacity_effect
        self.stage_size = stage_size
        self._relayout = relayout
        self._raise_foreground = raise_foreground
        self._on_portrait_changed = on_portrait_changed

        self.current_path = profile.default_portrait_path
        self.pixmap_cache: dict[Path, QPixmap] = {}
        self.pixmap = self.load_portrait()
        self.transition_animation: QParallelAnimationGroup | None = None
        self.transition_id = 0

    def apply_current(self) -> None:
        self._stop_transition()
        if self.pixmap.isNull():
            self.parent_widget.resize(*self.stage_size)
            return

        self._apply_pixmap_to_label(self.main_label, self.pixmap)
        self.parent_widget.resize(*self.stage_size)
        self._relayout()

    def set_profile(self, profile: CharacterProfile) -> QPixmap:
        self.profile = profile
        self.current_path = profile.default_portrait_path
        self.pixmap = self.load_portrait()
        self.apply_current()
        self._on_portrait_changed(self.pixmap)
        return self.pixmap

    def preload_for_segment(self, segment: ChatSegment) -> None:
        next_portrait_path = self.profile.portrait_for_segment(segment.portrait, segment.tone)
        if next_portrait_path not in self.pixmap_cache:
            self.load_portrait(next_portrait_path)

    def apply_for_segment(self, segment: ChatSegment) -> None:
        next_portrait_path = self.profile.portrait_for_segment(segment.portrait, segment.tone)
        if next_portrait_path == self.current_path:
            return

        should_crossfade = should_crossfade_portrait(self.current_path, next_portrait_path)
        next_pixmap = self.load_portrait(next_portrait_path)
        self.current_path = next_portrait_path
        if should_crossfade:
            self._crossfade(next_pixmap)
        else:
            self.pixmap = next_pixmap
            self.apply_current()
        self._on_portrait_changed(self.pixmap)

    def load_portrait(self, portrait_path: Path | None = None) -> QPixmap:
        target_path = portrait_path or self.current_path
        cached = self.pixmap_cache.get(target_path)
        if cached is not None:
            return cached

        pixmap = QPixmap(str(target_path))
        if pixmap.isNull():
            QMessageBox.critical(
                self.parent_widget,
                "立绘加载失败",
                f"无法加载立绘：{target_path}",
            )
        self.pixmap_cache[target_path] = pixmap
        return pixmap

    def _apply_pixmap_to_label(self, label: QLabel, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            560,
            570,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.resize(scaled.size())

    def _crossfade(self, next_pixmap: QPixmap) -> None:
        self._stop_transition(finish_current=True)
        self.pixmap = next_pixmap
        if self.pixmap.isNull():
            self.apply_current()
            return

        self._apply_pixmap_to_label(self.transition_label, self.pixmap)
        self.parent_widget.resize(*self.stage_size)
        self._relayout()
        self.main_opacity_effect.setOpacity(1.0)
        self.transition_opacity_effect.setOpacity(0.0)
        self.transition_label.show()
        self.transition_label.raise_()
        self._raise_foreground()

        self.transition_id += 1
        transition_id = self.transition_id
        animation = QParallelAnimationGroup(self)

        fade_out = QPropertyAnimation(self.main_opacity_effect, b"opacity")
        fade_out.setDuration(PORTRAIT_TRANSITION_MS)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        fade_in = QPropertyAnimation(self.transition_opacity_effect, b"opacity")
        fade_in.setDuration(PORTRAIT_TRANSITION_MS)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)

        animation.addAnimation(fade_out)
        animation.addAnimation(fade_in)
        animation.finished.connect(lambda: self._finish_transition(transition_id))
        self.transition_animation = animation
        animation.start()

    def _stop_transition(self, finish_current: bool = False) -> None:
        if self.transition_animation is not None:
            self.transition_animation.stop()
            self.transition_animation.deleteLater()
            self.transition_animation = None
            self.transition_id += 1
        self.transition_label.hide()
        self.transition_label.clear()
        self.main_opacity_effect.setOpacity(1.0)
        self.transition_opacity_effect.setOpacity(0.0)
        if finish_current and not self.pixmap.isNull():
            self._apply_pixmap_to_label(self.main_label, self.pixmap)
            self.parent_widget.resize(*self.stage_size)
            self._relayout()

    def _finish_transition(self, transition_id: int) -> None:
        if transition_id != self.transition_id:
            return
        if self.transition_animation is not None:
            self.transition_animation.deleteLater()
            self.transition_animation = None
        self._apply_pixmap_to_label(self.main_label, self.pixmap)
        self.transition_label.hide()
        self.transition_label.clear()
        self.main_opacity_effect.setOpacity(1.0)
        self.transition_opacity_effect.setOpacity(0.0)
        self._relayout()
