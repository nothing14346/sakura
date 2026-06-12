from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, QTimer

from app.backchannel.classifier import RuleClassifier
from app.backchannel.models import BackchannelManifest
from app.backchannel.resolver import BackchannelChoice, TemplateResolver

if TYPE_CHECKING:
    from app.config.settings_service import BackchannelSettings

DisplayCallback = Callable[[BackchannelChoice], None]


class BackchannelController(QObject):
    """等待期接话调度:延迟 → 分类 → 匹配 → 显示;正式回复到达即取消。

    不直接依赖任何 UI 类:显示动作由 display 回调注入,宿主(PetWindow)
    决定怎么呈现。回调只应走轻量字幕/立绘路径——临时段绝不进入
    回复历史、聊天记录、LLM 上下文或分段播放队列。

    分类是纯规则(<10ms),直接跑在主线程 QTimer 回调里,无需线程。
    """

    def __init__(
        self,
        classifier: RuleClassifier,
        display: DisplayCallback,
        *,
        settings: "BackchannelSettings",
        rng: random.Random | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._classifier = classifier
        self._display = display
        self._settings = settings.normalized()
        self._rng = rng if rng is not None else random.Random()
        self._resolver: TemplateResolver | None = None
        self._pending_text = ""
        # armed 标志防住一个窄竞态:timeout 事件已入队但 cancel 先被处理。
        self._armed = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    # --- 对外接口 -----------------------------------------------------------
    def set_manifest(self, manifest: BackchannelManifest | None) -> None:
        """启动/切换角色时注入清单;None 表示该角色 opt-out,功能空转。"""
        self.cancel()
        if manifest:
            self._resolver = TemplateResolver(manifest, rng=self._rng)
        else:
            self._resolver = None

    def set_settings(self, settings: "BackchannelSettings") -> None:
        self._settings = settings.normalized()
        if not self._settings.active:
            self.cancel()

    def schedule(self, text: str) -> None:
        """用户消息已发送:启动接话延迟计时。延迟内回复到达则被 cancel 跳过。"""
        self.cancel()
        if not self._settings.active or self._resolver is None:
            return
        if not (text or "").strip():
            return
        # 触发概率:防罐头感的调节阀。
        if self._settings.probability < 1.0 and self._rng.random() >= self._settings.probability:
            return
        self._pending_text = text
        self._armed = True
        self._timer.start(self._settings.delay_ms)

    def cancel(self) -> None:
        """正式回复到达/请求失败/重新发送:放弃本轮接话。幂等。"""
        self._armed = False
        self._timer.stop()

    @property
    def is_pending(self) -> bool:
        return self._armed

    # --- 内部逻辑 -----------------------------------------------------------
    def _on_timeout(self) -> None:
        if not self._armed or self._resolver is None:
            return
        self._armed = False
        label = self._classifier.classify(self._pending_text)
        choice = self._resolver.resolve(label)
        if choice is not None:
            self._display(choice)
