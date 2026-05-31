from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.api_client import ApiSettings
from app.portrait_utils import portrait_kind_key, should_crossfade_portrait
from app.proactive_care import ProactiveCareSettings
from app.screen_observation import ScreenObservation
from app.tts import GPTSoVITSTTSSettings


def test_portrait_kind_key_uses_filename_suffix_group() -> None:
    assert portrait_kind_key(Path("portraits/A020.png")) == "A"
    assert portrait_kind_key(Path("portraits/B180.png")) == "B"
    assert portrait_kind_key(Path("portraits/I010.png")) == "I"


def test_same_portrait_kind_crossfades_when_file_changes() -> None:
    assert should_crossfade_portrait(
        Path("portraits/A020.png"),
        Path("portraits/A150.png"),
    )
    assert should_crossfade_portrait(
        Path("portraits/I010.png"),
        Path("portraits/I180.png"),
    )


def test_different_portrait_kind_crossfades() -> None:
    assert should_crossfade_portrait(
        Path("portraits/A020.png"),
        Path("portraits/B180.png"),
    )


def test_same_portrait_file_does_not_crossfade() -> None:
    assert not should_crossfade_portrait(
        Path("portraits/A020.png"),
        Path("portraits/A020.png"),
    )


def test_pet_window_menu_keeps_only_allowed_checkable_switches() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    if not hasattr(qtwidgets, "QApplication") or not hasattr(qtwidgets, "QWidget"):
        pytest.skip("当前测试环境只提供了 PySide6 stub。")

    from app.pet_window import PetWindow, SUBTITLE_LANGUAGE_ZH

    QApplication = qtwidgets.QApplication
    QWidget = qtwidgets.QWidget
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.subtitle_language = SUBTITLE_LANGUAGE_ZH
    host.free_access_enabled = True
    host._toggle_chinese_subtitles = lambda _checked: None
    host._toggle_free_access = lambda _checked: None
    host.show_history = lambda: None
    host.show_settings = lambda: None

    menu = PetWindow._build_menu(host)  # type: ignore[arg-type]
    actions = [action for action in menu.actions() if not action.isSeparator()]
    texts = [action.text() for action in actions]
    checkable_texts = [action.text() for action in actions if action.isCheckable()]

    assert texts[0] == "隐藏至托盘"
    assert "启用模型视觉" not in texts
    assert "允许自主看屏幕" not in texts
    assert "自由访问权限" not in texts
    assert "显示中文字幕" in checkable_texts
    assert "完整访问权限" in checkable_texts
    assert len(checkable_texts) == 2

    menu.deleteLater()
    host.deleteLater()
    app.processEvents()


def test_settings_dialog_disables_proactive_intervals_when_screen_context_disabled() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    if not hasattr(qtwidgets, "QApplication"):
        pytest.skip("当前测试环境只提供了 PySide6 stub。")

    from app.settings_dialog import SettingsDialog

    QApplication = qtwidgets.QApplication
    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog(
        api_settings=ApiSettings(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
        ),
        tts_settings=_minimal_tts_settings(),
        base_dir=Path("."),
        proactive_care_settings=ProactiveCareSettings(
            screen_context_enabled=False,
            check_interval_minutes=20,
            cooldown_minutes=10,
            screen_context_batch_limit=6,
        ),
    )

    assert not dialog.proactive_check_interval_spin.isEnabled()
    assert not dialog.proactive_cooldown_spin.isEnabled()
    assert not dialog.proactive_batch_limit_spin.isEnabled()

    dialog.proactive_screen_context_enabled_check.setChecked(True)
    app.processEvents()
    assert dialog.proactive_check_interval_spin.isEnabled()
    assert dialog.proactive_cooldown_spin.isEnabled()
    assert dialog.proactive_batch_limit_spin.isEnabled()

    dialog.proactive_screen_context_enabled_check.setChecked(False)
    app.processEvents()
    assert not dialog.proactive_check_interval_spin.isEnabled()
    assert not dialog.proactive_cooldown_spin.isEnabled()
    assert not dialog.proactive_batch_limit_spin.isEnabled()

    dialog.deleteLater()
    app.processEvents()


def test_proactive_care_batches_screenshots_until_cooldown(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.pet_window as pet_window_module

    current_time = {"value": 0.0}
    captures: list[str] = []
    events = []
    history = []
    window = _build_minimal_proactive_window(
        screen_context_enabled=True,
        check_interval_minutes=1,
        cooldown_minutes=2,
        events=events,
        history=history,
    )

    def fake_capture(_window):  # type: ignore[no-untyped-def]
        index = len(captures) + 1
        data_url = f"data:image/jpeg;base64,{index}"
        captures.append(data_url)
        return ScreenObservation(
            data_url=data_url,
            width=800,
            height=600,
            captured_at=f"2026-05-30T12:0{index}:00+08:00",
            screen_name="DISPLAY1",
        )

    monkeypatch.setattr(pet_window_module.time, "perf_counter", lambda: current_time["value"])
    monkeypatch.setattr(pet_window_module, "capture_screen_observation", fake_capture)

    current_time["value"] = 60
    window._check_proactive_care()
    assert captures == ["data:image/jpeg;base64,1"]
    assert events == []

    current_time["value"] = 120
    window._check_proactive_care()
    assert captures == ["data:image/jpeg;base64,1", "data:image/jpeg;base64,2"]
    assert events == []

    current_time["value"] = 180
    window._check_proactive_care()

    assert captures == [
        "data:image/jpeg;base64,1",
        "data:image/jpeg;base64,2",
        "data:image/jpeg;base64,3",
    ]
    assert len(events) == 1
    assert [context["data_url"] for context in events[0].payload["screen_contexts"]] == captures
    assert events[0].payload["screen_context_count"] == 3
    assert history
    assert window.proactive_screen_contexts == []


def test_proactive_care_keeps_recent_screenshot_batch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.pet_window as pet_window_module

    captures = []
    window = _build_minimal_proactive_window(
        screen_context_enabled=True,
        check_interval_minutes=1,
        cooldown_minutes=10,
    )

    def fake_capture(_window):  # type: ignore[no-untyped-def]
        index = len(captures) + 1
        captures.append(index)
        return ScreenObservation(
            data_url=f"data:image/jpeg;base64,{index}",
            width=800,
            height=600,
            captured_at=f"2026-05-30T12:{index:02d}:00+08:00",
            screen_name="DISPLAY1",
        )

    monkeypatch.setattr(pet_window_module, "capture_screen_observation", fake_capture)

    for index in range(8):
        window._capture_proactive_screen_context(float(index * 60))

    assert len(window.proactive_screen_contexts) == 6
    assert window.proactive_screen_context_dropped_count == 2
    assert [context["data_url"] for context in window.proactive_screen_contexts] == [
        "data:image/jpeg;base64,3",
        "data:image/jpeg;base64,4",
        "data:image/jpeg;base64,5",
        "data:image/jpeg;base64,6",
        "data:image/jpeg;base64,7",
        "data:image/jpeg;base64,8",
    ]


def test_proactive_care_uses_configured_screenshot_batch_limit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.pet_window as pet_window_module

    captures = []
    window = _build_minimal_proactive_window(
        screen_context_enabled=True,
        check_interval_minutes=1,
        cooldown_minutes=10,
        screen_context_batch_limit=3,
    )

    def fake_capture(_window):  # type: ignore[no-untyped-def]
        index = len(captures) + 1
        captures.append(index)
        return ScreenObservation(
            data_url=f"data:image/jpeg;base64,{index}",
            width=800,
            height=600,
            captured_at=f"2026-05-30T12:{index:02d}:00+08:00",
            screen_name="DISPLAY1",
        )

    monkeypatch.setattr(pet_window_module, "capture_screen_observation", fake_capture)

    for index in range(5):
        window._capture_proactive_screen_context(float(index * 60))

    assert len(window.proactive_screen_contexts) == 3
    assert window.proactive_screen_context_dropped_count == 2
    assert [context["data_url"] for context in window.proactive_screen_contexts] == [
        "data:image/jpeg;base64,3",
        "data:image/jpeg;base64,4",
        "data:image/jpeg;base64,5",
    ]


def test_proactive_care_disabled_does_not_capture_or_send(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.pet_window as pet_window_module

    current_time = {"value": 600.0}
    events = []
    window = _build_minimal_proactive_window(
        screen_context_enabled=False,
        check_interval_minutes=1,
        cooldown_minutes=1,
        events=events,
    )

    def fail_capture(_window):  # type: ignore[no-untyped-def]
        raise AssertionError("关闭主动屏幕获取时不应该截图")

    monkeypatch.setattr(pet_window_module.time, "perf_counter", lambda: current_time["value"])
    monkeypatch.setattr(pet_window_module, "capture_screen_observation", fail_capture)

    window._check_proactive_care()

    assert events == []
    assert window.proactive_screen_contexts == []


def test_user_activity_clears_pending_proactive_screenshot_batch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.pet_window as pet_window_module

    window = _build_minimal_proactive_window(
        screen_context_enabled=True,
        check_interval_minutes=1,
        cooldown_minutes=10,
    )
    window.proactive_screen_contexts = [{"data_url": "data:image/jpeg;base64,old"}]
    window.proactive_screen_context_batch_started_at = 60
    window.last_proactive_screen_context_at = 60
    window.proactive_screen_context_dropped_count = 2
    monkeypatch.setattr(pet_window_module.time, "perf_counter", lambda: 300.0)

    window._mark_user_activity()

    assert window.last_user_activity_at == 300.0
    assert window.proactive_screen_contexts == []
    assert window.proactive_screen_context_batch_started_at is None
    assert window.last_proactive_screen_context_at is None
    assert window.proactive_screen_context_dropped_count == 0


class _DummyTextInput:
    def text(self) -> str:
        return ""


class _DummyTimer:
    def isActive(self) -> bool:
        return False


class _DummyButton:
    def setVisible(self, _visible: bool) -> None:
        pass


def _build_minimal_proactive_window(
    *,
    screen_context_enabled: bool,
    check_interval_minutes: int,
    cooldown_minutes: int,
    screen_context_batch_limit: int = 6,
    events=None,  # type: ignore[no-untyped-def]
    history=None,  # type: ignore[no-untyped-def]
):
    from app.pet_window import PetWindow

    class MinimalProactiveWindow:
        _can_run_proactive_care = PetWindow._can_run_proactive_care
        _check_proactive_care = PetWindow._check_proactive_care
        _should_capture_proactive_screen_context = (
            PetWindow._should_capture_proactive_screen_context
        )
        _capture_proactive_screen_context = PetWindow._capture_proactive_screen_context
        _should_send_proactive_care_batch = PetWindow._should_send_proactive_care_batch
        _build_proactive_care_event = PetWindow._build_proactive_care_event
        _proactive_screen_context_allowed = PetWindow._proactive_screen_context_allowed
        _clear_proactive_screen_context_batch = PetWindow._clear_proactive_screen_context_batch
        _mark_user_activity = PetWindow._mark_user_activity

    window = MinimalProactiveWindow()
    window.proactive_care_settings = ProactiveCareSettings(
        enabled=screen_context_enabled,
        screen_context_enabled=screen_context_enabled,
        check_interval_minutes=check_interval_minutes,
        cooldown_minutes=cooldown_minutes,
        screen_context_batch_limit=screen_context_batch_limit,
    )
    window.worker_thread = None
    window.active_reminder_id = None
    window.active_event_type = ""
    window.pending_tool_action = None
    window.pending_screen_observation_messages = None
    window.screen_observation_followup_in_progress = False
    window.active_interaction_id = ""
    window.input_edit = _DummyTextInput()
    window.speech_timer = _DummyTimer()
    window.current_segment_sequence_id = None
    window.current_segment_speech_done = True
    window.current_segment_tts_done = True
    window.last_user_activity_at = 0.0
    window.last_proactive_care_at = None
    window.last_proactive_screen_context_at = None
    window.proactive_screen_context_batch_started_at = None
    window.proactive_screen_contexts = []
    window.proactive_screen_context_dropped_count = 0
    window.confirm_action_button = _DummyButton()
    window.cancel_action_button = _DummyButton()
    captured_events = events if events is not None else []
    captured_history = history if history is not None else []
    window._run_event_worker = lambda event, reminder_id=None: captured_events.append(event)
    window._record_history = lambda *args: captured_history.append(args)
    return window


def _minimal_tts_settings() -> GPTSoVITSTTSSettings:
    return GPTSoVITSTTSSettings(
        enabled=False,
        api_url="http://127.0.0.1:9880/tts",
        ref_audio_path=Path("characters/sakura/voice/refs/tone_refs/00_中性_VO01_2785.ogg"),
        ref_text_path=Path("characters/sakura/voice/refs/ref.txt"),
        ref_text="テスト",
        ref_lang="ja",
        text_lang="ja",
        timeout_seconds=1,
    )
