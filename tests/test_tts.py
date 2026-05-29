from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

if importlib.util.find_spec("PySide6") is None:
    pyside_module = types.ModuleType("PySide6")
    qtcore_module = types.ModuleType("PySide6.QtCore")
    qtmultimedia_module = types.ModuleType("PySide6.QtMultimedia")

    class QObject:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class QTimer:
        @staticmethod
        def singleShot(*_args: object, **_kwargs: object) -> None:
            pass

    class QUrl:
        @staticmethod
        def fromLocalFile(path: str) -> str:
            return path

    class Signal:
        def __init__(self, *_args: object) -> None:
            pass

        def connect(self, *_args: object, **_kwargs: object) -> None:
            pass

        def emit(self, *_args: object, **_kwargs: object) -> None:
            pass

    def Slot(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        def decorator(function):  # type: ignore[no-untyped-def]
            return function

        return decorator

    class QAudioOutput:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class QMediaPlayer:
        class MediaStatus:
            EndOfMedia = object()

        class PlaybackState:
            PlayingState = object()

        class Error:
            pass

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    qtcore_module.QObject = QObject
    qtcore_module.QTimer = QTimer
    qtcore_module.QUrl = QUrl
    qtcore_module.Signal = Signal
    qtcore_module.Slot = Slot
    qtmultimedia_module.QAudioOutput = QAudioOutput
    qtmultimedia_module.QMediaPlayer = QMediaPlayer
    sys.modules["PySide6"] = pyside_module
    sys.modules["PySide6.QtCore"] = qtcore_module
    sys.modules["PySide6.QtMultimedia"] = qtmultimedia_module

from app.tts import _load_tone_references, _resolve_request_text_lang


def test_tts_mixed_japanese_and_english_uses_auto_lang() -> None:
    text = "Steamを開いているんだね。Muse Dash…楽しそうなゲーム。"

    assert _resolve_request_text_lang(text, "ja") == "auto"


def test_tts_plain_japanese_keeps_configured_lang() -> None:
    text = "でも私、初めて君に会った時、思ったよ。"

    assert _resolve_request_text_lang(text, "ja") == "ja"


def test_tts_explicit_english_lang_is_not_overridden() -> None:
    text = "Steam is open."

    assert _resolve_request_text_lang(text, "en") == "en"


def test_tts_yue_mixed_english_uses_auto_yue() -> None:
    text = "Steam 打开咗。"

    assert _resolve_request_text_lang(text, "all_yue") == "auto_yue"


def test_tone_references_load_four_part_rows_only() -> None:
    ref_path = Path("characters/sakura/voice/refs/ref.txt")
    rows = [line for line in ref_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    references = _load_tone_references(ref_path, Path("characters/sakura"))

    assert all(len(row.split("|")) == 4 for row in rows)
    assert references
    assert all("|" not in reference.ref_text for items in references.values() for reference in items)
    assert all(reference.ref_audio_path.exists() for items in references.values() for reference in items)
