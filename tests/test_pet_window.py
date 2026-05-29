from __future__ import annotations

from pathlib import Path

from app.portrait_utils import portrait_kind_key, should_crossfade_portrait


def test_portrait_kind_key_uses_filename_suffix_group() -> None:
    assert portrait_kind_key(Path("portraits/idle_soft_smile_A020.png")) == "A"
    assert portrait_kind_key(Path("portraits/confident_callout_B120.png")) == "B"
    assert portrait_kind_key(Path("portraits/gesture_invite_smile_I170.png")) == "I"


def test_same_portrait_kind_does_not_crossfade() -> None:
    assert not should_crossfade_portrait(
        Path("portraits/idle_soft_smile_A020.png"),
        Path("portraits/react_surprised_A110.png"),
    )
    assert not should_crossfade_portrait(
        Path("portraits/gesture_wave_neutral_I010.png"),
        Path("portraits/gesture_invite_smile_I170.png"),
    )


def test_different_portrait_kind_crossfades() -> None:
    assert should_crossfade_portrait(
        Path("portraits/idle_soft_smile_A020.png"),
        Path("portraits/confident_neutral_B010.png"),
    )
