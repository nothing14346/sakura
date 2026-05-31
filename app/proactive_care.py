from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.env_config import load_env_file, save_env_values


PROACTIVE_CARE_ENABLED_KEY = "PROACTIVE_CARE_ENABLED"
PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY = "PROACTIVE_SCREEN_CONTEXT_ENABLED"
PROACTIVE_CHECK_INTERVAL_MINUTES_KEY = "PROACTIVE_CHECK_INTERVAL_MINUTES"
PROACTIVE_COOLDOWN_MINUTES_KEY = "PROACTIVE_COOLDOWN_MINUTES"
PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY = "PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT"
PROACTIVE_DEFAULT_CHECK_INTERVAL_MINUTES = 20
PROACTIVE_DEFAULT_COOLDOWN_MINUTES = 10
PROACTIVE_DEFAULT_SCREEN_CONTEXT_BATCH_LIMIT = 6
PROACTIVE_MIN_CHECK_INTERVAL_MINUTES = 1
PROACTIVE_MAX_CHECK_INTERVAL_MINUTES = 120
PROACTIVE_MIN_COOLDOWN_MINUTES = 1
PROACTIVE_MAX_COOLDOWN_MINUTES = 120
PROACTIVE_MIN_SCREEN_CONTEXT_BATCH_LIMIT = 1
PROACTIVE_MAX_SCREEN_CONTEXT_BATCH_LIMIT = 20
PROACTIVE_TIMER_POLL_INTERVAL_MS = 60_000
PROACTIVE_SCREEN_CONTEXT_HISTORY_MARKER = "[已抓取屏幕上下文]"


@dataclass(frozen=True)
class ProactiveCareSettings:
    """主动关怀配置；由主动屏幕获取开关控制是否运行。"""

    enabled: bool = False
    screen_context_enabled: bool = False
    check_interval_minutes: int = PROACTIVE_DEFAULT_CHECK_INTERVAL_MINUTES
    cooldown_minutes: int = PROACTIVE_DEFAULT_COOLDOWN_MINUTES
    screen_context_batch_limit: int = PROACTIVE_DEFAULT_SCREEN_CONTEXT_BATCH_LIMIT

    @classmethod
    def load(cls, env_path: Path) -> "ProactiveCareSettings":
        try:
            values = load_env_file(env_path)
        except OSError:
            values = {}
        return cls(
            enabled=_parse_bool(values.get(PROACTIVE_CARE_ENABLED_KEY), default=False),
            screen_context_enabled=_parse_bool(
                values.get(PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY),
                default=False,
            ),
            check_interval_minutes=_parse_interval_minutes(
                values.get(PROACTIVE_CHECK_INTERVAL_MINUTES_KEY),
                default=PROACTIVE_DEFAULT_CHECK_INTERVAL_MINUTES,
                min_value=PROACTIVE_MIN_CHECK_INTERVAL_MINUTES,
                max_value=PROACTIVE_MAX_CHECK_INTERVAL_MINUTES,
            ),
            cooldown_minutes=_parse_interval_minutes(
                values.get(PROACTIVE_COOLDOWN_MINUTES_KEY),
                default=PROACTIVE_DEFAULT_COOLDOWN_MINUTES,
                min_value=PROACTIVE_MIN_COOLDOWN_MINUTES,
                max_value=PROACTIVE_MAX_COOLDOWN_MINUTES,
            ),
            screen_context_batch_limit=_parse_bounded_int(
                values.get(PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY),
                default=PROACTIVE_DEFAULT_SCREEN_CONTEXT_BATCH_LIMIT,
                min_value=PROACTIVE_MIN_SCREEN_CONTEXT_BATCH_LIMIT,
                max_value=PROACTIVE_MAX_SCREEN_CONTEXT_BATCH_LIMIT,
            ),
        )

    def normalized(self) -> "ProactiveCareSettings":
        screen_context_enabled = self.screen_context_enabled
        return ProactiveCareSettings(
            enabled=screen_context_enabled,
            screen_context_enabled=screen_context_enabled,
            check_interval_minutes=_clamp_interval_minutes(
                self.check_interval_minutes,
                min_value=PROACTIVE_MIN_CHECK_INTERVAL_MINUTES,
                max_value=PROACTIVE_MAX_CHECK_INTERVAL_MINUTES,
            ),
            cooldown_minutes=_clamp_interval_minutes(
                self.cooldown_minutes,
                min_value=PROACTIVE_MIN_COOLDOWN_MINUTES,
                max_value=PROACTIVE_MAX_COOLDOWN_MINUTES,
            ),
            screen_context_batch_limit=_clamp_bounded_int(
                self.screen_context_batch_limit,
                min_value=PROACTIVE_MIN_SCREEN_CONTEXT_BATCH_LIMIT,
                max_value=PROACTIVE_MAX_SCREEN_CONTEXT_BATCH_LIMIT,
            ),
        )

    def allows_screen_context(self) -> bool:
        """允许主动获取屏幕信息时，主动关怀才会运行。"""
        return self.screen_context_enabled

    def save(self, env_path: Path) -> None:
        settings = self.normalized()
        save_env_values(
            env_path,
            {
                PROACTIVE_CARE_ENABLED_KEY: _format_bool(settings.enabled),
                PROACTIVE_SCREEN_CONTEXT_ENABLED_KEY: _format_bool(settings.screen_context_enabled),
                PROACTIVE_CHECK_INTERVAL_MINUTES_KEY: str(settings.check_interval_minutes),
                PROACTIVE_COOLDOWN_MINUTES_KEY: str(settings.cooldown_minutes),
                PROACTIVE_SCREEN_CONTEXT_BATCH_LIMIT_KEY: str(
                    settings.screen_context_batch_limit
                ),
            },
        )


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _parse_interval_minutes(
    value: str | None,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    return _parse_bounded_int(
        value,
        default=default,
        min_value=min_value,
        max_value=max_value,
    )


def _parse_bounded_int(
    value: str | None,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return _clamp_bounded_int(parsed, min_value=min_value, max_value=max_value)


def _clamp_interval_minutes(value: int, *, min_value: int, max_value: int) -> int:
    return _clamp_bounded_int(value, min_value=min_value, max_value=max_value)


def _clamp_bounded_int(value: int, *, min_value: int, max_value: int) -> int:
    return max(
        min_value,
        min(max_value, value),
    )
