from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.api_client import ApiSettings, OpenAICompatibleClient
from app.character_loader import CharacterConfigError, CharacterRegistry
from app.debug_log import debug_log
from app.pet_window import PetWindow
from app.tts import create_tts_provider


BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sakura Desktop Pet")
    app.setQuitOnLastWindowClosed(False)

    settings = ApiSettings.load(BASE_DIR / ".env")
    api_client = OpenAICompatibleClient(settings)
    debug_log(
        "Startup",
        "API 配置已加载",
        {
            "base_url": settings.base_url,
            "model": settings.model,
            "timeout_seconds": settings.timeout_seconds,
            "api_key": settings.api_key,
        },
    )
    try:
        character_registry = CharacterRegistry(BASE_DIR)
        character_profile = character_registry.current(BASE_DIR / ".env")
    except CharacterConfigError as exc:
        print(f"[Character] 配置无效：{exc}")
        return 1
    debug_log(
        "Startup",
        "角色配置已加载",
        {
            "character_id": character_profile.id,
            "display_name": character_profile.display_name,
            "reply_tones": character_profile.reply_tones,
        },
    )
    tts_provider = create_tts_provider(BASE_DIR, character_profile)
    debug_log(
        "Startup",
        "TTS Provider 已创建",
        {"provider": type(tts_provider).__name__},
    )

    pet_window = PetWindow(
        base_dir=BASE_DIR,
        character_registry=character_registry,
        character_profile=character_profile,
        api_client=api_client,
        tts_provider=tts_provider,
    )
    pet_window.show()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
