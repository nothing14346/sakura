from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.api_client import ApiSettings, OpenAICompatibleClient
from app.character_loader import load_system_prompt
from app.pet_window import PetWindow
from app.tts import create_tts_provider


BASE_DIR = Path(__file__).resolve().parent
PORTRAIT_PATH = BASE_DIR / "st" / "ST31A_A020_结果.png"
PERSONA_PATH = BASE_DIR / "夜乃桜_桌宠人格说明.md"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sakura Desktop Pet")
    app.setQuitOnLastWindowClosed(False)

    settings = ApiSettings.load(BASE_DIR / ".env")
    api_client = OpenAICompatibleClient(settings)
    system_prompt = load_system_prompt(PERSONA_PATH)
    tts_provider = create_tts_provider(BASE_DIR)

    pet_window = PetWindow(
        base_dir=BASE_DIR,
        portrait_path=PORTRAIT_PATH,
        api_client=api_client,
        system_prompt=system_prompt,
        tts_provider=tts_provider,
    )
    pet_window.show()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
