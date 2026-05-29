from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.api_client import ApiSettings
from app.tts import GPTSoVITSTTSSettings, TTSConfigError


class SettingsDialog(QDialog):
    def __init__(
        self,
        api_settings: ApiSettings,
        tts_settings: GPTSoVITSTTSSettings,
        base_dir: Path,
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self.tts_settings = tts_settings
        self.result_api_settings: ApiSettings | None = None
        self.result_tts_settings: GPTSoVITSTTSSettings | None = None

        self.setWindowTitle("设置")
        self.resize(560, 360)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_api_tab(api_settings), "API")
        tabs.addTab(self._build_tts_tab(tts_settings), "TTS")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.setStyleSheet(
            """
            QDialog {
                background: #f4fbfd;
                color: #24343a;
                font-family: "Microsoft YaHei", "Yu Gothic UI", sans-serif;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid rgba(120, 176, 188, 0.48);
                border-radius: 8px;
                background: rgba(226, 246, 250, 0.70);
            }
            QTabBar::tab {
                background: rgba(226, 246, 250, 0.75);
                border: 1px solid rgba(120, 176, 188, 0.42);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 7px 18px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #27616d;
                font-weight: 700;
            }
            QLineEdit, QSpinBox {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(120, 176, 188, 0.50);
                border-radius: 7px;
                padding: 6px 8px;
            }
            QPushButton {
                background: #72c7d6;
                border: none;
                border-radius: 8px;
                color: white;
                min-width: 72px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #5eb7c8;
            }
            """
        )

    def _build_api_tab(self, settings: ApiSettings) -> QWidget:
        tab = QWidget(self)
        self.base_url_edit = QLineEdit(settings.base_url, tab)
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")

        self.api_key_edit = QLineEdit(settings.api_key, tab)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("请输入 API Key")

        self.model_edit = QLineEdit(settings.model, tab)
        self.model_edit.setPlaceholderText("gpt-4.1-mini")

        self.api_timeout_spin = QSpinBox(tab)
        self.api_timeout_spin.setRange(1, 600)
        self.api_timeout_spin.setSuffix(" 秒")
        self.api_timeout_spin.setValue(settings.timeout_seconds)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(16, 18, 16, 16)
        form_layout.setSpacing(12)
        form_layout.addRow("Base URL", self.base_url_edit)
        form_layout.addRow("API Key", self.api_key_edit)
        form_layout.addRow("模型", self.model_edit)
        form_layout.addRow("超时", self.api_timeout_spin)
        tab.setLayout(form_layout)
        return tab

    def _build_tts_tab(self, settings: GPTSoVITSTTSSettings) -> QWidget:
        tab = QWidget(self)
        self.tts_enabled_check = QCheckBox("启用 GPT-SoVITS 语音", tab)
        self.tts_enabled_check.setChecked(settings.enabled)

        self.tts_api_url_edit = QLineEdit(settings.api_url, tab)
        self.tts_api_url_edit.setPlaceholderText("http://127.0.0.1:9880/tts")

        self.ref_lang_edit = QLineEdit(settings.ref_lang, tab)
        self.text_lang_edit = QLineEdit(settings.text_lang, tab)

        self.tts_timeout_spin = QSpinBox(tab)
        self.tts_timeout_spin.setRange(1, 600)
        self.tts_timeout_spin.setSuffix(" 秒")
        self.tts_timeout_spin.setValue(settings.timeout_seconds)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(16, 18, 16, 16)
        form_layout.setSpacing(12)
        form_layout.addRow("", self.tts_enabled_check)
        form_layout.addRow("API URL", self.tts_api_url_edit)
        form_layout.addRow("参考语言", self.ref_lang_edit)
        form_layout.addRow("文本语言", self.text_lang_edit)
        form_layout.addRow("超时", self.tts_timeout_spin)
        tab.setLayout(form_layout)
        return tab

    def accept(self) -> None:
        api_settings = self._validated_api_settings()
        if api_settings is None:
            return
        tts_settings = self._validated_tts_settings()
        if tts_settings is None:
            return

        self.result_api_settings = api_settings
        self.result_tts_settings = tts_settings
        super().accept()

    def _validated_api_settings(self) -> ApiSettings | None:
        base_url = self.base_url_edit.text().strip().rstrip("/")
        api_key = self.api_key_edit.text().strip()
        model = self.model_edit.text().strip()

        if not _is_http_url(base_url):
            QMessageBox.warning(self, "配置无效", "Base URL 必须是有效的 http 或 https 地址。")
            return None
        if not api_key:
            QMessageBox.warning(self, "配置无效", "API Key 不能为空。")
            return None
        if not model:
            QMessageBox.warning(self, "配置无效", "模型不能为空。")
            return None

        return ApiSettings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=self.api_timeout_spin.value(),
        )

    def _validated_tts_settings(self) -> GPTSoVITSTTSSettings | None:
        enabled = self.tts_enabled_check.isChecked()
        api_url = self.tts_api_url_edit.text().strip()
        ref_audio_path = self.tts_settings.ref_audio_path
        ref_text_path = self.tts_settings.ref_text_path
        ref_lang = self.ref_lang_edit.text().strip()
        text_lang = self.text_lang_edit.text().strip()

        if enabled and not _is_http_url(api_url):
            QMessageBox.warning(self, "配置无效", "TTS API URL 必须是有效的 http 或 https 地址。")
            return None

        settings = GPTSoVITSTTSSettings(
            enabled=enabled,
            api_url=api_url,
            ref_audio_path=ref_audio_path,
            ref_text_path=ref_text_path,
            ref_text=self.tts_settings.ref_text,
            ref_lang=ref_lang,
            text_lang=text_lang,
            timeout_seconds=self.tts_timeout_spin.value(),
            tone_references=self.tts_settings.tone_references,
        )
        if enabled:
            try:
                settings.validate()
            except TTSConfigError as exc:
                QMessageBox.warning(self, "配置无效", str(exc))
                return None
        return settings


def _is_http_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)
