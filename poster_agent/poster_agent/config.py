"""Настройки приложения (ключи, последние папки) в профиле пользователя."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .airfail import DEFAULT_ENDPOINT


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "PosterAgent"


@dataclass
class Settings:
    anthropic_api_key: str = ""
    airfail_api_key: str = ""
    airfail_endpoint: str = DEFAULT_ENDPOINT
    airfail_auth_scheme: str = ""
    airfail_info_json: str = ""
    out_dir: str = str(Path.home() / "Афиши")
    with_text: bool = True
    effort: str = "medium"

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        path = config_dir() / "settings.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for f in fields(cls):
                    if f.name in data:
                        setattr(s, f.name, data[f.name])
            except (ValueError, OSError):
                pass
        # Переменные окружения — запасной источник ключей.
        s.anthropic_api_key = s.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        s.airfail_api_key = s.airfail_api_key or os.environ.get("AIRFAIL_API_KEY", "")
        return s

    def save(self) -> None:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "settings.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
