"""Клиент генерации изображений через публичный API air.fail (модель gptimage).

Публичный API air.fail принимает запросы вида
    POST https://api.air.fail/public/image/<model>
    Authorization: <API-ключ из личного кабинета>
    {"content": "<промпт>", "info": {<настройки генерации>}}
(та же схема MessageInSchema, что и у /public/text/<model>/stream в
https://api.air.fail/public/openapi.json).

Точный формат ответа в открытой спецификации не описан, поэтому ответ
разбирается терпимо: ищем в JSON ссылку на картинку или base64.
Всё, что может отличаться, настраивается переменными окружения.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field

import requests
from PIL import Image

DEFAULT_ENDPOINT = "https://api.air.fail/public/image/gptimage"

# Размеры, которые поддерживает gpt-image.
SUPPORTED_SIZES = [(1024, 1024), (1024, 1536), (1536, 1024)]

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_B64_KEYS = ("b64_json", "base64", "image_base64", "b64")


class AirFailError(RuntimeError):
    pass


def closest_size(aspect: float) -> tuple[int, int]:
    """Подбирает поддерживаемый размер с ближайшим соотношением сторон (ширина/высота)."""
    import math

    return min(SUPPORTED_SIZES, key=lambda s: abs(math.log((s[0] / s[1]) / aspect)))


@dataclass
class AirFailClient:
    api_key: str
    endpoint: str = DEFAULT_ENDPOINT
    auth_scheme: str = ""  # "" -> "Authorization: <key>", "Bearer" -> "Authorization: Bearer <key>"
    extra_info: dict = field(default_factory=dict)
    timeout: int = 300
    retries: int = 2

    @classmethod
    def from_env(cls) -> "AirFailClient":
        key = os.environ.get("AIRFAIL_API_KEY")
        if not key:
            raise AirFailError("Не задан AIRFAIL_API_KEY (ключ из личного кабинета app.air.fail)")
        extra = os.environ.get("AIRFAIL_INFO_JSON")
        return cls(
            api_key=key,
            endpoint=os.environ.get("AIRFAIL_ENDPOINT", DEFAULT_ENDPOINT),
            auth_scheme=os.environ.get("AIRFAIL_AUTH_SCHEME", ""),
            extra_info=json.loads(extra) if extra else {},
        )

    def _headers(self) -> dict:
        auth = f"{self.auth_scheme} {self.api_key}".strip()
        return {"Authorization": auth, "Content-Type": "application/json", "Accept": "application/json"}

    def generate(self, prompt: str, size: tuple[int, int] = (1024, 1536)) -> Image.Image:
        info = {"size": f"{size[0]}x{size[1]}", **self.extra_info}
        payload = {"content": prompt, "info": info}

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(self.endpoint, headers=self._headers(), json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
            else:
                if resp.status_code in (401, 403, 404) and "key" in resp.text.lower():
                    raise AirFailError(f"air.fail отклонил ключ ({resp.status_code}): {resp.text[:300]}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = AirFailError(f"air.fail {resp.status_code}: {resp.text[:300]}")
                elif resp.status_code >= 400:
                    raise AirFailError(f"air.fail {resp.status_code}: {resp.text[:500]}")
                else:
                    return self._extract_image(resp)
            time.sleep(2 * (attempt + 1))
        raise AirFailError(f"Не удалось сгенерировать изображение: {last_err}")

    def _extract_image(self, resp: requests.Response) -> Image.Image:
        ctype = resp.headers.get("Content-Type", "")
        if ctype.startswith("image/"):
            return Image.open(io.BytesIO(resp.content)).convert("RGBA")

        text = resp.text
        try:
            data = json.loads(text)
        except ValueError:
            # Возможен стриминговый ответ (несколько JSON-строк / SSE) — берём ссылки из текста.
            data = text

        b64 = _find_b64(data)
        if b64:
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

        for url in _find_urls(data):
            r = requests.get(url, timeout=self.timeout)
            if r.ok and (r.headers.get("Content-Type", "").startswith("image/") or _looks_like_image(r.content)):
                return Image.open(io.BytesIO(r.content)).convert("RGBA")

        raise AirFailError(f"В ответе air.fail не найдено изображение: {text[:500]}")


def _find_b64(data) -> str | None:
    if isinstance(data, dict):
        for k, v in data.items():
            if k in _B64_KEYS and isinstance(v, str) and len(v) > 100:
                return v.split(",", 1)[-1]
            found = _find_b64(v)
            if found:
                return found
    elif isinstance(data, list):
        for v in data:
            found = _find_b64(v)
            if found:
                return found
    elif isinstance(data, str) and data.startswith("data:image/"):
        return data.split(",", 1)[-1]
    return None


def _find_urls(data) -> list[str]:
    """Все http(s)-ссылки из ответа; ссылки на картинки — первыми."""
    urls = _URL_RE.findall(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
    urls = [u.rstrip(").,\\") for u in dict.fromkeys(urls)]
    return sorted(urls, key=lambda u: not re.search(r"\.(png|jpe?g|webp)(\?|$)", u, re.I))


def _looks_like_image(content: bytes) -> bool:
    return content[:8].startswith(b"\x89PNG") or content[:3] == b"\xff\xd8\xff" or content[8:12] == b"WEBP"
