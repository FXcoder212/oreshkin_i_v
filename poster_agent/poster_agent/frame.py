"""Работа с рамкой: поиск «окна» под изображение и вставка картинки в него.

Окно ищется так (в порядке приоритета):
1. явно заданный прямоугольник box=(x, y, w, h);
2. прозрачная область PNG — крупнейшая связная область с alpha < 128;
3. область хромакея (по умолчанию чистый зелёный #00FF00), она же делается прозрачной.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from PIL import Image

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Frame:
    image: Image.Image  # RGBA, окно прозрачное
    box: Box

    @property
    def aspect(self) -> float:
        return self.box[2] / self.box[3]


def load_frame(path: str, box: Box | None = None, chroma: tuple[int, int, int] | None = (0, 255, 0),
               chroma_tol: int = 60) -> Frame:
    img = Image.open(path).convert("RGBA")

    if box is None:
        mask = img.getchannel("A").point(lambda a: 255 if a < 128 else 0)
        box = _largest_region(mask)

    if box is None and chroma is not None:
        r, g, b = chroma
        rgb = img.convert("RGB")
        mask = Image.new("L", img.size)
        mask.putdata([255 if abs(p[0] - r) + abs(p[1] - g) + abs(p[2] - b) <= chroma_tol else 0
                      for p in rgb.getdata()])
        box = _largest_region(mask)
        if box is not None:
            alpha = img.getchannel("A")
            alpha.paste(0, mask=mask)
            img.putalpha(alpha)

    if box is None:
        raise ValueError(
            "Не удалось найти окно в рамке: сделайте область под картинку прозрачной (PNG), "
            "залейте её чистым зелёным #00FF00 или укажите координаты окна вручную."
        )

    if not _has_transparency(img, box):
        # Окно задано вручную на непрозрачной рамке — вырезаем его.
        x, y, w, h = box
        alpha = img.getchannel("A")
        alpha.paste(0, (x, y, x + w, y + h))
        img.putalpha(alpha)

    return Frame(img, box)


def compose(frame: Frame, art: Image.Image) -> Image.Image:
    """Масштабирует картинку «с обрезкой» под окно и кладёт рамку поверх."""
    x, y, w, h = frame.box
    art = art.convert("RGBA")
    scale = max(w / art.width, h / art.height)
    nw, nh = round(art.width * scale), round(art.height * scale)
    art = art.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    art = art.crop((left, top, left + w, top + h))

    canvas = Image.new("RGBA", frame.image.size, (0, 0, 0, 0))
    canvas.paste(art, (x, y))
    return Image.alpha_composite(canvas, frame.image)


def _has_transparency(img: Image.Image, box: Box) -> bool:
    x, y, w, h = box
    lo, _ = img.getchannel("A").crop((x, y, x + w, y + h)).getextrema()
    return lo < 128


def _largest_region(mask: Image.Image, work: int = 400, min_share: float = 0.02) -> Box | None:
    """Ограничивающий прямоугольник крупнейшей связной белой области маски.

    Считаем на уменьшенной копии, чтобы не обходить миллионы пикселей в Python.
    Области, касающиеся края картинки (скруглённые углы и т.п.), отбрасываем,
    если есть хоть одна внутренняя.
    """
    W, H = mask.size
    k = max(W, H) / work if max(W, H) > work else 1.0
    small = mask.resize((max(1, round(W / k)), max(1, round(H / k))), Image.NEAREST)
    sw, sh = small.size
    px = small.load()
    seen = bytearray(sw * sh)

    regions = []  # (size, touches_edge, x0, y0, x1, y1)
    for sy in range(sh):
        for sx in range(sw):
            if px[sx, sy] < 128 or seen[sy * sw + sx]:
                continue
            q = deque([(sx, sy)])
            seen[sy * sw + sx] = 1
            n, x0, y0, x1, y1 = 0, sx, sy, sx, sy
            while q:
                cx, cy = q.popleft()
                n += 1
                x0, y0, x1, y1 = min(x0, cx), min(y0, cy), max(x1, cx), max(y1, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < sw and 0 <= ny < sh and not seen[ny * sw + nx] and px[nx, ny] >= 128:
                        seen[ny * sw + nx] = 1
                        q.append((nx, ny))
            edge = x0 == 0 or y0 == 0 or x1 == sw - 1 or y1 == sh - 1
            regions.append((n, edge, x0, y0, x1, y1))

    regions = [r for r in regions if r[0] >= min_share * sw * sh]
    if not regions:
        return None
    inner = [r for r in regions if not r[1]]
    _, _, x0, y0, x1, y1 = max(inner or regions)
    bx, by = int(x0 * k), int(y0 * k)
    return bx, by, min(W, int((x1 + 1) * k)) - bx, min(H, int((y1 + 1) * k)) - by
