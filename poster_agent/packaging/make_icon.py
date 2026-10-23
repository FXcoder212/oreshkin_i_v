"""Рисует иконку приложения: assets/icon.png и assets/icon.ico."""
from pathlib import Path

from PIL import Image, ImageDraw

S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((16, 16, S - 16, S - 16), 110, fill=(61, 109, 242, 255))
d.rounded_rectangle((130, 80, 382, 432), 18, fill=(255, 255, 255, 255))   # лист афиши
d.rectangle((155, 105, 357, 300), fill=(255, 196, 61, 255))                # окно с картинкой
d.polygon([(155, 300), (220, 200), (270, 260), (310, 215), (357, 300)], fill=(46, 64, 120, 255))
d.ellipse((300, 125, 340, 165), fill=(255, 255, 255, 255))
for y in (330, 365, 400):
    d.rounded_rectangle((165, y, 347 if y == 330 else 300, y + 16), 8, fill=(190, 200, 220, 255))

out = Path(__file__).resolve().parent.parent / "assets"
out.mkdir(exist_ok=True)
img.save(out / "icon.png")
img.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
