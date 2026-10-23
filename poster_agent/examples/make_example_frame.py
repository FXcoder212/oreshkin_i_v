"""Генерирует пример рамки: фирменная плашка с прозрачным окном под картинку."""
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


W, H = 1200, 1700
img = Image.new("RGBA", (W, H), (24, 32, 56, 255))
d = ImageDraw.Draw(img)
d.rectangle((40, 40, W - 41, H - 41), outline=(232, 180, 80, 255), width=6)
d.rectangle((90, 90, W - 91, 1390), fill=(0, 0, 0, 0))          # окно под картинку
d.rectangle((90, 1430, W - 91, 1630), fill=(232, 180, 80, 255))  # плашка под логотип/контакты
d.text((120, 1500), "КНИЖНЫЙ КВАРТАЛ  ·  kvartal.example", fill=(24, 32, 56, 255), font=font(40))
img.save(__file__.replace("make_example_frame.py", "frame_example.png"))
