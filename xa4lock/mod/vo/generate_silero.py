#!/usr/bin/env python3
"""Бесплатная локальная озвучка реплик Xa4lock моделью Silero (без ключей и сети
после первой загрузки модели). Армянского говора у модели нет — это обычный
русский диктор; для говора см. README.

    pip install torch numpy
    python generate_silero.py bebop_pilot.json [speaker]

Модель v4_ru.pt скачивается с models.silero.ai при первом запуске.
Голоса: aidar, eugene (мужские), baya, kseniya, xenia (женские).
Результат: out/sounds/vo/<герой>/...wav (48 кГц, моно, 16 бит).
"""
import json, re, sys, urllib.request, wave
from pathlib import Path

import torch

MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
RATE = 48000


def clean(text):
    # теги ElevenLabs вроде [laughs] Silero не понимает
    return re.sub(r"\[[^\]]*\]\s*", "", text).strip()


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    speaker = sys.argv[2] if len(sys.argv) > 2 else cfg.get("silero_speaker", "aidar")
    model_path = Path("v4_ru.pt")
    if not model_path.exists():
        urllib.request.urlretrieve(MODEL_URL, model_path)
    model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
    model.to("cpu")
    out = Path("out")
    for i, (event, text) in enumerate(cfg["lines"].items(), 1):
        wav = out / Path(cfg["sounds"][event]).with_suffix(".wav")
        wav.parent.mkdir(parents=True, exist_ok=True)
        audio = model.apply_tts(text=clean(text), speaker=speaker, sample_rate=RATE)
        pcm = (audio.clamp(-1, 1) * 32767).short().numpy().tobytes()
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        print(f"[{i}/{len(cfg['lines'])}] {wav}")


if __name__ == "__main__":
    main()
