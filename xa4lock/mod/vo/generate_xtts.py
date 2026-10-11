#!/usr/bin/env python3
"""Озвучка реплик Xa4lock клоном голоса (XTTS-v2, бесплатно, локально).

    pip install coqui-tts
    python generate_xtts.py bebop_pilot.json образец.wav

образец.wav — 20–60 с речи одного человека (его голос и говор переносятся
на все реплики). Модель XTTS-v2 скачивается при первом запуске; её лицензия
(Coqui Public Model License) — только некоммерческое использование.
Результат: out/sounds/vo/<герой>/...wav (24 кГц, моно, 16 бит).
"""
import json, os, re, sys, wave
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # согласие с некоммерческой лицензией CPML
from TTS.api import TTS  # noqa: E402

RATE = 24000


def clean(text):
    return re.sub(r"\[[^\]]*\]\s*", "", text).strip()  # теги ElevenLabs не нужны


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    ref = sys.argv[2]
    only = set(sys.argv[3:])  # можно передать ключи отдельных реплик
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    out = Path("out")
    items = [(e, t) for e, t in cfg["lines"].items() if not only or e in only]
    for i, (event, text) in enumerate(items, 1):
        wav = out / Path(cfg["sounds"][event]).with_suffix(".wav")
        wav.parent.mkdir(parents=True, exist_ok=True)
        audio = tts.tts(text=clean(text), speaker_wav=ref, language="ru")
        pcm = b"".join(int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
                       for s in audio)
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        print(f"[{i}/{len(items)}] {wav}", flush=True)


if __name__ == "__main__":
    main()
