#!/usr/bin/env python3
"""Озвучка реплик Xa4lock через ElevenLabs. Запускается на вашем компьютере.

    set ELEVENLABS_API_KEY=ваш_ключ        (Windows, cmd)
    python generate.py bebop_pilot.json

Создаёт папку out/ с WAV-файлами по тем же путям, что у звуков игры
(out/sounds/vo/bebop/bebop_select_01.wav и т.д.). Уже готовые файлы
пропускаются, так что скрипт можно перезапускать. Нужен только Python 3.
"""
import json, os, sys, time, urllib.request, urllib.error, wave
from pathlib import Path

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_44100"
RATE = 44100


def tts(key, voice, text, model):
    body = {"text": text, "model_id": model, "language_code": "ru"}
    req = urllib.request.Request(API.format(voice=voice), data=json.dumps(body).encode(),
                                 headers={"xi-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key or len(sys.argv) < 2:
        sys.exit("Нужно: переменная ELEVENLABS_API_KEY и путь к json, например bebop_pilot.json")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    voice = cfg["voice_id"]
    model = cfg.get("model_id", "eleven_v3")
    prefix = cfg.get("text_prefix", "[Armenian accent] ")
    out = Path("out")
    for i, (event, text) in enumerate(cfg["lines"].items(), 1):
        wav = out / Path(cfg["sounds"][event]).with_suffix(".wav")
        if wav.exists():
            continue
        wav.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(cfg['lines'])}] {wav.name}: {text}")
        try:
            pcm = tts(key, voice, prefix + text, model)
        except urllib.error.HTTPError as e:
            sys.exit(f"Ошибка ElevenLabs {e.code}: {e.read().decode()[:300]}")
        with wave.open(str(wav), "wb") as w:   # 16 бит, моно, 44.1 кГц
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        time.sleep(0.3)
    print("Готово:", out.resolve())


if __name__ == "__main__":
    main()
