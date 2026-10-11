# Озвучка Xa4lock (пилот: Бебопян)

- `bebop_pilot.json` — 44 реплики Бебопа на русском с армянским говором,
  путь к звуку игры для каждой реплики и голос ElevenLabs.
- `generate.py` — создаёт WAV-файлы через ElevenLabs у вас на компьютере
  (бесплатный тариф ElevenLabs не работает из облака/через VPN).

Запуск (Windows, cmd):

    set ELEVENLABS_API_KEY=ваш_ключ
    python generate.py bebop_pilot.json

Результат — папка `out/sounds/vo/bebop/...wav`. Дальше файлы нужно
скомпилировать в формат игры (.vsnd_c) и положить в аддон.
