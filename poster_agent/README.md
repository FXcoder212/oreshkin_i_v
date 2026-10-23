# Агент для афиш по плану мероприятий

ИИ-агент делает готовые афиши по плану мероприятий:

1. **Claude** читает план (текст, таблицу, PDF, DOCX, XLSX) и рамку пользователя. Для каждого
   мероприятия он пишет подробный промпт с учётом стиля и палитры рамки и формы окна.
2. Через инструмент `create_poster` промпт уходит в **air.fail → gptimage**
   (`https://api.air.fail/public/image/gptimage`). Это тот же GPT Image, что на
   https://app.air.fail/images/gptimage, только через API-ключ.
3. Картинка масштабируется с обрезкой под окно рамки, рамка кладётся поверх.
4. Готовая афиша возвращается Claude на проверку. Если текст искажён, главный объект
   обрезан или стиль не подходит к рамке, агент исправляет промпт и генерирует заново
   (не больше 3 попыток на мероприятие).

## Рамка

Подойдёт PNG, где область под картинку:
- **прозрачная** (лучший вариант, окно находится автоматически), или
- залита **чистым зелёным `#00FF00`** (хромакей), или
- задана вручную: `--box x,y,w,h`.

Под пропорции окна выбирается ближайший размер генерации: 1024×1024, 1024×1536 или 1536×1024.
Пример рамки лежит в `examples/frame_example.png`, а создаёт её `examples/make_example_frame.py`.

## Установка

```bash
cd poster_agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # ключ Claude API
export AIRFAIL_API_KEY=...     # API-ключ из личного кабинета air.fail
```

## Запуск

Командная строка:

```bash
python -m poster_agent --plan examples/plan_example.md --frame examples/frame_example.png --out posters
```

Флаги:
- `--no-text` — картинки без надписей (если текст накладываете сами);
- `--dry-run` — только промпты, без генерации и без ключа air.fail;
- `--effort low|medium|high|xhigh|max` — насколько тщательно думает агент;
- `--box x,y,w,h` — задать окно вручную.

Веб-интерфейс (загрузка рамки и плана, галерея, скачивание zip):

```bash
streamlit run app.py
```

## Что получается

В папке `--out` для каждой попытки лежат три файла и общий манифест:
- `<id>_vN_prompt.txt` — промпт;
- `<id>_vN_art.png` — исходная картинка;
- `<id>_vN_poster.png` — афиша в рамке;
- `manifest.json` — все мероприятия, промпты, попытки и ошибки.

## Настройка air.fail

Открытой документации по image-эндпоинту у air.fail нет. Клиент повторяет схему их
публичного API (`api.air.fail/public/openapi.json`): тело `{"content": <промпт>, "info": {...}}`,
ключ передаётся в заголовке `Authorization`. В ответе клиент сам ищет ссылку на картинку
или base64. Если у вашего аккаунта формат другой, его можно поменять переменными окружения:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `AIRFAIL_ENDPOINT` | `https://api.air.fail/public/image/gptimage` | адрес генерации |
| `AIRFAIL_AUTH_SCHEME` | пусто (`Authorization: <ключ>`) | `Bearer`, если нужен `Authorization: Bearer <ключ>` |
| `AIRFAIL_INFO_JSON` | — | доп. параметры в `info`, например `{"quality": "high"}` |

## Устройство

- `poster_agent/agent.py` — агент: системный промпт, инструмент `create_poster`, цикл
  `client.beta.messages.tool_runner` (Claude Opus 5.5, адаптивное мышление, серверный fallback).
- `poster_agent/airfail.py` — клиент air.fail.
- `poster_agent/frame.py` — поиск окна в рамке и композиция.
- `app.py` — веб-интерфейс на Streamlit.
