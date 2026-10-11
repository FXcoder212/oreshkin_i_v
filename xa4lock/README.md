# Xa4lock

Фанатский армянский мод для Deadlock и его сайт (домен xa4lock.com).

## Сайт
Статика без сборки: открыть `index.html`. `CNAME` — домен для GitHub Pages.
Ростер (`heroes.js`) генерируется из `mod/heroes.json`, руками не править.

## Мод (v0.3 — имена, биографии, умения)
- `mod/heroes.json` — ростер: ключ героя в игре → армянское имя (ru/en), роль, биография,
  четыре умения (ключ в игре, оригинал, ru/en); `soon` — герой ещё не вышел
- `mod/build.py` — сборка: `python mod/build.py` (или `--game "<папка Deadlock>"`, чтобы взять
  актуальные файлы локализации из своей установки, в т.ч. настоящий русский)
- `mod/vendor/` — оригинальные файлы локализации из игры: английские из
  SteamDatabase/GameTracking-Deadlock, `citadel_heroes_russian.txt` — из игры пользователя

Результат: `mod/build/pak77_dir.vpk` (VPK v2) и `downloads/Xa4lock-0.3.zip` для сайта.
Инструкция по установке — `mod/INSTALL.txt`.
