# Xa4lock

Фанатский армянский мод для Deadlock и его сайт (домен xa4lock.com).

## Сайт
Статика без сборки: открыть `index.html`. `CNAME` — домен для GitHub Pages.
Ростер (`heroes.js`) генерируется из `mod/heroes.json`, руками не править.

## Мод (v0.2 — имена героев и умений)
- `mod/heroes.json` — ростер: ключ героя в игре → армянское имя (ru/en), роль, биография,
  четыре умения (ключ в игре, оригинал, ru/en); `soon` — герой ещё не вышел
- `mod/build.py` — сборка: `python mod/build.py` (или `--game "<папка Deadlock>"`, чтобы взять
  актуальные файлы локализации из своей установки, в т.ч. настоящий русский)
- `mod/vendor/` — оригинальные файлы локализации из игры: английские из
  SteamDatabase/GameTracking-Deadlock; русские (`*_russian.txt`) нужно положить сюда из игры —
  без `citadel_heroes_russian.txt` русские названия умений не собираются

Результат: `mod/build/pak77_dir.vpk` (VPK v2) и `downloads/Xa4lock-0.2.zip` для сайта.
Инструкция по установке — `mod/INSTALL.txt`.
