# Xa4lock

Фанатский армянский мод для Deadlock и его сайт (домен xa4lock.com).

## Сайт
Статика без сборки: открыть `index.html`. `CNAME` — домен для GitHub Pages.
Ростер (`heroes.js`) генерируется из `mod/heroes.json`, руками не править.

## Мод (v0.1 — имена героев)
- `mod/heroes.json` — ростер: ключ героя в игре → армянское имя (ru/en), роль, биография, ульта
- `mod/build.py` — сборка: `python mod/build.py` (или `--game "<папка Deadlock>"`, чтобы взять
  актуальные файлы локализации из своей установки, в т.ч. настоящий русский)
- `mod/vendor/` — английский `citadel_gc_hero_names_english.txt` из игры
  (снимок из SteamDatabase/GameTracking-Deadlock)

Результат: `mod/build/pak77_dir.vpk` (VPK v2) и `downloads/Xa4lock-0.1.zip` для сайта.
Инструкция по установке — `mod/INSTALL.txt`.
