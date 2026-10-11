#!/usr/bin/env python3
"""Сборка мода Xa4lock.

Берёт ростер из heroes.json, подставляет армянские имена в файлы локализации
Deadlock (citadel_gc_hero_names_<язык>.txt) и пакует их в VPK для папки
game/citadel/addons. Заодно обновляет ростер на сайте (../heroes.js).

    python build.py                      # база: vendor/ (английский файл из игры)
    python build.py --game "C:/.../Deadlock"   # база: файлы из вашей установки игры

Со своей установкой сборка подхватывает актуальные файлы после патча и
настоящий русский файл, меняя в них только имена героев.
"""
import argparse
import hashlib
import json
import re
import struct
import zipfile
import zlib
from pathlib import Path

VERSION = "0.1"
ROOT = Path(__file__).resolve().parent
LOC_DIR = "resource/localization/citadel_gc_hero_names"
LANGS = {"english": "en", "russian": "ru"}  # язык игры -> поле в heroes.json
PAK_NAME = "pak77_dir.vpk"


def base_file(lang, game_dir):
    """Исходный файл локализации: из установки игры, если есть, иначе vendor/."""
    name = f"citadel_gc_hero_names_{lang}.txt"
    if game_dir:
        p = Path(game_dir) / "game/citadel" / LOC_DIR / name
        if p.exists():
            return p.read_text(encoding="utf-8-sig"), str(p)
    p = ROOT / "vendor/citadel_gc_hero_names_english.txt"
    text = p.read_text(encoding="utf-8-sig")
    text = re.sub(r'("Language"\s+)"english"', rf'\1"{lang}"', text, count=1)
    return text, str(p)


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def patch(text, tokens):
    """Заменяет значения токенов; отсутствующие дописывает в конец блока Tokens."""
    missing = dict(tokens)
    for tok, val in tokens.items():
        pat = re.compile(rf'^(\s*"{re.escape(tok)}"\s+)"(?:[^"\\]|\\.)*"', re.M)
        text, n = pat.subn(lambda m: f'{m.group(1)}"{esc(val)}"', text)
        if n:
            missing.pop(tok)
    if missing:
        extra = "".join(f'\t\t"{t}" "{esc(v)}"\n' for t, v in missing.items())
        # последняя "}" закрывает "lang", предпоследняя — "Tokens"
        i = text.rstrip().rfind("}")
        j = text.rfind("\n", 0, text.rfind("}", 0, i)) + 1
        text = text[:j] + "\t\t// Xa4lock\n" + extra + text[j:]
    return text


def tokens_for(heroes, field, lang_base):
    out = {}
    for h in heroes:
        k, name = h["key"], h[field]
        out[f"{k}:n"] = name
        out[f"Steam_RP_{k}"] = name
        out[f"{k}_sort:n"] = name
        # поиск находит героя и по новому, и по оригинальному имени
        out[f"{k}_search:n"] = f"{name} {h['orig']}"
    # _sort есть не у всех героев — дописываем только существующие
    return {t: v for t, v in out.items()
            if not t.endswith("_sort:n") or f'"{t}"' in lang_base}


def write_vpk(path, files):
    """VPK v2, все данные внутри _dir.vpk (archive index 0x7FFF)."""
    by_ext = {}
    for rel, data in files.items():
        d, _, fname = rel.rpartition("/")
        stem, _, ext = fname.rpartition(".")
        by_ext.setdefault(ext, {}).setdefault(d or " ", []).append((stem, data))

    tree, blob = bytearray(), bytearray()
    for ext, dirs in sorted(by_ext.items()):
        tree += ext.encode() + b"\0"
        for d, entries in sorted(dirs.items()):
            tree += d.encode() + b"\0"
            for stem, data in sorted(entries):
                tree += stem.encode() + b"\0"
                tree += struct.pack("<IHHIIH", zlib.crc32(data) & 0xFFFFFFFF, 0,
                                    0x7FFF, len(blob), len(data), 0xFFFF)
                blob += data
            tree += b"\0"
        tree += b"\0"
    tree += b"\0"

    archive_md5 = b""
    header = struct.pack("<7I", 0x55AA1234, 2, len(tree), len(blob),
                         len(archive_md5), 48, 0)
    body = header + tree + blob + archive_md5
    other = hashlib.md5(tree).digest() + hashlib.md5(archive_md5).digest()
    other += hashlib.md5(body + other).digest()
    path.write_bytes(body + other)


def write_site_roster(heroes):
    keep = ("orig", "role", "glyph", "bio", "ult")
    rows = [{"name": h["ru"], **{k: h[k] for k in keep}} for h in heroes]
    js = ("// Сгенерировано mod/build.py из mod/heroes.json — правьте JSON.\n"
          "const HEROES = " + json.dumps(rows, ensure_ascii=False, indent=1) + ";\n")
    (ROOT.parent / "heroes.js").write_text(js, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", help="папка Deadlock (где лежит game/citadel)")
    args = ap.parse_args()

    heroes = json.loads((ROOT / "heroes.json").read_text(encoding="utf-8"))
    out = ROOT / "build"
    files = {}
    for lang, field in LANGS.items():
        text, src = base_file(lang, args.game)
        res = patch(text, tokens_for(heroes, field, text))
        rel = f"{LOC_DIR}/citadel_gc_hero_names_{lang}.txt"
        data = ("\ufeff" + res).encode("utf-8")
        files[rel] = data
        loose = out / "addons" / rel
        loose.parent.mkdir(parents=True, exist_ok=True)
        loose.write_bytes(data)
        print(f"{lang}: {src}")

    pak = out / PAK_NAME
    write_vpk(pak, files)
    zpath = out / f"Xa4lock-{VERSION}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pak, PAK_NAME)
        z.write(ROOT / "INSTALL.txt", "INSTALL.txt")
    dl = ROOT.parent / "downloads"
    dl.mkdir(exist_ok=True)
    (dl / zpath.name).write_bytes(zpath.read_bytes())
    write_site_roster(heroes)
    print(f"{len(heroes)} героев -> {pak.relative_to(ROOT)}, {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
