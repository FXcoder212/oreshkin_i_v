#!/usr/bin/env python3
"""Сборка мода Xa4lock.

Берёт ростер из heroes.json, подставляет армянские имена героев и умений
в файлы локализации Deadlock и пакует их в VPK для папки game/citadel/addons.
Заодно обновляет ростер на сайте (../heroes.js).

    python build.py                            # база: файлы в vendor/
    python build.py --game "C:/.../Deadlock"   # база: файлы из вашей установки игры

Игра читает файл локализации целиком, поэтому в мод кладётся полная копия
оригинала, в которой заменены только наши строки. Базовый файл ищется так:
установка игры (--game) -> vendor/<файл>_<язык>.txt -> английский vendor-файл
(последнее только для имён героев: там всё равно почти одни имена).
Файл умений без своей русской базы не собирается, чтобы не потерять
русские описания.
"""
import argparse
import hashlib
import json
import re
import struct
import zipfile
import zlib
from pathlib import Path

VERSION = "0.3"
ROOT = Path(__file__).resolve().parent
LOC_ROOT = "resource/localization"
LANGS = {"english": "en", "russian": "ru"}  # язык игры -> поле в heroes.json
PAK_NAME = "pak77_dir.vpk"


def base_file(stem, lang, game_dir, english_fallback):
    """Исходный файл локализации (текст, откуда взят) или (None, None)."""
    name = f"{stem}_{lang}.txt"
    candidates = [ROOT / "vendor" / name]
    if game_dir:
        candidates.insert(0, Path(game_dir) / "game/citadel" / LOC_ROOT / stem / name)
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8-sig"), str(p)
    if not english_fallback:
        return None, None
    p = ROOT / f"vendor/{stem}_english.txt"
    text = p.read_text(encoding="utf-8-sig")
    text = re.sub(r'("Language"\s+)"english"', rf'\1"{lang}"', text, count=1)
    return text, str(p)


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def patch(text, tokens):
    """Меняет значения токенов. Значение — строка (новый текст) или функция
    от старого значения (в экранированном виде). Отсутствующие строковые
    токены дописываются в конец блока Tokens."""
    missing = {t: v for t, v in tokens.items() if isinstance(v, str)}
    for tok, val in tokens.items():
        pat = re.compile(rf'^(\s*"{re.escape(tok)}"\s+)"((?:[^"\\]|\\.)*)"', re.M)
        new = (lambda m: f'{m.group(1)}"{esc(val)}"') if isinstance(val, str) else \
              (lambda m, f=val: f'{m.group(1)}"{f(m.group(2))}"')
        text, n = pat.subn(new, text)
        if n:
            missing.pop(tok, None)
    if missing:
        extra = "".join(f'\t\t"{t}" "{esc(v)}"\n' for t, v in missing.items())
        # последняя "}" закрывает "lang", предпоследняя — "Tokens"
        i = text.rstrip().rfind("}")
        j = text.rfind("\n", 0, text.rfind("}", 0, i)) + 1
        text = text[:j] + "\t\t// Xa4lock\n" + extra + text[j:]
    return text


def name_tokens(heroes, field, lang_base):
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


# Части оригинальных имён, которые встречаются в текстах сами по себе
EXTRA_EN_FORMS = {"hero_orion": {"Talon": "Talonyan"}, "hero_ghost": {"Geist": "Gayane"},
                  "hero_krill": {"Krill": "Krikor"}}
FLAVOR = '<span class="diminish">{}</span><br>'


def renamer(forms):
    """Функция замены оригинальных имён целыми словами (длинные формы первыми)."""
    if not forms:
        return None
    rx = re.compile(r"(?<!\w)(" + "|".join(
        re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")(?!\w)")
    return lambda raw: rx.sub(lambda m: esc(forms[m.group(1)]), raw)


def hero_tokens(heroes, field, lang_base):
    """Биографии, названия и описания умений, имена в текстах о герое."""
    known = (ROOT / "vendor/citadel_heroes_english.txt").read_text(encoding="utf-8-sig")
    keys = re.findall(r'^\s*"([^"]+)"\s+"', lang_base, re.M)
    out = {}

    def add(tok, val, insert=False):
        if f'"{tok}"' in lang_base or (insert and f'"{tok}"' in known):
            out[tok] = val

    for h in heroes:
        k = h["key"]
        bio = h["bio"] if field == "ru" else h["bio_en"]
        add(f"{k}_lore", bio, insert=True)
        forms = h.get("ru_forms") if field == "ru" else \
            {h["orig"]: h["en"], **EXTRA_EN_FORMS.get(k, {})}
        rename = renamer(forms)
        abil = [a["key"] + "_" for a in h["abilities"]]
        if rename:
            for tok in keys:
                if tok in (f"{k}_role", f"{k}_playstyle") or \
                        (tok.startswith(tuple(abil)) and not tok.endswith("_quip")):
                    out[tok] = rename
        for a in h["abilities"]:
            add(a["key"], a[field], insert=True)
            flavor = a[f"flavor_{field}"]
            if f'"{a["key"]}_quip"' in lang_base:
                out[f'{a["key"]}_quip'] = flavor
            elif f'"{a["key"]}_desc"' in lang_base:
                prev = out.get(f'{a["key"]}_desc') or (lambda raw: raw)
                out[f'{a["key"]}_desc'] = \
                    lambda raw, p=prev, f=flavor: esc(FLAVOR.format(f)) + p(raw)
    return out


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
    rows = [{"name": h["ru"], "orig": h["orig"], "role": h["role"], "glyph": h["glyph"],
             "bio": h["bio"], "soon": h.get("soon", False),
             "abilities": [[a["ru"], a["orig"]] for a in h["abilities"]]}
            for h in heroes]
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
        for stem, tokens_fn, fallback in (
            ("citadel_gc_hero_names", name_tokens, True),
            ("citadel_heroes", hero_tokens, False),
        ):
            text, src = base_file(stem, lang, args.game, fallback)
            if text is None:
                print(f"{lang}/{stem}: нет базового файла, пропущено")
                continue
            res = patch(text, tokens_fn(heroes, field, text))
            rel = f"{LOC_ROOT}/{stem}/{stem}_{lang}.txt"
            data = ("\ufeff" + res).encode("utf-8")
            files[rel] = data
            loose = out / "addons" / rel
            loose.parent.mkdir(parents=True, exist_ok=True)
            loose.write_bytes(data)
            print(f"{lang}/{stem}: {src}")

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
    n_ab = sum(len(h["abilities"]) for h in heroes)
    print(f"{len(heroes)} героев, {n_ab} умений -> {pak.relative_to(ROOT)}, {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
