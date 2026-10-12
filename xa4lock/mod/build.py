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

VERSION = "0.8"
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
        pat = re.compile(rf'^(\s*"{re.escape(tok)}"\s*)"((?:[^"\\]|\\.)*)"', re.M)
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
DESCRIPTIONS = json.loads((ROOT / "descriptions.json").read_text(encoding="utf-8"))
ITEMS = json.loads((ROOT / "items.json").read_text(encoding="utf-8"))
ITEM_DESCRIPTIONS = json.loads((ROOT / "item_descriptions.json").read_text(encoding="utf-8"))


def stem_pattern(name):
    r"""«Ледяная дорожка» -> Ледян\w{0,3} дорожк\w{0,3}: ловит падежные формы."""
    parts = re.split(r"(\W+)", name)
    return "".join(re.escape(p[:-2] if len(p) > 4 else p[:-1] if len(p) > 3 else p)
                   + r"\w{0,3}" if re.match(r"\w", p) else re.escape(p) for p in parts)


def renamer(forms, quoted_stems=False):
    """Функция замены оригинальных имён целыми словами (длинные формы первыми).
    quoted_stems: ещё и любые падежные формы внутри «кавычек» (для русского)."""
    if not forms:
        return None
    rx = re.compile(r"(?<!\w)(" + "|".join(
        re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")(?!\w)")
    quoted = [(re.compile("«" + stem_pattern(f) + "»"), f"«{n}»") for f, n in forms.items()] \
        if quoted_stems else []

    def run(raw):
        for q, rep in quoted:
            raw = q.sub(esc(rep), raw)
        return rx.sub(lambda m: esc(forms[m.group(1)]), raw)
    return run


def token_owners(heroes, keys):
    """Герой для каждой строки: чьё умение последним встречалось выше по файлу."""
    prefixes = {}
    for h in heroes:
        prefixes[h["key"]] = h["key"]
        for a in h["abilities"]:
            prefixes[a["key"]] = h["key"]
    owner, cur = {}, None
    for tok in keys:
        base = re.split(r"[/]", tok)[0]
        for p in (base, re.sub(r"_(desc.*|t\d_desc|quip|lore|role|playstyle|\w+)$", "", base)):
            if p in prefixes:
                cur = prefixes[p]
                break
        else:
            for p, hk in prefixes.items():
                if base.startswith(p + "_"):
                    cur = hk
                    break
        owner[tok] = cur
    return owner


def hero_tokens(heroes, field, lang_base):
    """Биографии, названия и описания умений, имена в текстах о герое."""
    known = (ROOT / "vendor/citadel_heroes_english.txt").read_text(encoding="utf-8-sig")
    keys = re.findall(r'^\s*"([^"]+)"\s*"', lang_base, re.M)
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
            add(f'{a["key"]}_quip', a[f"flavor_{field}"])
    # Названия умений встречаются и в других строках того же героя: эффекты
    # (бафы при нажатии), подсказки кнопок, подписи характеристик, улучшения.
    # Заменяем их только в строках «своего» героя, чтобы общее слово вроде
    # Grapple у одного героя не задело текст другого.
    values = dict(re.findall(r'^\s*"([^"]+)"\s*"((?:[^"\\]|\\.)*)"', lang_base, re.M))
    owner = token_owners(heroes, keys)
    renames = {}
    for h in heroes:
        forms = {values[a["key"]]: a[field] for a in h["abilities"]
                 if values.get(a["key"]) and values[a["key"]] != a[field]}
        renames[h["key"]] = renamer(forms, quoted_stems=(field == "ru"))
    for tok in keys:
        rn = renames.get(owner.get(tok))
        if not rn or isinstance(out.get(tok), str) or tok.endswith("_role"):
            continue
        prev = out.get(tok)
        out[tok] = rn if prev is None else (lambda raw, p=prev, r=rn: r(p(raw)))
    # описания умений, переписанные целиком (механика и подстановки те же)
    for tok, texts in DESCRIPTIONS.items():
        add(tok, texts[field], insert=True)
    return out


def item_tokens(heroes, field, lang_base):
    """Названия предметов; в поиске остаётся и старое название."""
    out = {}
    for it in ITEMS:
        k, name = it["key"], it[field]
        if f'"{k}"' in lang_base:
            out[k] = name
            out[f"{k}_search"] = lambda raw, n=name: f"{esc(n)} {raw}"
    return out


def item_desc_tokens(heroes, field, lang_base):
    """Описания предметов (новые — дописываются) и названия предметов в
    остальных строках файла: эффекты, подписи. Одиночное слово меняем только
    при полном совпадении строки, чтобы не задеть обычные слова в текстах."""
    out = {tok: texts[field] for tok, texts in ITEM_DESCRIPTIONS.items()}
    names = {it["key"]: it for it in ITEMS}
    if field == "ru":
        base_names = (ROOT / "vendor/citadel_gc_mod_names_russian.txt").read_text(encoding="utf-8-sig")
        orig = dict(re.findall(r'^\s*"([^"]+)"\s*"((?:[^"\\]|\\.)*)"', base_names, re.M))
    else:
        orig = {k: it["orig"] for k, it in names.items()}
    forms = {orig[k]: it[field] for k, it in names.items() if orig.get(k)}
    # названия-характеристики (Spirit Lifesteal, Extra Health...) в текстах
    # означают саму характеристику, а не предмет — их внутри текста не трогаем
    stat = re.compile(r"(?i)lifesteal|resist|resilience|extra |shielding|booster|reducer|"
                      r"вампиризм|сопротивл|дополнительн|усилитель|снижатель")
    multi = {o: n for o, n in forms.items() if " " in o and not stat.search(o)}
    rn = renamer(multi, quoted_stems=(field == "ru"))
    for tok, val in re.findall(r'^\s*"([^"]+)"\s*"((?:[^"\\]|\\.)*)"', lang_base, re.M):
        # подписи характеристик (Bullet Lifesteal и т.п.) — это не названия предметов
        if tok in out or tok.endswith(("_label", "_prefix", "_postfix")):
            continue
        if val in forms:
            out[tok] = forms[val]
        elif rn and rn(val) != val:
            out[tok] = rn
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
            ("citadel_gc_mod_names", item_tokens, False),
            ("citadel_mods", item_desc_tokens, False),
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
    print(f"{len(heroes)} героев, {n_ab} умений, {len(DESCRIPTIONS)} описаний, {len(ITEMS)} предметов ({len(ITEM_DESCRIPTIONS)} описаний)"
          f" -> {pak.relative_to(ROOT)}, {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
