#!/usr/bin/env python3
"""lang_extract.py: the mod's interface texts in English, for its translations (the user, 2026-11-14: "the Earth
interface in French").

    python3 tools/lang_extract.py > strings.json

Collects: the Vue screens' template texts and their title / placeholder / aria-label / label attributes, the quoted
sentences of their scripts, the settings pages' names, descriptions, notes, intros and option labels
(earthSettingsStore.js), settings.lua's labels, the loading line's parts (loadprogress.lua) and the banner's lines
(banner.lua). A text with a {{ }} or ${} in it is left out (its pieces are). Prints a sorted JSON list."""
import glob
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(ROOT, "mod", "beamng_earth", "ui", "ui-vue", "mods", "beamngEarth")
LUA = os.path.join(ROOT, "mod", "beamng_earth", "lua", "ge", "extensions", "beamngEarth")
WORDY = re.compile(r"[A-Za-z]{2}")


def norm(s):
    return " ".join(html.unescape(s.replace('\\"', '"').replace("\\'", "'")).split())


def keep(s):
    return bool(WORDY.search(s)) and not re.search(r"\{\{|\$\{|^[a-z_]+\(|=>|^[\w.-]+\.(vue|js|lua|dds|png|json)$", s)


def extract():
    out = set()
    for f in glob.glob(os.path.join(UI, "*.vue")):
        text = open(f, encoding="utf-8").read()
        tpl, _, script = text.partition("<script")
        tpl = re.sub(r"<!--.*?-->", "", tpl, flags=re.S)
        tpl = re.sub(r"<style.*?</style>", "", tpl, flags=re.S)
        for m in re.finditer(r">([^<>]+)<", tpl):
            for piece in re.split(r"\{\{.*?\}\}", m.group(1), flags=re.S):
                piece = norm(piece.strip(" ·|:"))
                if keep(piece):
                    out.add(piece)
        for m in re.finditer(r'(?<![:@\w-])(?:title|placeholder|aria-label|label)="([^"]+)"', tpl):
            out.add(norm(m.group(1)))
        for m in re.finditer(r'"((?:[^"\\\n]|\\.){4,})"', script.split("</script>")[0]):
            s = norm(m.group(1))
            if re.search(r"[A-Za-z]+ [a-z]+", s) and keep(s) and not re.search(r"[<>]|^\w+[./]\w", s):
                out.add(s)
    store = open(os.path.join(UI, "earthSettingsStore.js"), encoding="utf-8").read()
    for m in re.finditer(r'\b(?:name|desc|note|intro|label|title|text)\s*:\s*"((?:[^"\\]|\\.)*)"', store):
        s = norm(m.group(1))
        if keep(s):
            out.add(s)
    for name in ("settings.lua", "loadprogress.lua", "banner.lua"):
        src = open(os.path.join(LUA, name), encoding="utf-8").read()
        src = re.sub(r"--[^\n]*", "", src)
        for m in re.finditer(r"'((?:[^'\\\n]|\\.){2,})'", src):
            s = norm(m.group(1))
            if (" " in s or name != "loadprogress.lua") and keep(s) and not re.search(r"^[\w/.]+$|%[sdf]", s):
                if name == "settings.lua" and not re.search(r"\blabel\s*=\s*'" + re.escape(m.group(1)), src):
                    continue
                out.add(s)
    return sorted(out)


if __name__ == "__main__":
    json.dump(extract(), sys.stdout, ensure_ascii=False, indent=1)
    print()
