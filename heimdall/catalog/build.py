"""Сборка снимка каталога из спеки OpenAPI Heimdall.

Почему из OpenAPI, а не из рантайма ``describe_model``
-----------------------------------------------------
Описание каждого REST-пути модели содержит пять размеченных секций:

    Доступные поля:<br>- `имя` (Тип): описание<br>- …
    Виртуальные колонки:<br>…
    Виртуальные колонки с параметрами:<br>…
    Доступные метрики:<br>…
    Доступные параметризованные метрики:<br>…

То есть параметрические члены в спеке **размечены** — их не надо угадывать по
наличию ``param_columns`` в рецепте. Enum ``*_Cols`` даёт авторитетный список
имён, описание — типы и русские тексты. Объединение полное.

Чего в спеке действительно нет
------------------------------
* имена и типы аргументов параметрических членов → ``catalog/param_args.yaml``;
* тип элемента массива (спека пишет просто ``Array``) → ``catalog/array_elements.yaml``;
* физическое имя таблицы и видимость в каналах v1/orion → ``catalog/channels.yaml``.

Все три файла курируемые, маленькие и помечены как требующие подтверждения
командой сервиса.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .model import Catalog, Member, Model, TimeDim

SECTIONS = {
    "Доступные поля:": "column",
    "Виртуальные колонки:": "virtual",
    "Виртуальные колонки с параметрами:": "param_column",
    "Доступные метрики:": "metric",
    "Доступные параметризованные метрики:": "param_metric",
}

_ITEM = re.compile(r"- `([^`]+)` \((.*?)\): (.*?)(?=<br>- `|<br><br>|\Z)", re.S)
_EMPTY = "Нет доступных колонок"

#: Эвристика типа для имён, которые есть в enum, но не описаны в спеке
#: (колонки, появляющиеся из ARRAY JOIN у моделей-развёрток).
_NAME_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    (r"_dt$|_date$|date_", "Date32 (nullable)"),
    (r"_at$|_time$|_ts$", "DateTime (nullable)"),
    (r"^is_|_flag$|^flag_", "UInt8 (nullable)"),
    (r"_count$|_cnt$|_qty$|_num$", "UInt64 (nullable)"),
    (r"_id$", "String (nullable)"),
    (r"_status$|_type$|_format_name$", "String (LowCardinality) (nullable)"),
)


def _split_sections(desc: str) -> dict[str, str]:
    found = sorted((desc.find(s), s) for s in SECTIONS if desc.find(s) >= 0)
    out: dict[str, str] = {}
    for i, (pos, name) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(desc)
        out[name] = desc[pos + len(name):end]
    return out


def _parse_items(text: str) -> list[tuple[str, str, str]]:
    if not text or _EMPTY in text:
        return []
    return [(m.group(1), m.group(2).strip(), _clean(m.group(3)))
            for m in _ITEM.finditer(text)]


def _clean(s: str) -> str:
    """Убрать хвост следующей секции и схлопнуть дубли в описании.

    Спека часто пишет описание дважды: «Название. Название» — это артефакт
    склейки title и description на стороне сервиса.
    """
    s = s.split("\n\n")[0].strip()
    for marker in SECTIONS:
        s = s.split(marker)[0].strip()
    s = s.replace("<br>", " ").strip()
    if "." in s:
        head, _, tail = s.partition(".")
        if head.strip() and head.strip() == tail.strip().rstrip("."):
            return head.strip()
    return s


def _guess_type(name: str) -> str:
    for pattern, chtype in _NAME_TYPE_HINTS:
        if re.search(pattern, name):
            return chtype
    return "String (nullable)"


def _load_overlay(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_catalog(openapi_path: str | Path, overlay_dir: str | Path | None = None) -> Catalog:
    """Собрать снимок каталога. Детерминированно: тех же входах — тот же выход."""
    src = Path(openapi_path)
    raw = src.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    schemas = spec["components"]["schemas"]
    paths = spec["paths"]

    overlay_dir = Path(overlay_dir) if overlay_dir else src.parent / "catalog"
    param_args = _load_overlay(overlay_dir / "param_args.yaml")
    array_elems = _load_overlay(overlay_dir / "array_elements.yaml")
    channels_cfg = _load_overlay(overlay_dir / "channels.yaml")

    cat = Catalog(source=src.name, source_sha256=hashlib.sha256(raw).hexdigest())

    for path in sorted(paths):
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "api" or parts[1] != "v1" or "{" in path:
            continue
        schema, logic_model = parts[2], parts[3]
        op = paths[path].get("post")
        if not op:
            continue
        desc = op.get("description", "")
        if "Доступные поля" not in desc:
            continue

        model = Model(schema=schema, logic_model=logic_model,
                      summary=(op.get("summary") or "").strip(), rest_path=path,
                      deprecated=logic_model.endswith("_dep"))

        sections = _split_sections(desc)
        documented: set[str] = set()
        for section, kind in SECTIONS.items():
            for name, chtype, text in _parse_items(sections.get(section, "")):
                member = Member(name=name, type=chtype, description=text, kind=kind)
                target = model.metrics if member.is_metric else model.columns
                target[name] = member
                documented.add(name)

        # Enum *_Cols / *_Metrics — авторитетный список имён. У моделей-развёрток
        # (employee_education, employee_oshs) часть колонок приходит из ARRAY JOIN
        # и в описании отсутствует: их типы восстанавливаем по имени.
        for enum_suffix, container, kind in (("_Cols", model.columns, "column"),
                                             ("_Metrics", model.metrics, "metric")):
            enum = schemas.get(f"{logic_model}{enum_suffix}", {}).get("enum", [])
            for name in enum:
                if name in documented or name in container:
                    continue
                container[name] = Member(name=name, type=_guess_type(name),
                                         description="", kind=kind)
                model.undocumented.append(name)
        model.undocumented.sort()

        td_names = schemas.get(f"{logic_model}_TimeDimNames", {}).get("enum", [])
        grans = schemas.get(f"{logic_model}_Granularities", {}).get("enum", [])
        for name in td_names:
            model.time_dimensions[name] = TimeDim(name=name, granularities=list(grans))

        cat.models[model.key] = model

    _apply_overlays(cat, param_args, array_elems, channels_cfg)
    return cat


def _apply_overlays(cat: Catalog, param_args: dict, array_elems: dict, channels: dict) -> None:
    """Наложить курируемые дополнения к тому, чего нет в спеке."""
    for key, members in (param_args.get("members") or {}).items():
        model = cat.models.get(key)
        if not model:
            continue
        for name, params in members.items():
            member = model.member(name)
            if member is not None:
                member.parameters = list(params or [])

    for key, cols in (array_elems.get("models") or {}).items():
        model = cat.models.get(key)
        if not model:
            continue
        for name, elem in cols.items():
            member = model.columns.get(name)
            if member is not None:
                member.element_type = elem

    tables = channels.get("tables") or {}
    lookback = channels.get("default_lookback_years") or {}
    v1 = set(channels.get("v1") or [])
    orion = set(channels.get("orion") or [])
    for key, model in cat.models.items():
        if key in tables:
            model.table = tables[key]
        if key in lookback:
            model.default_lookback_years = int(lookback[key])
        chans = ["v2"]
        if key in v1:
            chans.append("v1")
        if key in orion:
            chans.append("orion")
        model.channels = chans


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Собрать снимок каталога Heimdall из OpenAPI")
    ap.add_argument("--openapi", default="Heimdall_openapi.json")
    ap.add_argument("--overlay-dir", default="catalog")
    ap.add_argument("--out", default="catalog/snapshot.json")
    args = ap.parse_args(argv)

    cat = build_catalog(args.openapi, args.overlay_dir)
    cat.save(args.out)
    c = cat.counts()
    print(f"моделей={c['models']} схем={c['schemas']} колонок={c['columns']} метрик={c['metrics']}")
    print(f"историй={len(cat.history_models())} → {[m.key for m in cat.history_models()]}")
    undoc = sum(len(m.undocumented) for m in cat)
    print(f"имён без описания в спеке: {undoc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
