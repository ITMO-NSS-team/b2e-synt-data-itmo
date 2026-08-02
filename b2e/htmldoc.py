"""Статическая HTML-документация корпуса: витрины, колонки, примеры значений.

Документация генерируется из каталога и **из самих данных**: рядом с типом
колонки стоит реальный пример. Это отвечает на вопрос, на который спецификация
не отвечает — «а что там лежит на самом деле».

Страница одна, без внешних файлов: её удобно открыть с диска и приложить к
отчёту. Тёмная тема учитывается через ``prefers-color-scheme``.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from heimdall.catalog import Catalog

from b2e.store import ProceduralSnapshot
from b2e.traps import describe as describe_traps

_CSS = """
:root { --bg:#fff; --fg:#16181d; --muted:#6b7280; --line:#e5e7eb; --accent:#0b7285;
        --code:#f6f8fa; --warn:#b45309; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e6e8eb; --muted:#9aa3af; --line:#262b33;
          --accent:#4dd0e1; --code:#161a21; --warn:#f59e0b; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:32px 20px 96px; }
h1 { font-size:30px; margin:0 0 4px; letter-spacing:-.02em; }
h2 { font-size:21px; margin:44px 0 10px; padding-top:14px; border-top:1px solid var(--line); }
h3 { font-size:16px; margin:26px 0 8px; }
.sub { color:var(--muted); margin:0 0 26px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
     letter-spacing:.04em; }
code,.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
td.mono { color:var(--accent); white-space:nowrap; }
.sample { color:var(--muted); max-width:430px; overflow:hidden; text-overflow:ellipsis;
          white-space:nowrap; }
.scroll { overflow-x:auto; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:10px; }
.card { border:1px solid var(--line); border-radius:9px; padding:12px 14px; }
.card b { display:block; font-size:13px; }
.card span { color:var(--muted); font-size:12px; }
.tag { display:inline-block; padding:1px 7px; border-radius:20px; font-size:11px;
       border:1px solid var(--line); color:var(--muted); margin-left:6px; }
.warn { color:var(--warn); }
a { color:var(--accent); }
.toc { columns:3; column-gap:24px; font-size:13px; }
@media (max-width:760px){ .toc{columns:1;} }
"""


def _sample(reader, name: str, limit: int = 2) -> str:
    try:
        values = reader.sample(name, limit)
    except Exception:
        return ""
    parts = []
    for v in values:
        text = (json.dumps(v, ensure_ascii=False, default=str)
                if isinstance(v, (list, dict)) else str(v))
        parts.append(text[:70])
    return " · ".join(parts)


def write_docs(catalog_path: str | Path, data_root: str | Path,
               out_dir: str | Path) -> Path:
    catalog = Catalog.load(catalog_path)
    snap = ProceduralSnapshot(data_root, catalog)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = catalog.counts()
    people = snap.manifest.get("people", "—")
    parts: list[str] = [
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Корпус B2E — витрины Heimdall</title><style>{_CSS}</style></head><body>",
        "<div class='wrap'>",
        "<h1>Синтетический HR-корпус B2E</h1>",
        f"<p class='sub'>Снимок <code>{html.escape(str(snap.snapshot_id))}</code> · "
        f"{people} сотрудников · {counts['models']} витрин · "
        f"{counts['columns']} колонок · {counts['metrics']} метрик · "
        f"срез на {html.escape(str(snap.manifest.get('as_of', '')))}</p>",
    ]

    parts.append("<div class='grid'>")
    for key in snap.keys():
        rows = snap.rows(key)
        parts.append(f"<div class='card'><b>{html.escape(key)}</b>"
                     f"<span>{rows} строк</span></div>")
    parts.append("</div>")

    parts.append("<h2>Каверзы</h2>")
    parts.append("<p class='sub'>Ситуации, где запрос возвращает 200 OK и неверный "
                 "ответ. Включаются флагом, чтобы эксперимент можно было провести "
                 "с ними и без них.</p>")
    parts.append(_md_table(describe_traps(set(snap.manifest.get("traps", [])))))

    parts.append("<h2>Содержание</h2><div class='toc'>")
    for key in sorted(catalog.models):
        parts.append(f"<div><a href='#{html.escape(key)}'>{html.escape(key)}</a></div>")
    parts.append("</div>")

    for key in sorted(catalog.models):
        model = catalog.models[key]
        reader = snap.table(key) if snap.has(key) else None
        rows = reader.rows if reader else 0
        materialised = set(reader._index) if reader else set()
        parts.append(f"<h2 id='{html.escape(key)}'>{html.escape(key)}"
                     f"<span class='tag'>{rows} строк</span>"
                     + ("<span class='tag'>история</span>" if model.is_history else "")
                     + "".join(f"<span class='tag'>{c}</span>" for c in model.channels)
                     + "</h2>")
        if model.summary:
            parts.append(f"<p class='sub'>{html.escape(model.summary[:400])}</p>")
        parts.append("<div class='scroll'><table><thead><tr>"
                     "<th>колонка</th><th>тип</th><th>источник</th>"
                     "<th>пример</th><th>описание</th></tr></thead><tbody>")
        for name, member in list(model.columns.items()):
            if member.kind == "param_column":
                origin = "параметрическая"
                sample = ""
            elif name in materialised:
                origin = "витрина"
                sample = _sample(reader, name)
            else:
                origin = "процедурная"
                sample = _sample(reader, name) if reader and rows else ""
            parts.append(
                f"<tr><td class='mono'>{html.escape(name)}</td>"
                f"<td class='mono'>{html.escape(member.type)}</td>"
                f"<td>{origin}</td>"
                f"<td class='sample'>{html.escape(sample)}</td>"
                f"<td>{html.escape((member.description or '')[:150])}</td></tr>")
        parts.append("</tbody></table></div>")
        if model.metrics:
            parts.append("<h3>Метрики</h3><div class='scroll'><table><thead><tr>"
                         "<th>метрика</th><th>тип</th><th>описание</th>"
                         "</tr></thead><tbody>")
            for name, member in model.metrics.items():
                parts.append(f"<tr><td class='mono'>{html.escape(name)}</td>"
                             f"<td class='mono'>{html.escape(member.kind)}</td>"
                             f"<td>{html.escape((member.description or '')[:150])}</td></tr>")
            parts.append("</tbody></table></div>")

    parts.append("</div></body></html>")
    path = out / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _md_table(markdown: str) -> str:
    """Небольшая таблица из Markdown — чтобы реестр каверз жил в одном месте."""
    lines = [l for l in markdown.strip().splitlines() if l.strip()]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    body = [[c.strip() for c in row.strip("|").split("|")] for row in lines[2:]]
    out = ["<div class='scroll'><table><thead><tr>"]
    out += [f"<th>{html.escape(h)}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(
            f"<td>{html.escape(c).replace('`', '')}</td>" for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)
