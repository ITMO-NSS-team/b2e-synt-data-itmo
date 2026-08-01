"""Единая точка входа: собрать каталог, корпус, документацию, проверить, поднять API."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _catalog(args) -> int:
    from heimdall.catalog.build import main as build_catalog
    return build_catalog(["--openapi", args.openapi, "--overlay-dir", args.overlay,
                          "--out", args.out])


def _build(args) -> int:
    from b2e.build import build
    build(args.seed, args.n, args.out, args.catalog)
    return 0


def _validate(args) -> int:
    from b2e.validate import main as validate_main
    return validate_main(["--data", args.data, "--catalog", args.catalog]
                         + (["--json", args.json] if args.json else []))


def _doc(args) -> int:
    from b2e.htmldoc import write_docs
    path = write_docs(args.catalog, args.data, args.out)
    print(f"документация: {path}")
    return 0


def _serve(args) -> int:
    import uvicorn
    from heimdall.app import create_app
    app = create_app(args.data, args.catalog, skills_root=args.skills)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _stats(args) -> int:
    from b2e.report import realism_report
    print(realism_report(args.data, args.catalog))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="b2e", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalog", help="собрать каталог из спецификации OpenAPI")
    c.add_argument("--openapi", required=True)
    c.add_argument("--overlay", default="catalog")
    c.add_argument("--out", default="catalog/snapshot.json")
    c.set_defaults(fn=_catalog)

    b = sub.add_parser("build", help="собрать корпус данных")
    b.add_argument("--seed", type=int, default=20260801)
    b.add_argument("--n", type=int, default=300_000)
    b.add_argument("--out", default="data")
    b.add_argument("--catalog", default="catalog/snapshot.json")
    b.set_defaults(fn=_build)

    v = sub.add_parser("validate", help="гейт согласованности")
    v.add_argument("--data", default="data")
    v.add_argument("--catalog", default="catalog/snapshot.json")
    v.add_argument("--json", default="")
    v.set_defaults(fn=_validate)

    d = sub.add_parser("doc", help="HTML-документация витрин")
    d.add_argument("--catalog", default="catalog/snapshot.json")
    d.add_argument("--data", default="data")
    d.add_argument("--out", default="docs/html")
    d.set_defaults(fn=_doc)

    s = sub.add_parser("serve", help="поднять эмулятор Heimdall")
    s.add_argument("--data", default="data")
    s.add_argument("--catalog", default="catalog/snapshot.json")
    s.add_argument("--skills", default="heimdall-skills")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=_serve)

    r = sub.add_parser("stats", help="отчёт о правдоподобии корпуса")
    r.add_argument("--data", default="data")
    r.add_argument("--catalog", default="catalog/snapshot.json")
    r.set_defaults(fn=_stats)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
