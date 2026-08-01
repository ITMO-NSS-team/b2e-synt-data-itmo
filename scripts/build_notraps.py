#!/usr/bin/env python3
"""Build a traps-off corpus.

``traps_enabled`` is the RQ1 independent variable, and it cannot be a pure
runtime switch. Channel quirks (``heimdall/engine/quirks.py``) toggle per
request, but the data traps in ``b2e/traps.py`` are baked in at build time —
they have to be, because a trap keyed to a *person* must be stable across all 37
marts, or the same employee would appear uppercased on one mart and not another
and the trap would degrade into an identity inconsistency.

So a traps-off run needs a traps-off corpus. ``b2e.build.build`` already accepts
``enabled_traps``; the CLI simply has no flag for it, which is why this script
exists rather than a patch to ``b2e/cli.py`` — the corpus generator is the
measuring instrument, and the simulation environment should not be editing it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out", default="data-small-notraps")
    parser.add_argument("--catalog", default="catalog/snapshot.json")
    args = parser.parse_args()

    from b2e.build import build

    out = Path(args.out)
    print(f"building traps-off corpus: n={args.n} seed={args.seed} -> {out}")
    summary = build(seed=args.seed, n=args.n, out=out,
                    catalog_path=args.catalog, enabled_traps=set())

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("traps"):
        raise SystemExit(
            f"refusing to finish: manifest still lists traps {manifest['traps']}. "
            f"A corpus labelled traps-off that contains traps would silently "
            f"corrupt every RQ1 comparison made against it.")

    print(f"snapshot_id = {manifest['snapshot_id']}")
    print(f"people      = {manifest['people']}")
    print(f"traps       = {manifest['traps'] or 'none'}")
    print(f"seconds     = {summary.get('seconds')}")
    print()
    print("set DATA_DIR_NOTRAPS in deploy/.env to this directory, then the "
          "admin UI can switch traps_enabled to false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
