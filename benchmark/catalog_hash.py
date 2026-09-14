"""Hash skill file paths and bytes, excluding README files like Registry.load.
It needs to fix skill set for the gold dataset (basket)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def catalog_hash(root: str | Path) -> str:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = sorted(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*")
        if path.is_file() and path.suffix in (".yaml", ".yml", ".md")
        and path.stem.upper() != "README"
    )
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1] / "heimdall-skills"))
    print(catalog_hash(parser.parse_args().root))
