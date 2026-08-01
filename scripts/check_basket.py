"""Verification for sim/oracle/basket.py — counts and non-emptiness.

Imports the module by path when the package __init__ cannot be imported yet
(sim/oracle/__init__.py re-exports the S8 label module, which a parallel task is
still writing). The basket itself has no dependency on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/mosyamac/b2e-synt-data")
sys.path.insert(0, str(ROOT))

try:
    from sim.oracle import basket as B
    HOW = "import sim.oracle.basket"
except Exception as exc:  # labels.py not written yet
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b2e_basket", ROOT / "sim" / "oracle" / "basket.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    HOW = f"loaded by path (package __init__ unavailable: {type(exc).__name__}: {exc})"

print(f"loaded via: {HOW}")
print(f"total questions: {len(B.BASKET)}  (DECLARED_SIZE={B.DECLARED_SIZE})")
print(f"basket digest:   {B.basket_digest()}")

base = B.load_basket(include_paraphrases=False)
para = [q for q in B.BASKET if q.is_paraphrase]
print(f"base wordings:   {len(base)}")
print(f"paraphrases:     {len(para)}")

print("\nby family")
for fam, n in B.counts_by_family().items():
    print(f"  {fam:<22} {n:>4}   ({B.FAMILY_LABELS[fam]})")

print("\nby category")
for cat, n in B.counts_by_category().items():
    print(f"  {cat:<18} {n:>4}")

print("\ngrid  family x category")
cats = B.CATEGORIES
head = "  " + " " * 22 + "".join(f"{c[:12]:>14}" for c in cats)
print(head)
grid = B.counts_grid()
for fam in B.FAMILIES:
    row = "".join(f"{grid[fam][c]:>14}" for c in cats)
    print(f"  {fam:<22}{row}")

print("\nby acting_role")
for role in ("self", "manager", "hr"):
    print(f"  {role:<10} {sum(1 for q in B.BASKET if q.acting_role == role):>4}")

print("\nexpected_behaviour distribution")
beh: dict[str, int] = {}
for q in B.BASKET:
    beh[q.expected_behaviour] = beh.get(q.expected_behaviour, 0) + 1
for k in sorted(beh):
    print(f"  {k:<30} {beh[k]:>4}")

print("\nprompt_injection payload origin")
for kind in ("injection:prompt", "injection:data"):
    print(f"  {kind:<20} {sum(1 for q in B.BASKET if q.gold_ref == kind):>4}")

print("\nno_data anchors")
anchors: dict[str, int] = {}
for q in B.BASKET:
    if q.category == "no_data":
        anchors[q.gold_ref] = anchors.get(q.gold_ref, 0) + 1
for k in sorted(anchors):
    print(f"  {k:<52} {anchors[k]:>4}")

print(f"\ngold labels S8 must provide: {len(B.gold_refs())}")
for ref in B.gold_refs()[:5]:
    print(f"  {ref}")
print(f"  ... ({len(B.gold_refs()) - 5} more)")

# ------------------------------------------------------------------ assertions

assert len(B.BASKET) == B.DECLARED_SIZE, "size drifted from DECLARED_SIZE"

by_cat = B.counts_by_category()
for cat in B.CATEGORIES:
    assert by_cat[cat] > 0, f"category {cat} is EMPTY"

by_fam = B.counts_by_family()
for fam in B.FAMILIES:
    assert by_fam[fam] > 0, f"family {fam} is EMPTY"

for fam in B.FAMILIES:
    for cat in B.CATEGORIES:
        assert grid[fam][cat] > 0, f"cell {fam}/{cat} is EMPTY"

assert para, "no paraphrases: robustness would not be measurable"
for q in para:
    assert q.paraphrase_of in {x.id for x in B.BASKET}

# The mandatory five caution categories, spelled out rather than derived, so the
# assertion still means something if CATEGORIES is edited.
for cat in ("out_of_scope", "ambiguous", "no_data", "access_control",
            "prompt_injection"):
    assert by_cat[cat] > 0, f"mandatory caution category {cat} missing"

# The no_data honest anchor really is the mart declared empty by design.
from b2e.gen.marts import EMPTY_BY_DESIGN
assert "empty_by_design:technical.memai_memmcp_config" in anchors
assert "technical.memai_memmcp_config" in EMPTY_BY_DESIGN

# Binding: every slot renders, and a missing slot refuses.
values = {"subject": "Иванов И.И.", "peer": "Петрова А.С.",
          "unit": "Управление данных", "requisition": "REQ-10024"}
rendered = [q.bind(values) for q in B.BASKET]
assert not any("{" in t for t in rendered), "unrendered slot left in text"
slotted = [q for q in B.BASKET if q.slots]
print(f"\nquestions with slots: {len(slotted)} / {len(B.BASKET)}")

try:
    next(q for q in B.BASKET if "subject" in q.slots).bind({})
except B.BasketError as exc:
    print(f"bind() refuses on missing slot: {str(exc)[:80]}...")
else:
    raise AssertionError("bind() accepted a missing slot")

# Round trip through the registry encoding.
assert B.from_records(B.to_records()) == B.BASKET
assert B.basket_digest(B.from_records(B.to_records())) == B.basket_digest()

# Snapshot reconciliation against the real corpus.
B.check_against_snapshot(ROOT / "data-small")
print("check_against_snapshot(data-small): OK")

print("\nALL ASSERTIONS PASSED")
