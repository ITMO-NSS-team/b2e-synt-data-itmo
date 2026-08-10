#!/usr/bin/env python
"""Two learning curves for the RQ4 run: pass rate and composite score by epoch.

Both charts show the same three arms over the same nine epochs, so they are read
together: the first asks "did it get more answers right", the second "was it a
better answer when it did". Epoch 0 is drawn but marked pre-treatment — no arm
carried memory then, so any gap there is draw luck and is not part of the
result.

The composite is ``sim.research.evaluate.quality``: zero for a wrong answer,
otherwise ``55*api_validity + 30*efficiency + 15*presentation``. Two of those
three terms are degenerate in this run and the subtitle says so rather than
letting the axis imply more than was measured — the run used ``--skip-spans``,
so ``api_validity`` has no per-call evidence and sits at its neutral 1.0, and
the presentation judge never ran, so its weight is 0. What the composite adds
over the pass rate is therefore the efficiency term: cost against the median for
that question class.
"""
from __future__ import annotations

import json
import statistics
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim.research.evaluate import (CorrectnessResult, TraceFacts,  # noqa: E402
                                   quality)

RUN = Path("var/rq4/rq4-2026-08-09")
OUT = Path("results/rq4-2026-08-09")

# Categorical slots 1-3 of the validated reference palette, assigned in fixed
# order and never cycled. Validated for the light surface: lightness band,
# chroma floor, CVD separation (worst adjacent deutan dE 9.2) and normal-vision
# floor (27.6) all pass. Slot 3 warns on contrast vs the surface, which is why
# every series also carries a direct label at its right end.
SERIES = {
    "A1": ("#2a78d6", "A1 · no memory"),
    "A2": ("#eb6834", "A2 · isolated"),
    "A3": ("#1baf7a", "A3 · shared"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


def load() -> list[dict]:
    return [json.loads(l) for l in (RUN / "turns.jsonl").open(encoding="utf-8")]


def class_medians(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Median cost per question class, pooled across arms.

    Pooled deliberately: a per-arm median would normalise each arm against
    itself and hide exactly the between-arm cost difference the chart is for.
    """
    buckets: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r["scored"]:
            buckets.setdefault(r["question_class"] or r["category"], []).append(
                (r["tokens"], r["wall_seconds"] or 0.0))
    return {k: (max(statistics.median(t for t, _ in v), 1.0),
                max(statistics.median(s for _, s in v), 1e-6))
            for k, v in buckets.items()}


def composite(r: dict, medians: dict[str, tuple[float, float]]) -> float | None:
    if not r["scored"]:
        return None
    facts = TraceFacts(
        http_statuses=(), heimdall_calls=int(r["heimdall_calls"]),
        rows_returned=(), error_codes=(), repeated_calls=0,
        pagination_walks=0, columns_requested=(), tokens=int(r["tokens"]),
        seconds=float(r["wall_seconds"] or 0.0),
        memory_tokens=int(r.get("memory_tokens") or 0))
    mt, ms = medians.get(r["question_class"] or r["category"], (1.0, 1e-6))
    return quality(CorrectnessResult(correct=bool(r["correct"]), scored=True,
                                     reason=""),
                   facts, median_tokens=mt, median_seconds=ms,
                   presentation=0.0, judge_weight=0.0)


def series_by_epoch(rows: list[dict], fn) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for arm in SERIES:
        vals = []
        for e in range(9):
            got = [fn(r) for r in rows
                   if r["arm"] == arm and r["epoch"] == e and r["scored"]]
            got = [g for g in got if g is not None]
            vals.append(sum(got) / len(got) if got else None)
        out[arm] = vals
    return out


def draw(data: dict[str, list[float | None]], *, title: str, subtitle: str,
         ylabel: str, path: Path, pct: bool) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = list(range(9))
    # Pre-treatment shading: epoch 0 predates any memory, so the arms are the
    # same configuration and the spread there is sampling noise, not effect.
    # The band is labelled in the subtitle rather than inside the plot: an
    # in-plot note anchored to the top of the axes collides with the subtitle
    # as soon as the subtitle wraps to three lines.
    ax.axvspan(-0.35, 0.5, color="#000000", alpha=0.045, lw=0)

    for arm, (colour, label) in SERIES.items():
        ys = data[arm]
        ax.plot(xs, ys, color=colour, lw=2.0, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.4, label=label,
                zorder=3, clip_on=False)
        last = next((v for v in reversed(ys) if v is not None), None)
        if last is not None:
            ax.annotate(arm, xy=(8, last), xytext=(11, 0),
                        textcoords="offset points", color=colour,
                        fontsize=10, fontweight="bold", va="center")

    ax.set_xlabel("epoch", fontsize=10, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xlim(-0.35, 8.6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")

    # Title and subtitle are placed on the FIGURE, not the axes, with the top
    # margin reserved for them. Two earlier attempts failed here and both are
    # worth remembering: an unwrapped subtitle is one very long text artist, so
    # `bbox_inches="tight"` widens the canvas to fit it and squeezes the plot
    # into a third of the image; and an axes-level title with padding collides
    # with the subtitle as soon as the subtitle wraps, because the padding is
    # in points while the text position is in axes fractions.
    wrapped = textwrap.fill(subtitle, width=100)
    lines = wrapped.count("\n") + 1
    top = 1 - (0.052 + 0.028 * lines)
    fig.subplots_adjust(top=top, left=0.085, right=0.955, bottom=0.20)
    fig.text(0.085, 0.975, title, fontsize=13.5, color=INK, va="top",
             fontweight="bold")
    fig.text(0.085, 0.918, wrapped, fontsize=8.5, color=INK_MUTED, va="top",
             linespacing=1.5)

    leg = ax.legend(frameon=False, fontsize=9, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), ncols=3)
    for text in leg.get_texts():
        text.set_color(INK_MUTED)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    rows = load()
    medians = class_medians(rows)

    draw(series_by_epoch(rows, lambda r: 1.0 if r["correct"] else 0.0),
         title="Pass rate by epoch — did memory make the agent more correct?",
         subtitle="Shaded epoch 0 is pre-treatment — no arm has memory there. "
                  "810 turns, one replication; all three arms answer the same "
                  "30 questions each epoch, so differences are paired per "
                  "question.",
         ylabel="pass rate (correct / scored)", pct=True,
         path=OUT / "learning-curve-pass-rate.png")

    draw(series_by_epoch(rows, lambda r: composite(r, medians)),
         title="Composite score by epoch — and was the answer cheaper when right?",
         subtitle="Shaded epoch 0 is pre-treatment. quality = 0 if wrong, else "
                  "55·api_validity + 30·efficiency + 15·presentation. In this "
                  "run api_validity is neutral (spans not collected) and the "
                  "presentation weight is 0 (judge unvalidated), so the live "
                  "terms are correctness and efficiency.",
         ylabel="composite score (0–100)", pct=False,
         path=OUT / "learning-curve-composite.png")


if __name__ == "__main__":
    main()
