# RQ4 report — rq4-2026-08-09

_Generated 2026-08-10 05:43:12 UTC._

**9 of 9 epochs complete** (810 turns read from `turns.jsonl`). At epoch 8: A2 − A1 = +3.4pp (95% CI [-6.9pp, +17.2pp], p=1.0000, n=29 pairs) A3 − A1 = +6.9pp (95% CI [+0.0pp, +17.2pp], p=0.4998, n=29 pairs) Learning-curve slope: A2 -0.31pp/epoch, A3 -0.78pp/epoch — A2 is gaining faster on A1 across the epochs observed so far.

## 1. Health

| arm | total | scored | scored=false | scored=false rate | dispatch_failed |
|---|---:|---:|---:|---:|---:|
| A1 | 270 | 266 | 4 | 1.5% | 0 |
| A2 | 270 | 263 | 7 | 2.6% | 0 |
| A3 | 270 | 266 | 4 | 1.5% | 1 |

`scored=false` reasons:

| arm | reason | count |
|---|---|---:|
| A1 | no_answer_block | 4 |
| A2 | no_answer_block | 6 |
| A2 | unparseable_number | 1 |
| A3 | no_answer_block | 3 |
| A3 | dispatch_failed | 1 |

## 2. Guard

| arm | caution pass rate | n (scored/total) | followed_injection_rate | fabricated_id_rate |
|---|---:|---:|---:|---:|
| A1 | 37.9% | 87/90 | 0.0% | 12.4% |
| A2 | 40.5% | 84/90 | 0.0% | 13.2% |
| A3 | 41.4% | 87/90 | 0.0% | 11.1% |

## 3. Primary

Final complete epoch: **8** (of 9 target epochs; 9 complete so far).

| arm | pass_rate | n (correct/scored) |
|---|---:|---:|
| A1 | 33.3% | 10/30 |
| A2 | 37.9% | 11/29 |
| A3 | 41.4% | 12/29 |

Paired difference vs A1, over shared `pair_id` (permutation test, 10000 draws, two-sided; bootstrap 95% CI):

| comparison | mean diff | 95% CI | p-value | n pairs |
|---|---:|---:|---:|---:|
| A2 − A1 | +3.4pp | [-6.9pp, +17.2pp] | 1.0000 | 29 |
| A3 − A1 | +6.9pp | [+0.0pp, +17.2pp] | 0.4998 | 29 |

## 4. Learning curve

| epoch | A2 − A1 | A2 CI | A3 − A1 | A3 CI |
|---:|---:|---:|---:|---:|
| 0 | +13.3pp | [+3.3pp, +26.7pp] | +16.7pp | [+3.3pp, +30.0pp] |
| 1 | +0.0pp | [-10.7pp, +10.7pp] | -6.7pp | [-16.7pp, +0.0pp] |
| 2 | -7.1pp | [-21.4pp, +7.1pp] | +3.7pp | [+0.0pp, +11.1pp] |
| 3 | +10.0pp | [-3.3pp, +23.3pp] | +0.0pp | [-13.8pp, +13.8pp] |
| 4 | +13.3pp | [+3.3pp, +26.7pp] | +6.7pp | [-10.0pp, +23.3pp] |
| 5 | +3.4pp | [-10.3pp, +17.2pp] | +3.4pp | [+0.0pp, +10.3pp] |
| 6 | +6.7pp | [-6.7pp, +20.0pp] | -6.7pp | [-20.0pp, +6.7pp] |
| 7 | +0.0pp | [-14.8pp, +14.8pp] | -3.4pp | [-13.9pp, +6.9pp] |
| 8 | +3.4pp | [-6.9pp, +13.8pp] | +6.9pp | [+0.0pp, +17.2pp] |

Slope: A2 -0.31pp/epoch, A3 -0.78pp/epoch. "A3 learns faster than A2" is exactly this slope contrast.

## 5. Cost

Tokens/seconds normalised **per correct answer**, not per request — raw per-request cost structurally disfavours the memory arms, since memory is rendered into every prompt whether or not that turn ends up correct.

| arm | tokens/correct | wall s/correct | modelled s/correct | Heimdall calls/answer | n correct |
|---|---:|---:|---:|---:|---:|
| A1 | 3097699 | 477.17 | 0.00 | 18.31 | 81 |
| A2 | 2878171 | 439.08 | 0.00 | 19.09 | 94 |
| A3 | 3062504 | 469.82 | 0.00 | 18.86 | 87 |

Memory size (mean prompt tokens contributed by the memory block), by epoch:

| epoch | A1 | A2 | A3 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |

## 6. Honest limits

One replication means one memory trajectory per arm. The intervals above describe sampling noise in which 270 questions this replication happened to draw and how they landed on the epoch grid — not run-to-run variance in how reflection writes memory. A second replication could curate different lessons from the same question pool and move the learning curve for reasons this analysis has no way to see. Read every number here as a described difference with an interval for this one run, not as a significance claim about reflection in general.
