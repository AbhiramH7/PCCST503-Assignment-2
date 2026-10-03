# Vector Embedding for Capability Composition (PCCST503 – Assignment 2)

A problem-specific embedding of **states, goals and executable capabilities** that preserves
*directed compatibility* (effects/outputs → preconditions/inputs), *conflicts*, *goal relevance*,
and supports **closed, associative composition** of capabilities. The planner from Assignment 1 is
out of scope: nothing here searches for action sequences.

**Idea in one paragraph.** Every condition becomes an *atom* `(key, value)` with a hashed random code
`[literal | variable]`. A capability keeps two bundles, **Req** (preconditions, constraints, required
inputs, resources) and **Prov** (effects, outputs). Inner products count *matches* and *conflicts*,
so `Prov_i · Req_j` gives directed compatibility, while cosine of `[Req | Prov | Type]` gives functional
*similarity* — two different questions. Composition is a guarded update on (Req, Prov): satisfied
requirements cancel, overwritten effects are replaced. Cost/reliability/availability live in a separate
profile (ablation shows why). Full design: [`docs/report.md`](docs/report.md).

## Deliverables → files
| Deliverable | Where |
|---|---|
| 1. Formal embedding design | `docs/report.md` §4–§6 |
| 2. Implementation (`encode`, `compose`, `similarity`, …) | `capemb/` |
| 3. Experimental dataset (3 formal problems) | `data/ecommerce.json`, `travel.json`, `pipeline.json` |
| 4. Technical report (12 sections) | `docs/report.md` |
| Experiments, logs, figures | `experiments/run_experiments.py`, `results/` |
| Tests | `tests/test_embedding.py` |

## Quick start
```bash
pip install -r requirements.txt
pytest -q                                   # 10 tests
python experiments/run_experiments.py       # ~2 min; --quick for a fast run
```
```python
from capemb import Problem, CapabilityEmbedder
p = Problem.load("data/ecommerce.json")
E = CapabilityEmbedder(p)                   # d=2048, deterministic
L = {c.name: c for c in E.encode_library()}

E.compat(L["CreateOrder"], L["MakePayment"]).composable     # True
E.compat(L["CreateOrder"], L["CancelCart"])                 # mismatched≈1 -> conflict
cp = E.compose([L["CreateOrder"], L["MakePayment"], L["SendNotification"]], name="CompletePurchase")
s, g = E.encode_state(), E.encode_goal()
E.satisfies(E.apply(s, cp), g)              # 1.0
E.similarity(L["CreateOrder"], L["CreateOrder_DB"])         # ~0.89: same function, different mechanism
E.relevance(E.encode_library(), s, g)       # goal relevance per capability
```

## Headline results (10 seeds, 3 domains)
| | ours | baseline |
|---|---|---|
| Compatibility F1 (vs symbolic reference) | **1.00** | 0.44–0.50 (symmetric cosine, oracle threshold) |
| Goal-relevance AUC | **1.00** | 0.66–0.73 on travel/pipeline (cos(Prov, goal)) |
| Composition (224 chains): Req/Prov F1, associativity, order recovery, invalid-pair rejection | **1.00** | Req F1 0.72–0.91 (naive sum) |
| Alternatives (API/DB/GUI) | cos φ 0.87–0.89, never identical, all compose with `MakePayment` | – |
| Cost | ≈5 µs / pair, 8 832 floats / capability | – |

Perfect scores partly reflect that the reference semantics is the same atom model (see report §8, §11);
the dimension sweep (`results/dim_sweep.png`) shows where the representation breaks.

## Repo layout
```
capemb/       model.py (formal model) · embedding.py (embedding system) · reference.py (symbolic ground truth)
data/         three formally specified application problems (JSON)
experiments/  run_experiments.py  (E1–E5, consistency, efficiency, dimension sweep)
results/      experiment_log.md · results.json · compat_heatmap.png · dim_sweep.png
tests/        pytest suite
docs/         report.md
```
