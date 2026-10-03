# Experiment log

Default configuration: d=2048 (so Req/Prov bundles are R^4096), dt=128, ds=512, weights (wR,wP,wT)=(0.5,1.0,0.35); seeds=10.

## E1. Capability compatibility (ecommerce, seed 0)

| pair | matched | mismatched | dependency | composable | evidence (decoded, for inspection) |
|---|---|---|---|---|---|
| CreateOrder → MakePayment | 1.98 | -0.19 | 0.28 | **yes** | 1 state + 1 data links; conflicts=[] |
| CreateOrder → CancelCart | -0.04 | 1.06 | -0.01 | no | 0 state + 0 data links; conflicts=['Order.exists=false'] |
| CreateOrder → SendNotification | -0.04 | 0.03 | -0.02 | no | 0 state + 0 data links; conflicts=[] |

State awareness: applicability (unmet requirements) before/after executing CreateOrder

| capability | in S_I | in S' = Apply(S_I, CreateOrder) |
|---|---|---|
| CreateOrder | 0 unmet / 0 violated → applicable | 0 unmet / 0 violated → applicable |
| MakePayment | 2 unmet / 1 violated → blocked | 0 unmet / 0 violated → applicable |
| CancelCart | 0 unmet / 0 violated → applicable | 1 unmet / 1 violated → blocked |
| SendNotification | 1 unmet / 1 violated → blocked | 1 unmet / 1 violated → blocked |

Similarity ≠ composability (same pairs, different measures)

| pair | cos φ | cos Prov | baseline bag-of-atoms cos | matched/mismatched | composable |
|---|---|---|---|---|---|
| CreateOrder → CreateOrder_DB | 0.886 | 1.000 | 0.951 | -0.16/0.04 | no |
| CreateOrder → CreateOrder_GUI | 0.876 | 1.000 | 0.902 | -0.19/0.02 | no |
| CreateOrder → CancelOrder | 0.367 | 0.313 | 0.375 | 1.12/-0.15 | yes |
| CreateOrder → MakePayment | 0.124 | -0.001 | 0.384 | 1.98/-0.19 | yes |
| MakePayment → SendNotification | 0.022 | 0.001 | 0.156 | 0.96/0.09 | yes |
| CreateOrder → ArchiveLogs | 0.038 | 0.033 | -0.002 | -0.02/0.08 | no |

Compatibility over all ordered pairs vs. symbolic reference (mean over seeds)

| domain | ordered pairs | F1 (ours: Prov·Req) | F1 (symmetric cosine baseline, oracle threshold) |
|---|---|---|---|
| ecommerce | 132 pairs (13 composable) | 1.000 ± 0.000 | 0.450 ± 0.012 |
| travel | 110 pairs (8 composable) | 1.000 ± 0.000 | 0.440 ± 0.014 |
| pipeline | 90 pairs (9 composable) | 1.000 ± 0.000 | 0.500 ± 0.021 |

## E2. Capability composition (ecommerce, seed 0)

CompletePurchase = SendNotification ∘ MakePayment ∘ CreateOrder; decoded Req matches symbolic: **True**, Prov matches: **True**

- Composite Req (11 atoms) excludes internally satisfied atoms, e.g. [('Data.order_id', 'UUID|uuid'), ('Order.exists', 'true'), ('Payment.status', 'SUCCESS')]
- Naive additive composition would have kept 14 Req atoms (vs 11)

| compared with | cos φ(composite, ·) | cos Prov | cos Req |
|---|---|---|---|
| CreateOrder | 0.731 | 0.700 | 0.798 |
| MakePayment | 0.628 | 0.618 | 0.574 |
| SendNotification | 0.356 | 0.351 | 0.320 |
| ArchiveLogs (unrelated control) | 0.016 | 0.015 | 0.004 |

Order-preserving provenance block recovers: CreateOrder (0.63) → MakePayment (0.55) → SendNotification (0.52)

Goal satisfaction (fraction of G in state): before = 0.00; after Apply(S_I, CompletePurchase) = 1.00; composite applicable in S_I: True

Composite operational profile: time=1070 ms, money=0.31, risk=0.162 (=1-Π(1-r)), reliability=0.955 (=Π rel)

Aggregate composition quality vs symbolic reference over all valid connected chains (len 2-4), 10 seeds

| domain | #chains | Req F1 | Prov F1 | exact match | associativity | order recovery | invalid rejected | naive-sum Req F1 |
|---|---|---|---|---|---|---|---|---|
| ecommerce | 139 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.914 |
| travel | 43 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.818 |
| pipeline | 42 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.718 |

## E3. Alternative implementations (ecommerce: API / DATABASE / GUI)

| cos φ | CreateOrder | CreateOrder_DB | CreateOrder_GUI | MakePayment |
|---|---|---|---|---|
| CreateOrder | 1.000 | 0.886 | 0.876 | 0.124 |
| CreateOrder_DB | 0.886 | 1.000 | 0.870 | 0.005 |
| CreateOrder_GUI | 0.876 | 0.870 | 1.000 | 0.044 |
| MakePayment | 0.124 | 0.005 | 0.044 | 1.000 |

| A | B | cos Prov (function) | cos Req | cos Type/mechanism |
|---|---|---|---|---|
| CreateOrder | CreateOrder_DB | 1.000 | 0.925 | -0.125 |
| CreateOrder | CreateOrder_GUI | 1.000 | 0.850 | -0.089 |
| CreateOrder_DB | CreateOrder_GUI | 1.000 | 0.762 | 0.035 |

Type-weight ablation. 'within' = mean cos φ among the three CreateOrder implementations; 'clone' = cos φ between CreateOrder and a
resource-identical EVENT-typed clone (same Req/Prov, different type+mechanism); 'cross' = mean cos φ to MakePayment.

| w_type | within | clone | cross | margin (within − cross) |
|---|---|---|---|---|
| 0.0 | 0.969 | 1.0000 | 0.047 | 0.922 |
| 0.1 | 0.961 | 0.9913 | 0.048 | 0.913 |
| 0.35 | 0.877 | 0.9025 | 0.058 | 0.820 |
| 0.7 | 0.679 | 0.6923 | 0.080 | 0.600 |
| 1.4 | 0.341 | 0.3329 | 0.117 | 0.224 |

All three implementations compose with MakePayment: [True, True, True]. Composite Req symmetric-difference vs API version (resources only): [[], ['Resource.Browser', 'Resource.Database']]

Nearest functional neighbour (cos φ, excluding self) of each canonical capability

| query | top-3 neighbours | alternatives ranked first |
|---|---|---|
| ecommerce/CreateOrder | CreateOrder_DB (0.89), CreateOrder_GUI (0.88), CancelOrder (0.37) | True |
| pipeline/TrainModel | TrainModel_CPU (0.82), EvaluateModel (0.07), CleanData (0.06) | True |

## E4. Irrelevant capabilities / goal relevance

| capability | type | relevance (ours) | cos(Prov, goal) baseline | reference label | note |
|---|---|---|---|---|---|
| SendNotification | EVENT | 0.312 | 0.565 | relevant | canonical C3 |
| CreateOrder | API | 0.299 | 0.281 | relevant | canonical C1 |
| CreateOrder_DB | DATABASE | 0.299 | 0.281 | relevant | bypasses validation: fast but risky |
| CreateOrder_GUI | GUI | 0.299 | 0.281 | relevant | slow human-speed UI automation |
| MakePayment | API | 0.297 | 0.348 | relevant | canonical C2 |
| CancelCart | API | 0.019 | 0.004 | irrelevant/harmful | assignment's C3 for compatibility test (needs OrderExists=false) |
| GenerateSalesReport | FILE | 0.007 | 0.008 | irrelevant/harmful | irrelevant |
| ApplyDiscount | FUNCTION | 0.003 | -0.021 | irrelevant/harmful | irrelevant to goal |
| UpdateProfilePhoto | GUI | 0.001 | -0.013 | irrelevant/harmful | irrelevant |
| CancelOrder | API | 0.000 | 0.209 | irrelevant/harmful | harmful: undoes the goal |
| AuthenticateUser | API | 0.000 | 0.013 | irrelevant/harmful | redundant in this initial state (user already authenticated) |
| ArchiveLogs | COMPUTATION | 0.000 | 0.009 | irrelevant/harmful | irrelevant |

| domain | AUC ours | AUC cos(Prov,goal) | AUC bag-of-atoms cos |
|---|---|---|---|
| ecommerce | 1.000 ± 0.000 | 1.000 | 0.914 |
| travel | 1.000 ± 0.000 | 0.664 | 0.632 |
| pipeline | 1.000 ± 0.000 | 0.733 | 0.725 |

Depth check (travel): relevance of capabilities that contribute only indirectly to the goal

| capability | relevance | reference |
|---|---|---|
| PayTrip | 0.676 | relevant |
| HoldBooking | 0.475 | relevant |
| EmailItinerary | 0.312 | relevant |
| SelectHotel | 0.168 | relevant |
| SelectFlight | 0.164 | relevant |
| SearchHotels | 0.118 | relevant |
| SearchFlights | 0.115 | relevant |
| CheckWeather | 0.007 | irrelevant |
| TranslatePhrases | 0.004 | irrelevant |
| RentCar | 0.000 | irrelevant |
| CancelTrip | 0.000 | irrelevant |

## E5. Operational attributes (ecommerce, alternatives for CreateOrder)

| scenario | ranking by quality = rel × avail(t) × exp(−cost), blocked → 0 |
|---|---|
| baseline (declared costs, always available) | CreateOrder (0.90) > CreateOrder_DB (0.87) > CreateOrder_GUI (0.69) |
| risk-averse weighting (risk weight 2.0) | CreateOrder (0.82) > CreateOrder_GUI (0.71) > CreateOrder_DB (0.55) |
| reliability: DB variant drops 0.95 → 0.60 | CreateOrder (0.90) > CreateOrder_GUI (0.69) > CreateOrder_DB (0.55) |
| availability: API only open 06–22h, t = 03h | CreateOrder_DB (0.87) > CreateOrder_GUI (0.69) > CreateOrder (0.00) |
| availability: same window, t = 12h | CreateOrder (0.90) > CreateOrder_DB (0.87) > CreateOrder_GUI (0.69) |
| resource: Database unavailable | CreateOrder_GUI (0.69) > CreateOrder_DB (0.00) > CreateOrder (0.00) |
| resource: Browser unavailable | CreateOrder (0.90) > CreateOrder_DB (0.87) > CreateOrder_GUI (0.00) |

Composite quality of CompletePurchase = 0.532 vs. its parts: CreateOrder 0.898, MakePayment 0.631, SendNotification 0.936

Ablation: embedding operational attributes INSIDE φ (weight w_ops) instead of keeping them separate

| w_ops | alternative-retrieval precision@k | cos(CreateOrder, CreateOrder_DB) |
|---|---|---|
| 0.0 | 1.00 | 0.893 |
| 0.5 | 1.00 | 0.915 |
| 1.0 | 1.00 | 0.941 |
| 2.0 | 1.00 | 0.961 |
| 4.0 | 0.33 | 0.969 |

## Consistency, noise and efficiency

Dimension sweep (mean over domains and seeds; applicability uses raw inner products vs. clean-up)

| d | compat F1 | composition Req F1 | relevance AUC | applicability acc (clean-up) | applicability acc (raw dot) |
|---|---|---|---|---|---|
| 16 | 0.482 | 0.369 | 0.753 | 0.542 | 0.614 |
| 32 | 0.628 | 0.982 | 0.899 | 0.733 | 0.654 |
| 64 | 0.782 | 1.000 | 0.949 | 1.000 | 0.778 |
| 128 | 0.908 | 1.000 | 0.998 | 1.000 | 0.862 |
| 256 | 0.956 | 1.000 | 0.967 | 1.000 | 0.930 |
| 512 | 0.991 | 1.000 | 1.000 | 1.000 | 0.959 |
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Efficiency (CPU, single thread NumPy)

encode(capability) ≈ 195 µs; compose(3 caps) ≈ 0.43 ms; pairwise compat ≈ 5.1 µs; vector size = 8832 floats/capability

| #capabilities | all-pairs compat matrix | storage |
|---|---|---|
| 100 | 4.5 ms | 3.5 MB (float32) / 7.1 MB (float64) |
| 1000 | 146.0 ms | 35.3 MB (float32) / 70.7 MB (float64) |
| 5000 | 2749.8 ms | 176.6 MB (float32) / 353.3 MB (float64) |
