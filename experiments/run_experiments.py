"""Runs the five required experiments + consistency / efficiency / dimension-sweep evaluations.

    python experiments/run_experiments.py            # full run (~1-2 min)
    python experiments/run_experiments.py --quick    # fewer seeds

Outputs: results/results.json, results/experiment_log.md, results/compat_heatmap.png, results/dim_sweep.png
"""
from __future__ import annotations
import argparse, dataclasses, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capemb import Problem, CapabilityEmbedder, IncompatibleComposition, cosine, reference as ref  # noqa: E402

DOMAINS = ["ecommerce", "travel", "pipeline"]
LOG: list[str] = []


def out(s=""):
    print(s)
    LOG.append(s)


def table(header, rows):
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out("| " + " | ".join(str(x) for x in r) + " |")
    out()


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def load(name):
    return Problem.load(ROOT / "data" / f"{name}.json")


def with_ops(ec, **kw):
    return dataclasses.replace(ec, ops=dataclasses.replace(ec.ops, **kw))


# ----------------------------------------------------------------------------- evaluations
def eval_compat(E, p, caps):
    """Embedding compat matrix vs symbolic reference; symmetric bag-of-atoms cosine baseline."""
    n = len(caps)
    truth = np.zeros((n, n), bool)
    for i, a in enumerate(p.caps):
        for j, b in enumerate(p.caps):
            if i != j:
                truth[i, j] = ref.sym_compat(p.prov_atoms(a), p.req_atoms(b))[2]
    pred = E.composable_matrix(caps)
    off = ~np.eye(n, dtype=bool)
    tp = int((pred & truth & off).sum()); fp = int((pred & ~truth & off).sum()); fn = int((~pred & truth & off).sum())
    ours = f1(tp, fp, fn)[0]
    # baseline: one undirected bag-of-atoms vector per capability, composable iff cosine > tau (oracle tau)
    V = np.stack([E.cb.vec(p.req_atoms(c) | p.prov_atoms(c))[: E.d] for c in p.caps])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    best = 0.0
    for tau in np.linspace(0.0, 1.0, 101):
        b = (S > tau) & off
        best = max(best, f1(int((b & truth).sum()), int((b & ~truth).sum()), int((~b & truth).sum()))[0])
    return ours, best, int(truth[off].sum()), int(off.sum())


def eval_compose(E, p, caps, max_len=4):
    by = {c.name: c for c in caps}
    chains = ref.enumerate_chains(p, max_len=max_len, limit=150)
    tpR = fpR = fnR = tpP = fpP = fnP = 0
    exact = assoc = order = n3 = 0
    naive_tp = naive_fp = naive_fn = 0
    for names, (Rt, Pt) in chains:
        try:
            comp = E.compose([by[n] for n in names])
        except IncompatibleComposition:      # noise-induced false rejection counts as a failure
            fnR += len(Rt); fnP += len(Pt)
            continue
        R, P = E.decode_sets(comp)
        tpR += len(R & Rt); fpR += len(R - Rt); fnR += len(Rt - R)
        tpP += len(P & Pt); fpP += len(P - Pt); fnP += len(Pt - P)
        exact += (R == Rt and P == Pt)
        dec = [n for n, _ in E.decompose(comp, list(by))]
        order += (tuple(dec) == names)
        # naive additive composition = union of all component Req (no cancellation)
        Rn = set().union(*(p.req_atoms(p.cap(n)) for n in names))
        naive_tp += len(Rn & Rt); naive_fp += len(Rn - Rt); naive_fn += len(Rt - Rn)
        if len(names) == 3:
            n3 += 1
            try:
                l = E.compose([E.compose([by[names[0]], by[names[1]]], require_link=False), by[names[2]]], require_link=False)
                r = E.compose([by[names[0]], E.compose([by[names[1]], by[names[2]]], require_link=False)], require_link=False)
                assoc += (E.decode_sets(l) == E.decode_sets(r))
            except IncompatibleComposition:
                pass
    # invalid pairs must be rejected
    rej = tot = 0
    for a in p.caps:
        for b in p.caps:
            if a.name == b.name:
                continue
            if ref.sym_compose((p.req_atoms(a), p.prov_atoms(a)), (p.req_atoms(b), p.prov_atoms(b))) is None:
                tot += 1
                try:
                    E.compose([by[a.name], by[b.name]])
                except IncompatibleComposition:
                    rej += 1
    return dict(n_chains=len(chains), req_f1=f1(tpR, fpR, fnR)[0], prov_f1=f1(tpP, fpP, fnP)[0],
                exact=exact / max(len(chains), 1), assoc=assoc / max(n3, 1), order_recovery=order / max(len(chains), 1),
                reject=rej / max(tot, 1), naive_req_f1=f1(naive_tp, naive_fp, naive_fn)[0],
                naive_req_precision=f1(naive_tp, naive_fp, naive_fn)[1])


def eval_relevance(E, p, caps):
    names = [c.name for c in p.caps]
    truth = ref.sym_relevant(p)
    labels = [n in truth for n in names]
    s, g = E.encode_state(), E.encode_goal()
    ours = E.relevance(caps, s, g)
    base_cos = [E.similarity(c, g) for c in caps]               # goal cosine on the Prov channel
    gb = E.cb.vec(p.goal_atoms())[: E.d]
    base_bag = [cosine(E.cb.vec(p.req_atoms(c) | p.prov_atoms(c))[: E.d], gb) for c in p.caps]
    return auc(ours, labels), auc(base_cos, labels), auc(base_bag, labels)


def eval_applicability(E, p, caps):
    s = E.encode_state()
    S = p.state_atoms()
    ok_exact = ok_soft = 0
    for c, cap in zip(caps, p.caps):
        truth = p.req_atoms(cap) <= S
        ok_exact += E.state_fit(s, c, exact=True).applicable == truth
        ok_soft += E.state_fit(s, c, exact=False).applicable == truth
    return ok_exact / len(caps), ok_soft / len(caps)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = list(range(3 if args.quick else 10))
    R: dict = {}
    P = {n: load(n) for n in DOMAINS}

    out("# Experiment log\n")
    out(f"Default configuration: d=2048 (so Req/Prov bundles are R^4096), dt=128, ds=512, weights (wR,wP,wT)=(0.5,1.0,0.35); seeds={len(seeds)}.\n")

    # ===================================================================== E1
    out("## E1. Capability compatibility (ecommerce, seed 0)\n")
    p = P["ecommerce"]; E = CapabilityEmbedder(p, seed=0); caps = E.encode_library(); L = {c.name: c for c in caps}
    rows = []
    for b in ["MakePayment", "CancelCart", "SendNotification"]:
        k = E.compat(L["CreateOrder"], L[b]); ex = E.explain(L["CreateOrder"], L[b])
        rows.append([f"CreateOrder → {b}", f"{k.matched:.2f}", f"{k.mismatched:.2f}", f"{k.dependency:.2f}",
                     "**yes**" if k.composable else "no",
                     f"{len(ex['state_links'])} state + {len(ex['data_links'])} data links; conflicts={[c[0]+'='+c[1] for c in ex['conflicts']]}"])
    table(["pair", "matched", "mismatched", "dependency", "composable", "evidence (decoded, for inspection)"], rows)
    R["E1_compat"] = {r[0]: r[1:5] for r in rows}

    out("State awareness: applicability (unmet requirements) before/after executing CreateOrder\n")
    s0 = E.encode_state(); s1 = E.apply(s0, L["CreateOrder"])
    rows = []
    for n in ["CreateOrder", "MakePayment", "CancelCart", "SendNotification"]:
        a, b = E.state_fit(s0, L[n]), E.state_fit(s1, L[n])
        rows.append([n, f"{a.unmet:.0f} unmet / {a.violated:.0f} violated → {'applicable' if a.applicable else 'blocked'}",
                     f"{b.unmet:.0f} unmet / {b.violated:.0f} violated → {'applicable' if b.applicable else 'blocked'}"])
    table(["capability", "in S_I", "in S' = Apply(S_I, CreateOrder)"], rows)
    R["E1_state_awareness"] = rows

    out("Similarity ≠ composability (same pairs, different measures)\n")
    pairs = [("CreateOrder", "CreateOrder_DB"), ("CreateOrder", "CreateOrder_GUI"), ("CreateOrder", "CancelOrder"),
             ("CreateOrder", "MakePayment"), ("MakePayment", "SendNotification"), ("CreateOrder", "ArchiveLogs")]
    rows = []
    gb = lambda c: E.cb.vec(p.req_atoms(p.cap(c.name)) | p.prov_atoms(p.cap(c.name)))[: E.d]
    for a, b in pairs:
        k = E.compat(L[a], L[b])
        rows.append([f"{a} → {b}", f"{E.similarity(L[a], L[b]):.3f}", f"{E.similarity(L[a], L[b], 'prov'):.3f}",
                     f"{cosine(gb(L[a]), gb(L[b])):.3f}", f"{k.matched:.2f}/{k.mismatched:.2f}", "yes" if k.composable else "no"])
    table(["pair", "cos φ", "cos Prov", "baseline bag-of-atoms cos", "matched/mismatched", "composable"], rows)
    R["E1_sim_vs_compat"] = rows

    out("Compatibility over all ordered pairs vs. symbolic reference (mean over seeds)\n")
    rows = []; R["compat_f1"] = {}
    for dom in DOMAINS:
        vals = []
        for sd in seeds:
            E_ = CapabilityEmbedder(P[dom], seed=sd)
            vals.append(eval_compat(E_, P[dom], E_.encode_library()))
        o, b = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        rows.append([dom, f"{vals[0][3]} pairs ({vals[0][2]} composable)", f"{o.mean():.3f} ± {o.std():.3f}", f"{b.mean():.3f} ± {b.std():.3f}"])
        R["compat_f1"][dom] = dict(ours=float(o.mean()), baseline=float(b.mean()))
    table(["domain", "ordered pairs", "F1 (ours: Prov·Req)", "F1 (symmetric cosine baseline, oracle threshold)"], rows)

    # heatmap
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        M, MM, _ = E.compat_matrix(caps)
        Z = np.where(MM > 0.5, -1.0, np.where(M > 0.5, 1.0, 0.0)); np.fill_diagonal(Z, 0)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(Z, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(caps))); ax.set_yticks(range(len(caps)))
        ax.set_xticklabels([c.name for c in caps], rotation=70, ha="right", fontsize=7)
        ax.set_yticklabels([c.name for c in caps], fontsize=7)
        ax.set_xlabel("successor Cj (Req)"); ax.set_ylabel("predecessor Ci (Prov)")
        ax.set_title("Embedding compatibility Ci→Cj\n(green = composable, red = conflict, yellow = no link)", fontsize=10)
        fig.tight_layout(); fig.savefig(ROOT / "results" / "compat_heatmap.png", dpi=140); plt.close(fig)
    except Exception as e:  # pragma: no cover
        out(f"(heatmap skipped: {e})")

    # ===================================================================== E2
    out("## E2. Capability composition (ecommerce, seed 0)\n")
    parts = [L["CreateOrder"], L["MakePayment"], L["SendNotification"]]
    cp = E.compose(parts, name="CompletePurchase")
    Rc, Pc = E.decode_sets(cp)
    Rt = ref.sym_compose(ref.sym_compose((p.req_atoms(p.cap("CreateOrder")), p.prov_atoms(p.cap("CreateOrder"))),
                                         (p.req_atoms(p.cap("MakePayment")), p.prov_atoms(p.cap("MakePayment"))))
                         , (p.req_atoms(p.cap("SendNotification")), p.prov_atoms(p.cap("SendNotification"))))
    out(f"CompletePurchase = SendNotification ∘ MakePayment ∘ CreateOrder; decoded Req matches symbolic: **{Rc == Rt[0]}**, Prov matches: **{Pc == Rt[1]}**\n")
    internal = sorted(a for a in (E.decode_sets(L['MakePayment'])[0] | E.decode_sets(L['SendNotification'])[0]) if a in Pc)
    out(f"- Composite Req ({len(Rc)} atoms) excludes internally satisfied atoms, e.g. {[a for a in internal if a[0] in ('Order.exists','Payment.status','Data.order_id')]}")
    out(f"- Naive additive composition would have kept {len(set().union(*(E.decode_sets(c)[0] for c in parts)))} Req atoms (vs {len(Rc)})\n")
    rows = [[c.name, f"{cosine(E.phi(cp), E.phi(c)):.3f}", f"{cosine(cp.prov, c.prov):.3f}", f"{cosine(cp.req, c.req):.3f}"] for c in parts]
    rows.append(["ArchiveLogs (unrelated control)", f"{cosine(E.phi(cp), E.phi(L['ArchiveLogs'])):.3f}",
                 f"{cosine(cp.prov, L['ArchiveLogs'].prov):.3f}", f"{cosine(cp.req, L['ArchiveLogs'].req):.3f}"])
    table(["compared with", "cos φ(composite, ·)", "cos Prov", "cos Req"], rows)
    R["E2_cos"] = rows
    dec = E.decompose(cp, [c.name for c in caps])
    out("Order-preserving provenance block recovers: " + " → ".join(f"{n} ({c:.2f})" for n, c in dec) + "\n")
    s0 = E.encode_state(); g = E.encode_goal()
    out(f"Goal satisfaction (fraction of G in state): before = {E.satisfies(s0, g):.2f}; after Apply(S_I, CompletePurchase) = {E.satisfies(E.apply(s0, cp), g):.2f}; "
        f"composite applicable in S_I: {E.state_fit(s0, cp).applicable}\n")
    o = cp.ops
    out(f"Composite operational profile: time={o.time_ms:.0f} ms, money={o.money:.2f}, risk={o.risk:.3f} (=1-Π(1-r)), reliability={o.reliability:.3f} (=Π rel)\n")
    out(f"Aggregate composition quality vs symbolic reference over all valid connected chains (len 2-4), {len(seeds)} seeds\n")
    rows = []; R["compose"] = {}
    for dom in DOMAINS:
        ms = []
        for sd in seeds:
            E_ = CapabilityEmbedder(P[dom], seed=sd)
            ms.append(eval_compose(E_, P[dom], E_.encode_library()))
        a = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        R["compose"][dom] = a
        rows.append([dom, int(a["n_chains"]), f"{a['req_f1']:.3f}", f"{a['prov_f1']:.3f}", f"{a['exact']:.3f}", f"{a['assoc']:.3f}",
                     f"{a['order_recovery']:.3f}", f"{a['reject']:.3f}", f"{a['naive_req_f1']:.3f}"])
    table(["domain", "#chains", "Req F1", "Prov F1", "exact match", "associativity", "order recovery", "invalid rejected", "naive-sum Req F1"], rows)

    # ===================================================================== E3
    out("## E3. Alternative implementations (ecommerce: API / DATABASE / GUI)\n")
    alts = ["CreateOrder", "CreateOrder_DB", "CreateOrder_GUI", "MakePayment"]
    rows = []
    for a in alts:
        rows.append([a] + [f"{E.similarity(L[a], L[b]):.3f}" for b in alts])
    table(["cos φ"] + alts, rows)
    rows = [[a, b, f"{E.similarity(L[a], L[b], 'prov'):.3f}", f"{E.similarity(L[a], L[b], 'req'):.3f}", f"{E.similarity(L[a], L[b], 'type'):.3f}"]
            for a, b in [("CreateOrder", "CreateOrder_DB"), ("CreateOrder", "CreateOrder_GUI"), ("CreateOrder_DB", "CreateOrder_GUI")]]
    table(["A", "B", "cos Prov (function)", "cos Req", "cos Type/mechanism"], rows)
    out("Type-weight ablation. 'within' = mean cos φ among the three CreateOrder implementations; 'clone' = cos φ between CreateOrder and a\n"
        "resource-identical EVENT-typed clone (same Req/Prov, different type+mechanism); 'cross' = mean cos φ to MakePayment.\n")
    import dataclasses as _dc
    clone = _dc.replace(p.cap("CreateOrder"), name="CreateOrder_EVENT", type="EVENT",
                        mechanism={"trigger": "CartSubmitted", "handler": "CreateOrder"})
    rows = []; R["E3_wtype"] = {}
    for wt in [0.0, 0.1, 0.35, 0.7, 1.4]:
        Ew = CapabilityEmbedder(p, seed=0, w_type=wt); Lw = {c.name: c for c in Ew.encode_library()}
        trio = ["CreateOrder", "CreateOrder_DB", "CreateOrder_GUI"]
        within = np.mean([Ew.similarity(Lw[a], Lw[b]) for i, a in enumerate(trio) for b in trio[i + 1:]])
        ctrl = np.mean([Ew.similarity(Lw[a], Lw["MakePayment"]) for a in trio])
        cl = Ew.similarity(Lw["CreateOrder"], Ew.encode_capability(clone))
        rows.append([wt, f"{within:.3f}", f"{cl:.4f}", f"{ctrl:.3f}", f"{within - ctrl:.3f}"]); R["E3_wtype"][str(wt)] = [float(within), float(cl), float(ctrl)]
    table(["w_type", "within", "clone", "cross", "margin (within − cross)"], rows)
    k = [E.compat(L[a], L["MakePayment"]).composable for a in alts[:3]]
    comps = [E.compose([L[a], L["MakePayment"], L["SendNotification"]]) for a in alts[:3]]
    diff = [sorted(x[0] for x in (E.decode_sets(comps[0])[0] ^ E.decode_sets(c)[0])) for c in comps[1:]]
    out(f"All three implementations compose with MakePayment: {k}. Composite Req symmetric-difference vs API version (resources only): {diff}\n")
    # retrieval of alternatives (pipeline too)
    out("Nearest functional neighbour (cos φ, excluding self) of each canonical capability\n")
    rows = []
    for dom, nm, want in [("ecommerce", "CreateOrder", {"CreateOrder_DB", "CreateOrder_GUI"}), ("pipeline", "TrainModel", {"TrainModel_CPU"})]:
        Ed = CapabilityEmbedder(P[dom], seed=0); cd = {c.name: c for c in Ed.encode_library()}
        rk = sorted([(Ed.similarity(cd[nm], c), c.name) for n, c in cd.items() if n != nm], reverse=True)[:3]
        rows.append([f"{dom}/{nm}", ", ".join(f"{n} ({s:.2f})" for s, n in rk), all(n in want for _, n in rk[: len(want)])])
    table(["query", "top-3 neighbours", "alternatives ranked first"], rows)

    # ===================================================================== E4
    out("## E4. Irrelevant capabilities / goal relevance\n")
    s, g = E.encode_state(), E.encode_goal(); rel = E.relevance(caps, s, g)
    truth = ref.sym_relevant(p)
    rows = [[c.name, p.cap(c.name).type, f"{r:.3f}", f"{E.similarity(c, g):.3f}", "relevant" if c.name in truth else "irrelevant/harmful", p.meta[c.name]]
            for c, r in sorted(zip(caps, rel), key=lambda t: -t[1])]
    table(["capability", "type", "relevance (ours)", "cos(Prov, goal) baseline", "reference label", "note"], rows)
    R["E4_ecommerce"] = [[r[0], r[2], r[3], r[4]] for r in rows]
    rows = []; R["relevance_auc"] = {}
    for dom in DOMAINS:
        v = [eval_relevance(CapabilityEmbedder(P[dom], seed=sd), P[dom], CapabilityEmbedder(P[dom], seed=sd).encode_library()) for sd in seeds]
        v = np.array(v)
        R["relevance_auc"][dom] = dict(ours=float(v[:, 0].mean()), cos_goal=float(v[:, 1].mean()), bag_cos=float(v[:, 2].mean()))
        rows.append([dom, f"{v[:,0].mean():.3f} ± {v[:,0].std():.3f}", f"{v[:,1].mean():.3f}", f"{v[:,2].mean():.3f}"])
    table(["domain", "AUC ours", "AUC cos(Prov,goal)", "AUC bag-of-atoms cos"], rows)
    # relevant-but-not-direct
    out("Depth check (travel): relevance of capabilities that contribute only indirectly to the goal\n")
    pt = P["travel"]; Et = CapabilityEmbedder(pt, seed=0); ct = Et.encode_library()
    rt = Et.relevance(ct, Et.encode_state(), Et.encode_goal())
    table(["capability", "relevance", "reference"], [[c.name, f"{r:.3f}", "relevant" if c.name in ref.sym_relevant(pt) else "irrelevant"]
                                                     for c, r in sorted(zip(ct, rt), key=lambda t: -t[1])])

    # ===================================================================== E5
    out("## E5. Operational attributes (ecommerce, alternatives for CreateOrder)\n")
    trio = ["CreateOrder", "CreateOrder_DB", "CreateOrder_GUI"]
    s0 = E.encode_state()

    def rank(Eo, modified=None, state=None, t=None):
        res = []
        for n in trio:
            c = (modified or {}).get(n, L[n])
            fit = Eo.state_fit(state if state is not None else s0, c)
            res.append((Eo.quality(c, t) if fit.applicable else 0.0, n))
        res.sort(reverse=True)
        return " > ".join(f"{n} ({q:.2f})" for q, n in res)

    st_noDB = dict(p.initial_state); st_noDB.pop("Resource.Database")
    st_noBr = dict(p.initial_state); st_noBr.pop("Resource.Browser")
    risk_E = CapabilityEmbedder(p, seed=0, cost_weights=(0.1, 0.1, 0.1, 2.0, 0.1))
    scen = [
        ("baseline (declared costs, always available)", rank(E)),
        ("risk-averse weighting (risk weight 2.0)", rank(risk_E)),
        ("reliability: DB variant drops 0.95 → 0.60", rank(E, {"CreateOrder_DB": with_ops(L["CreateOrder_DB"], reliability=0.60)})),
        ("availability: API only open 06–22h, t = 03h", rank(E, {"CreateOrder": with_ops(L["CreateOrder"], window=(6, 22))}, t=3)),
        ("availability: same window, t = 12h", rank(E, {"CreateOrder": with_ops(L["CreateOrder"], window=(6, 22))}, t=12)),
        ("resource: Database unavailable", rank(E, state=E.encode_state(st_noDB))),
        ("resource: Browser unavailable", rank(E, state=E.encode_state(st_noBr))),
    ]
    table(["scenario", "ranking by quality = rel × avail(t) × exp(−cost), blocked → 0"], scen)
    R["E5_scenarios"] = scen

    # composition of operational attributes
    cp3 = E.compose([L["CreateOrder"], L["MakePayment"], L["SendNotification"]])
    out(f"Composite quality of CompletePurchase = {E.quality(cp3):.3f} vs. its parts: " +
        ", ".join(f"{c.name} {E.quality(c):.3f}" for c in parts) + "\n")

    out("Ablation: embedding operational attributes INSIDE φ (weight w_ops) instead of keeping them separate\n")
    rows = []; R["E5_ablation"] = {}
    for w in [0.0, 0.5, 1.0, 2.0, 4.0]:
        hits = tot = 0; sims = []
        for sd in range(5):
            for dom, nm, want in [("ecommerce", "CreateOrder", {"CreateOrder_DB", "CreateOrder_GUI"}), ("pipeline", "TrainModel", {"TrainModel_CPU"})]:
                Ed = CapabilityEmbedder(P[dom], seed=sd); cd = {c.name: c for c in Ed.encode_library()}
                phi = {n: Ed.phi(c, w_ops=w) for n, c in cd.items()}
                rk = sorted(((cosine(phi[nm], phi[n]), n) for n in cd if n != nm), reverse=True)[: len(want)]
                hits += sum(n in want for _, n in rk); tot += len(want)
                if dom == "ecommerce": sims.append(cosine(phi["CreateOrder"], phi["CreateOrder_DB"]))
        rows.append([w, f"{hits/tot:.2f}", f"{np.mean(sims):.3f}"]); R["E5_ablation"][str(w)] = hits / tot
    table(["w_ops", "alternative-retrieval precision@k", "cos(CreateOrder, CreateOrder_DB)"], rows)

    # ===================================================================== consistency / dimension sweep
    out("## Consistency, noise and efficiency\n")
    out("Dimension sweep (mean over domains and seeds; applicability uses raw inner products vs. clean-up)\n")
    rows = []; sweep = {}
    for d in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        cf, cr, ra, ap_e, ap_s = [], [], [], [], []
        for dom in DOMAINS:
            for sd in seeds[:5]:
                E_ = CapabilityEmbedder(P[dom], d=d, seed=sd); cs = E_.encode_library()
                cf.append(eval_compat(E_, P[dom], cs)[0])
                cr.append(eval_compose(E_, P[dom], cs, max_len=3)["req_f1"])
                ra.append(eval_relevance(E_, P[dom], cs)[0])
                e_, s_ = eval_applicability(E_, P[dom], cs); ap_e.append(e_); ap_s.append(s_)
        sweep[d] = [float(np.mean(x)) for x in (cf, cr, ra, ap_e, ap_s)]
        rows.append([d] + [f"{v:.3f}" for v in sweep[d]])
    table(["d", "compat F1", "composition Req F1", "relevance AUC", "applicability acc (clean-up)", "applicability acc (raw dot)"], rows)
    R["dim_sweep"] = sweep
    try:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ds_ = list(sweep); names = ["compat F1", "composition Req F1", "relevance AUC", "applicability (clean-up)", "applicability (raw dot)"]
        for i, nme in enumerate(names):
            ax.plot(ds_, [sweep[d][i] for d in ds_], marker="o", label=nme)
        ax.set_xscale("log", base=2); ax.set_xlabel("codebook dimension d"); ax.set_ylabel("score"); ax.set_ylim(0, 1.03)
        ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title("Accuracy vs embedding dimension")
        fig.tight_layout(); fig.savefig(ROOT / "results" / "dim_sweep.png", dpi=140); plt.close(fig)
    except Exception as e:  # pragma: no cover
        out(f"(sweep plot skipped: {e})")

    out("Efficiency (CPU, single thread NumPy)\n")
    E_ = CapabilityEmbedder(p, seed=0)
    t0 = time.perf_counter(); [E_.encode_capability(c) for c in p.caps for _ in range(20)]; t_enc = (time.perf_counter() - t0) / (20 * len(p.caps))
    cs = E_.encode_library()
    t0 = time.perf_counter(); [E_.compose([cs[0], cs[3], cs[4]]) for _ in range(50)]; t_cmp = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter(); [E_.compat(cs[0], cs[3]) for _ in range(2000)]; t_cmat = (time.perf_counter() - t0) / 2000
    rows = []
    d_, dt_, ds_ = E_.d, E_.dt, E_.ds
    per_cap = (4 * d_ + dt_ + ds_)
    for n in [100, 1000, 5000]:
        reps = int(np.ceil(n / len(cs)))
        big = (cs * reps)[:n]
        t0 = time.perf_counter(); E_.compat_matrix(big); tm = time.perf_counter() - t0
        rows.append([n, f"{tm*1000:.1f} ms", f"{n*per_cap*4/1e6:.1f} MB (float32) / {n*per_cap*8/1e6:.1f} MB (float64)"])
    out(f"encode(capability) ≈ {t_enc*1e6:.0f} µs; compose(3 caps) ≈ {t_cmp*1e3:.2f} ms; pairwise compat ≈ {t_cmat*1e6:.1f} µs; vector size = {per_cap} floats/capability\n")
    table(["#capabilities", "all-pairs compat matrix", "storage"], rows)
    R["efficiency"] = dict(encode_us=t_enc * 1e6, compose_ms=t_cmp * 1e3, compat_us=t_cmat * 1e6, floats_per_cap=per_cap)

    (ROOT / "results" / "results.json").write_text(json.dumps(R, indent=1, default=str))
    (ROOT / "results" / "experiment_log.md").write_text("\n".join(LOG))
    print("\nWrote results/results.json, results/experiment_log.md")


if __name__ == "__main__":
    main()
