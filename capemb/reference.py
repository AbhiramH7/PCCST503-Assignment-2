"""Pure set-based (symbolic) reference semantics.  Used ONLY as ground truth for evaluation."""
from __future__ import annotations


def keys(atoms): return {k for k, _ in atoms}


def sym_compat(prov: set, req: set):
    """(matched, mismatched, composable) between provided atoms and required atoms."""
    pk = keys(prov)
    matched = len(prov & req)
    mismatched = sum(1 for a in req if a[0] in pk and a not in prov)
    return matched, mismatched, matched > 0 and mismatched == 0


def sym_compose(a: tuple, b: tuple, require_link: bool = True):
    """a, b = (req, prov).  Returns (req, prov) of b∘a, or None if the composition is invalid."""
    Ra, Pa = a
    Rb, Pb = b
    matched, mismatched, _ = sym_compat(Pa, Rb)
    if mismatched > 0 or (require_link and matched == 0):
        return None
    residual = Rb - Pa
    if any(x[0] in keys(Ra) and x not in Ra for x in residual):
        return None
    return Ra | residual, Pb | {x for x in Pa if x[0] not in keys(Pb)}


def sym_apply(state: set, prov: set) -> set:
    return {x for x in state if x[0] not in keys(prov)} | prov


def sym_relevant(prob, caps=None) -> set:
    """Backward regression over unmet needs (goal-relevance ground truth)."""
    caps = caps or prob.caps
    S, G = prob.state_atoms(), prob.goal_atoms()
    needs = G - S
    rel, changed = set(), True
    while changed:
        changed = False
        for c in caps:
            if c.name in rel:
                continue
            P, R = prob.prov_atoms(c), prob.req_atoms(c)
            if P & needs:
                rel.add(c.name)
                needs |= (R - S)
                changed = True
    return rel


def enumerate_chains(prob, max_len=4, limit=400):
    """All symbolically valid chains (test-set generator for composition; NOT a planner)."""
    out = []
    base = {c.name: (prob.req_atoms(c), prob.prov_atoms(c)) for c in prob.caps}

    def dfs(names, acc):
        if len(names) >= 2:
            out.append((tuple(names), acc))
        if len(names) >= max_len or len(out) >= limit:
            return
        for n, rp in base.items():
            if n in names:
                continue
            nxt = sym_compose(acc, rp)
            if nxt is not None:
                dfs(names + [n], nxt)

    for n, rp in base.items():
        dfs([n], rp)
    return out
