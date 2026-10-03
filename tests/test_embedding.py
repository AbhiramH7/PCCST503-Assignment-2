import dataclasses
from pathlib import Path

import numpy as np
import pytest

from capemb import Problem, CapabilityEmbedder, IncompatibleComposition, reference as ref
from capemb.model import parse_cond, parse_assign

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def eco():
    p = Problem.load(DATA / "ecommerce.json")
    E = CapabilityEmbedder(p, seed=0)
    return p, E, {c.name: c for c in E.encode_library()}


def test_parsing():
    c = parse_cond("User.role in {CUSTOMER, ADMIN}")
    assert c.op == "in" and c.holds("ADMIN") and not c.holds("GUEST")
    assert parse_cond("Cart.item_count > 0").holds(3)
    assert parse_assign("Order.exists = true") == ("Order.exists", True)


def test_compatibility_pattern(eco):          # Required experiment 1
    _, E, L = eco
    assert E.compat(L["CreateOrder"], L["MakePayment"]).composable
    c3 = E.compat(L["CreateOrder"], L["CancelCart"])
    assert not c3.composable and c3.mismatched > 0.5       # conflict: OrderExists=true vs false
    assert not E.compat(L["CreateOrder"], L["SendNotification"]).composable


def test_matches_symbolic_reference_all_pairs():
    for dom in ("ecommerce", "travel", "pipeline"):
        p = Problem.load(DATA / f"{dom}.json"); E = CapabilityEmbedder(p, seed=1); caps = E.encode_library()
        pred = E.composable_matrix(caps)
        for i, a in enumerate(p.caps):
            for j, b in enumerate(p.caps):
                if i != j:
                    assert pred[i, j] == ref.sym_compat(p.prov_atoms(a), p.req_atoms(b))[2], (dom, a.name, b.name)


def test_composition_matches_reference_and_goal(eco):   # Required experiment 2
    p, E, L = eco
    parts = [L["CreateOrder"], L["MakePayment"], L["SendNotification"]]
    cp = E.compose(parts)
    sym = None
    for n in ("CreateOrder", "MakePayment", "SendNotification"):
        rp = (p.req_atoms(p.cap(n)), p.prov_atoms(p.cap(n)))
        sym = rp if sym is None else ref.sym_compose(sym, rp)
    assert E.decode_sets(cp) == sym
    assert [n for n, _ in E.decompose(cp, list(L))] == ["CreateOrder", "MakePayment", "SendNotification"]
    s0, g = E.encode_state(), E.encode_goal()
    assert E.satisfies(s0, g) == 0.0 and E.satisfies(E.apply(s0, cp), g) == 1.0


def test_associativity_and_rejection(eco):
    _, E, L = eco
    a, b, c = L["CreateOrder"], L["MakePayment"], L["SendNotification"]
    l = E.compose([E.compose([a, b]), c]); r = E.compose([a, E.compose([b, c])])
    assert E.decode_sets(l) == E.decode_sets(r)
    with pytest.raises(IncompatibleComposition):
        E.compose([a, L["CancelCart"]])
    with pytest.raises(IncompatibleComposition):
        E.compose([a, L["CreateOrder_DB"]])               # similar but no dependency link


def test_alternatives_close_but_not_identical(eco):     # Required experiment 3
    _, E, L = eco
    for alt in ("CreateOrder_DB", "CreateOrder_GUI"):
        s = E.similarity(L["CreateOrder"], L[alt])
        assert 0.6 < s < 0.99
        assert E.similarity(L["CreateOrder"], L[alt], "prov") > 0.99      # same function
        assert s > E.similarity(L["CreateOrder"], L["MakePayment"]) + 0.5


def test_irrelevant_capabilities(eco):                  # Required experiment 4
    p, E, L = eco
    caps = E.encode_library()
    rel = dict(zip([c.name for c in caps], E.relevance(caps, E.encode_state(), E.encode_goal())))
    truth = ref.sym_relevant(p)
    assert min(rel[n] for n in truth) > 0.1 > max(v for n, v in rel.items() if n not in truth)


def test_operational_attributes(eco):                   # Required experiment 5
    p, E, L = eco
    api = L["CreateOrder"]
    closed = dataclasses.replace(api, ops=dataclasses.replace(api.ops, window=(6, 22)))
    assert E.quality(closed, t=3) == 0.0 and E.quality(closed, t=12) > 0.5
    lowrel = dataclasses.replace(api, ops=dataclasses.replace(api.ops, reliability=0.5))
    assert E.quality(lowrel) < E.quality(api)
    st = dict(p.initial_state); st.pop("Resource.Browser")
    assert not E.state_fit(E.encode_state(st), L["CreateOrder_GUI"]).applicable
    cp = E.compose([L["CreateOrder"], L["MakePayment"]])
    assert cp.ops.reliability == pytest.approx(api.ops.reliability * L["MakePayment"].ops.reliability)
    assert cp.ops.time_ms == api.ops.time_ms + L["MakePayment"].ops.time_ms


def test_state_awareness(eco):
    _, E, L = eco
    s0 = E.encode_state(); s1 = E.apply(s0, L["CreateOrder"])
    assert not E.state_fit(s0, L["MakePayment"]).applicable and E.state_fit(s1, L["MakePayment"]).applicable
    assert E.state_fit(s0, L["CancelCart"]).applicable and not E.state_fit(s1, L["CancelCart"]).applicable


def test_determinism():
    p = Problem.load(DATA / "ecommerce.json")
    a = CapabilityEmbedder(p, seed=3).encode_capability(p.cap("MakePayment"))
    b = CapabilityEmbedder(p, seed=3).encode_capability(p.cap("MakePayment"))
    assert np.array_equal(a.req, b.req) and np.array_equal(a.prov, b.prov)
