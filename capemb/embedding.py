"""Dual-channel (Requires / Provides) atom-codebook embedding for capability composition.

Design summary (full maths in docs/report.md, Sec. 4-6)
-------------------------------------------------------
* Every atom a=(key,value) gets a deterministic pseudo-random code  u_a = [l_a | w_key]  in R^{2d}:
    l_a    : "literal" code  (unique per key+value)
    w_key  : "variable" code (shared by all values of the same key)
* A set of atoms is embedded as the SUM of its codes (a bundle).  Inner products then count:
    <A_lit, B_lit>                    ~ |A ∩ B|                      (matches)
    <A_var, B_var> - <A_lit, B_lit>   ~ #keys in both with different values  (conflicts)
* A capability keeps two separate bundles: Req (preconditions, constraints, required inputs,
  resources) and Prov (effects, outputs).  Compatibility is the ASYMMETRIC pairing Prov_i · Req_j.
* Functional similarity is the cosine of phi(C) = [wR*Req^ | wP*Prov^ | wT*Type^] (symmetric).
* Operational properties (cost, reliability, availability, risk) live in a separate profile.
* Composition operates on the Req/Prov channels (cancel satisfied requirements, overwrite
  clobbered effects) via codebook clean-up, plus an order-preserving provenance block.
"""
from __future__ import annotations
import hashlib, math
from dataclasses import dataclass, field
import numpy as np

from .model import Problem, Capability, OpProfile, parse_cond, Cond


class IncompatibleComposition(ValueError):
    pass


def _rng(label: str, seed: int):
    h = hashlib.blake2b(f"{seed}|{label}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def _bip(label: str, n: int, seed: int) -> np.ndarray:
    return _rng(label, seed).choice([-1.0, 1.0], size=n) / math.sqrt(n)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


class Codebook:
    """Deterministic hashed atom codes + least-squares clean-up (decode)."""

    def __init__(self, d: int, seed: int = 0):
        self.d, self.seed = d, seed
        self._lit, self._var = {}, {}
        self.atoms: list = []
        self._idx: dict = {}
        self._pinv = None
        self._L = None

    def register(self, atom):
        if atom not in self._idx:
            self._idx[atom] = len(self.atoms)
            self.atoms.append(atom)
            self._lit[atom] = _bip(f"L|{atom[0]}|{atom[1]}", self.d, self.seed)
            self._pinv = None
            self._L = None
        if atom[0] not in self._var:
            self._var[atom[0]] = _bip(f"K|{atom[0]}", self.d, self.seed)

    def vec(self, atoms) -> np.ndarray:
        """Bundle of atoms (iterable or dict atom->weight) -> R^{2d}."""
        items = atoms.items() if isinstance(atoms, dict) else ((a, 1.0) for a in atoms)
        v = np.zeros(2 * self.d)
        for a, w in items:
            self.register(a)
            v[: self.d] += w * self._lit[a]
            v[self.d:] += w * self._var[a[0]]
        return v

    def decode(self, v: np.ndarray, thr: float = 0.5) -> set:
        """Clean-up memory: recover the (sparse, 0/1) atom set from a bundle.
        d >= V : exact least squares (codes are linearly independent).
        d <  V : orthogonal matching pursuit (sparse recovery; needs d ~ k log V, k = set size)."""
        if not self.atoms:
            return set()
        V = len(self.atoms)
        if V <= self.d:
            if self._pinv is None:
                L = np.stack([self._lit[a] for a in self.atoms])      # V x d
                self._pinv = np.linalg.pinv(L.T)                      # V x d
            w = self._pinv @ v[: self.d]
            return {a for a, x in zip(self.atoms, w) if x > thr}
        if self._L is None or self._L.shape[0] != V:
            self._L = np.stack([self._lit[a] for a in self.atoms])
        L, y = self._L, v[: self.d]
        resid, sel = y.copy(), []
        y_norm = np.linalg.norm(y)
        for _ in range(max(1, self.d // 2)):
            if np.linalg.norm(resid) < 0.05 * y_norm:
                break
            c = np.abs(L @ resid)
            c[sel] = -1
            sel.append(int(c.argmax()))
            coef, *_ = np.linalg.lstsq(L[sel].T, y, rcond=None)
            resid = y - L[sel].T @ coef
        coef, *_ = np.linalg.lstsq(L[sel].T, y, rcond=None) if sel else (np.array([]),)
        return {self.atoms[i] for i, x in zip(sel, coef) if x > thr}


@dataclass
class EncodedCap:
    name: str
    req: np.ndarray            # R^{2d}
    prov: np.ndarray           # R^{2d}
    typ: np.ndarray            # R^{dt} (sum of component type/mechanism codes)
    seq: np.ndarray            # R^{ds} order-preserving provenance block
    length: int = 1
    ops: OpProfile = field(default_factory=OpProfile)
    components: tuple = ()

    @property
    def is_composite(self) -> bool:
        return self.length > 1


@dataclass
class Compat:
    matched: float
    mismatched: float
    req_mass: float
    dependency: float      # matched / |Req_j|
    composable: bool       # matched >= 1 and no conflicts


@dataclass
class StateFit:
    unmet: float
    violated: float
    coverage: float
    applicable: bool


class CapabilityEmbedder:
    def __init__(self, problem: Problem, d: int = 2048, dt: int = 128, ds: int = 512, seed: int = 0,
                 w_req: float = 0.5, w_prov: float = 1.0, w_type: float = 0.35,
                 cost_weights=(0.3, 0.3, 0.1, 0.2, 0.1)):
        self.p, self.d, self.dt, self.ds, self.seed = problem, d, dt, ds, seed
        self.w = (w_req, w_prov, w_type)
        self.cost_w = cost_weights
        self.cb = Codebook(d, seed)
        # pre-register every atom of the problem so clean-up is computed once
        for c in problem.caps:
            for a in problem.req_atoms(c) | problem.prov_atoms(c):
                self.cb.register(a)
        for a in problem.state_atoms() | problem.goal_atoms():
            self.cb.register(a)
        caps = problem.caps
        mx = lambda f: max([f(c.ops) for c in caps] + [1e-9])
        self.scales = (mx(lambda o: o.time_ms), mx(lambda o: o.money), mx(lambda o: o.resource),
                       1.0, mx(lambda o: o.energy))

    # ------------------------------------------------------------------ encoders
    def encode_state(self, state: dict | None = None) -> np.ndarray:
        return self.cb.vec(self.p.state_atoms(state))

    def encode_goal(self, goal=None) -> np.ndarray:
        if goal is None:
            return self.cb.vec(self.p.goal_atoms())
        conds = [parse_cond(g) if isinstance(g, str) else g for g in goal]
        return self.cb.vec({c.atom() for c in conds})

    def _type_vec(self, cap: Capability) -> np.ndarray:
        v = _bip(f"T|{cap.type}", self.dt, self.seed)
        for k, val in sorted(cap.mechanism.items()):
            v = v + 0.5 * _bip(f"M|{k}={val}", self.dt, self.seed)
        return v

    def id_vec(self, name: str) -> np.ndarray:
        return _bip(f"ID|{name}", self.ds, self.seed)

    def encode_capability(self, cap: Capability) -> EncodedCap:
        return EncodedCap(cap.name, self.cb.vec(self.p.req_atoms(cap)), self.cb.vec(self.p.prov_atoms(cap)),
                          self._type_vec(cap), self.id_vec(cap.name), 1, cap.ops)

    def encode_library(self) -> list:
        return [self.encode_capability(c) for c in self.p.caps]

    def encode(self, x):
        """Polymorphic encode: Capability | dict (state) | list[str] (goal)."""
        if isinstance(x, Capability): return self.encode_capability(x)
        if isinstance(x, dict): return self.encode_state(x)
        return self.encode_goal(x)

    # --------------------------------------------------------------- similarity
    def phi(self, c: EncodedCap, w_ops: float = 0.0) -> np.ndarray:
        wR, wP, wT = self.w
        parts = [wR * _unit(c.req), wP * _unit(c.prov), wT * _unit(c.typ)]
        if w_ops:
            parts.append(w_ops * self.ops_vector(c))
        return np.concatenate(parts)

    def similarity(self, x, y, view: str = "phi") -> float:
        """Functional similarity (symmetric).  view in {phi, prov, req, type}; mixed cap/vector -> Prov view."""
        cx, cy = isinstance(x, EncodedCap), isinstance(y, EncodedCap)
        if cx and cy:
            if view == "phi": return cosine(self.phi(x), self.phi(y))
            return cosine(getattr(x, {"prov": "prov", "req": "req", "type": "typ"}[view]),
                          getattr(y, {"prov": "prov", "req": "req", "type": "typ"}[view]))
        if cx: return cosine(x.prov, y)
        if cy: return cosine(x, y.prov)
        return cosine(x, y)

    # -------------------------------------------------------------- compatibility
    def _split(self, v):
        return v[: self.d], v[self.d:]

    def compat(self, a: EncodedCap, b: EncodedCap) -> Compat:
        """Directed compatibility a -> b via Prov_a . Req_b."""
        pl, pv = self._split(a.prov)
        rl, rv = self._split(b.req)
        matched = float(pl @ rl)
        mism = float(pv @ rv) - matched
        mass = float(rl @ rl)
        return Compat(matched, mism, mass, matched / mass if mass > 1e-9 else 0.0,
                      matched > 0.5 and mism < 0.5)

    def compat_matrix(self, caps: list):
        """Vectorised all-pairs compat: returns (matched, mismatched, req_mass) with [i,j] = i -> j."""
        d = self.d
        P = np.stack([c.prov for c in caps]); R = np.stack([c.req for c in caps])
        M = P[:, :d] @ R[:, :d].T
        V = P[:, d:] @ R[:, d:].T
        mass = (R[:, :d] ** 2).sum(1)
        return M, V - M, mass

    def composable_matrix(self, caps: list) -> np.ndarray:
        M, MM, _ = self.compat_matrix(caps)
        C = (M > 0.5) & (MM < 0.5)
        np.fill_diagonal(C, False)
        return C

    def state_fit(self, state_vec: np.ndarray, c: EncodedCap, exact: bool = True) -> StateFit:
        """State awareness: can capability c run in this state?  (Req_c checked against the state bundle).
        exact=True uses codebook clean-up (robust for large states); exact=False uses raw inner products."""
        if exact:
            S, R = self.cb.decode(state_vec), self.cb.decode(c.req)
            sk = {k: v for k, v in S}
            unmet = float(len(R - S))
            viol = float(sum(1 for a in R if a[0] in sk and a not in S))
            n = len(R)
            return StateFit(unmet, viol, (n - unmet) / n if n else 1.0, unmet == 0)
        sl, sv = self._split(state_vec)
        rl, rv = self._split(c.req)
        matched = float(sl @ rl)
        viol = float(sv @ rv) - matched
        mass = float(rl @ rl)
        unmet = mass - matched
        return StateFit(unmet, viol, matched / mass if mass > 1e-9 else 1.0, unmet < 0.5)

    def apply(self, state_vec: np.ndarray, c: EncodedCap) -> np.ndarray:
        """S' = Apply(S, E_c) in vector space (clean-up, overwrite by key, re-bundle)."""
        S, P = self.cb.decode(state_vec), self.cb.decode(c.prov)
        pk = {k for k, _ in P}
        return self.cb.vec({a for a in S if a[0] not in pk} | P)

    def satisfies(self, state_vec: np.ndarray, goal_vec: np.ndarray) -> float:
        """Fraction of goal atoms satisfied by a state (S |= G  <=>  1.0), via clean-up."""
        G, S = self.cb.decode(goal_vec), self.cb.decode(state_vec)
        return len(G & S) / len(G) if G else 1.0

    def explain(self, a: EncodedCap, b: EncodedCap) -> dict:
        """Which atoms link a -> b?  Splits the compatibility into state (precondition/effect) and
        data (input/output) channels, and lists conflicts.  Uses clean-up, for inspection/reporting only."""
        P, R = self.cb.decode(a.prov), self.cb.decode(b.req)
        pk = {k for k, _ in P}
        m = P & R
        return {"state_links": sorted(x for x in m if not x[0].startswith("Data.")),
                "data_links": sorted(x for x in m if x[0].startswith("Data.")),
                "conflicts": sorted(x for x in R if x[0] in pk and x not in P)}

    # ---------------------------------------------------------------- composition
    def _compose2(self, a: EncodedCap, b: EncodedCap, require_link: bool, name: str | None) -> EncodedCap:
        Ra, Pa = self.cb.decode(a.req), self.cb.decode(a.prov)
        Rb, Pb = self.cb.decode(b.req), self.cb.decode(b.prov)
        kPa, kPb, kRa = ({k for k, _ in s} for s in (Pa, Pb, Ra))
        satisfied = Rb & Pa
        clash = [x for x in Rb if x[0] in kPa and x not in Pa]
        residual = Rb - Pa
        clash += [x for x in residual if x[0] in kRa and x not in Ra]
        # conflicts ALWAYS invalidate; a missing dependency link only when require_link (kept out of the
        # algebra so that composition itself stays associative)
        if clash or (require_link and not satisfied):
            why = f"conflict on {sorted(x[0] for x in clash)}" if clash else "no output/effect satisfies a requirement"
            raise IncompatibleComposition(f"{a.name} -> {b.name}: {why}")
        Rc = Ra | residual
        Pc = Pb | {x for x in Pa if x[0] not in kPb}
        ao, bo = a.ops, b.ops
        win = None
        if ao.window or bo.window:
            lo = max(w[0] for w in (ao.window, bo.window) if w) if (ao.window or bo.window) else 0
            hi = min(w[1] for w in (ao.window, bo.window) if w)
            win = (lo, hi if hi > lo else lo)
        ops = OpProfile(ao.time_ms + bo.time_ms, ao.money + bo.money, ao.resource + bo.resource,
                        1 - (1 - ao.risk) * (1 - bo.risk), ao.energy + bo.energy,
                        ao.reliability * bo.reliability, win)
        return EncodedCap(name or f"{a.name}>{b.name}", self.cb.vec(Rc), self.cb.vec(Pc), a.typ + b.typ,
                          a.seq + np.roll(b.seq, a.length), a.length + b.length, ops,
                          a.components + b.components if a.is_composite or b.is_composite else (a.name, b.name))

    def compose(self, caps: list, name: str | None = None, require_link: bool = True) -> EncodedCap:
        """C_n ∘ ... ∘ C_1 for caps = [C_1, ..., C_n] (execution order).
        Always raises IncompatibleComposition on conflicts.  With require_link=True each C_k must also
        consume at least one output/effect of the composite built so far (a *connected* chain)."""
        acc = caps[0]
        for c in caps[1:]:
            acc = self._compose2(acc, c, require_link, None)
        if len(caps) > 1:
            acc = EncodedCap(name or ">".join(c.name for c in caps), acc.req, acc.prov, acc.typ, acc.seq,
                             acc.length, acc.ops, tuple(c.name for c in caps))
        return acc

    def decompose(self, c: EncodedCap, names: list) -> list:
        """Recover the ordered component names from the provenance block (nearest id code per slot)."""
        ids = {n: self.id_vec(n) for n in names}
        out = []
        for k in range(c.length):
            slot = np.roll(c.seq, -k)
            best = max(ids, key=lambda n: cosine(slot, ids[n]))
            out.append((best, cosine(slot, ids[best])))
        return out

    def decode_sets(self, c: EncodedCap):
        return self.cb.decode(c.req), self.cb.decode(c.prov)

    # ------------------------------------------------------------------ goal relevance
    def relevance(self, caps: list, state_vec: np.ndarray, goal_vec: np.ndarray,
                  gamma: float = 0.7, depth: int = 8) -> np.ndarray:
        """Goal relevance of each capability: direct contribution to UNMET goal atoms, propagated
        backwards through (embedding-derived) dependency links with discount gamma.
        Not a planner: no sequence search, only a fixed-point of  rel_i = max(direct_i, g * max_j D_ij rel_j)."""
        d = self.d
        S, G = self.cb.decode(state_vec), self.cb.decode(goal_vec)
        unmet = G - S
        n = len(caps)
        if not unmet:
            return np.zeros(n)
        u = self.cb.vec(unmet)
        gl, gv = self._split(goal_vec)
        direct = np.zeros(n)
        for i, c in enumerate(caps):
            pl, pv = self._split(c.prov)
            m = float(pl @ u[:d])
            harm = float(pv @ gv) - float(pl @ gl)          # clobbers a goal variable with another value
            direct[i] = max(0.0, (m - max(harm, 0.0)) / len(unmet))
        # dependency links on the *residual* (not yet satisfied) requirements
        resid, full = [], []
        for c in caps:
            R = self.cb.decode(c.req)
            resid.append(self.cb.vec(R - S)); full.append(c.req)
        Rres = np.stack(resid); Rfull = np.stack(full)
        P = np.stack([c.prov for c in caps])
        matched = P[:, :d] @ Rres[:, :d].T
        nres = (Rres[:, :d] ** 2).sum(1)
        clob = (P[:, d:] @ Rfull[:, d:].T) - (P[:, :d] @ Rfull[:, :d].T)
        D = np.where((nres[None, :] > 0.5) & (clob < 0.5), matched / np.maximum(nres[None, :], 1e-9), 0.0)
        D[D < 0.25 / 4] = 0.0  # drop noise-level links
        np.fill_diagonal(D, 0.0)
        rel = direct.copy()
        for _ in range(depth):
            new = np.maximum(direct, gamma * (D * rel[None, :]).max(axis=1))
            if np.allclose(new, rel): break
            rel = new
        return rel

    # ----------------------------------------------------------- operational properties
    def cost_scalar(self, c: EncodedCap) -> float:
        o, s, w = c.ops, self.scales, self.cost_w
        comps = (o.time_ms / s[0], o.money / s[1], o.resource / s[2], o.risk, o.energy / s[4])
        return float(sum(wi * ci for wi, ci in zip(w, comps)))

    def quality(self, c: EncodedCap, t: float | None = None) -> float:
        """Operational utility = reliability * availability(t) * exp(-weighted cost)."""
        return c.ops.reliability * (1.0 if c.ops.available(t) else 0.0) * math.exp(-self.cost_scalar(c))

    def ops_vector(self, c: EncodedCap) -> np.ndarray:
        o, s = c.ops, self.scales
        return np.array([o.time_ms / s[0], o.money / s[1], o.resource / s[2], o.risk, o.energy / s[4],
                         o.reliability, 1.0 if o.window is None else 0.5])

    def score_link(self, a: EncodedCap, b: EncodedCap, t: float | None = None) -> float:
        """Ranking score for a -> b: compatibility gated, then weighted by operational quality of b."""
        k = self.compat(a, b)
        return (min(k.dependency, 1.0) if k.composable else 0.0) * self.quality(b, t)
