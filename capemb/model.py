"""Formal application model: states, goals, conditions, capabilities (Assignment 2, Sec. 3-4).

Everything the embedding sees is reduced to *atoms* = (key, value-string) pairs:
  * equality condition  x == v      -> atom (x, v)
  * other predicates    x > 0, ...  -> atom (canonical-predicate-string, "true"/"false")
  * data items          input/output-> atom ("Data.<name>", "<type>|<domain>")
  * resources           ("Resource.<R>", "available")
Atoms sharing a *key* are mutually exclusive (a variable has one value), which is what
lets the embedding detect conflicts (x=false vs x=true) and not only matches.
"""
from __future__ import annotations
import json, operator, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Atom = tuple  # (key:str, value:str)

_COND = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(==|!=|>=|<=|>|<|\bin\b)\s*(.+?)\s*$")
_ASSIGN = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*(.+?)\s*$")
_OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}


def parse_value(tok: str):
    t = tok.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t.strip("'\"")


def vstr(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(vstr(x) for x in v)) + "}"
    return str(v)


@dataclass(frozen=True)
class Cond:
    """A condition  var <op> val  used for preconditions, constraints and goals."""
    var: str
    op: str
    val: Any

    @property
    def key(self) -> str:
        return self.var if self.op == "==" else f"{self.var}{self.op}{vstr(self.val)}"

    def atom(self) -> Atom:
        return (self.var, vstr(self.val)) if self.op == "==" else (self.key, "true")

    def holds(self, v) -> bool:
        try:
            if self.op == "==": return v == self.val
            if self.op == "!=": return v != self.val
            if self.op == "in": return v in self.val
            return bool(_OPS[self.op](v, self.val))
        except TypeError:
            return False


def parse_cond(s: str) -> Cond:
    m = _COND.match(s)
    if not m:
        raise ValueError(f"cannot parse condition: {s!r}")
    var, op, raw = m.groups()
    val = frozenset(parse_value(x) for x in raw.strip("{}[]() ").split(",")) if op == "in" else parse_value(raw)
    return Cond(var, op, val)


def parse_assign(s: str):
    m = _ASSIGN.match(s)
    if not m:
        raise ValueError(f"cannot parse effect: {s!r}")
    return m.group(1), parse_value(m.group(2))


class Vocabulary:
    """Registry of non-equality predicates so that assignments can be grounded consistently."""
    def __init__(self):
        self.preds: dict[str, dict[str, Cond]] = {}

    def register(self, c: Cond):
        if c.op != "==":
            self.preds.setdefault(c.var, {})[c.key] = c

    def ground(self, var: str, val) -> set:
        atoms = {(var, vstr(val))}
        for k, p in self.preds.get(var, {}).items():
            atoms.add((k, vstr(p.holds(val))))
        return atoms

    def ground_state(self, state: dict) -> set:
        out = set()
        for var, val in state.items():
            out |= self.ground(var, val)
        return out


@dataclass
class OpProfile:
    time_ms: float = 0.0
    money: float = 0.0
    resource: float = 0.0
    risk: float = 0.0
    energy: float = 0.0
    reliability: float = 1.0
    window: tuple | None = None   # availability A(t): open if open <= hour < close; None = always

    def available(self, t: float | None) -> bool:
        if self.window is None or t is None:
            return True
        return self.window[0] <= t < self.window[1]


@dataclass
class Capability:
    name: str
    type: str
    mechanism: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)       # (name, type, domain, required)
    outputs: list = field(default_factory=list)      # (name, type, domain)
    preconditions: list = field(default_factory=list)  # Cond
    effects: list = field(default_factory=list)      # (var, value)
    constraints: list = field(default_factory=list)  # Cond
    resources: list = field(default_factory=list)
    ops: OpProfile = field(default_factory=OpProfile)

    @staticmethod
    def from_dict(d: dict, policies: dict) -> "Capability":
        cost = d.get("cost", {})
        win = d.get("availability")
        cons = [parse_cond(s) for s in d.get("constraints", [])] + [policies[p] for p in d.get("policies", [])]
        return Capability(
            name=d["name"], type=d["type"], mechanism=d.get("mechanism", {}),
            inputs=[tuple(i) for i in d.get("inputs", [])], outputs=[tuple(o) for o in d.get("outputs", [])],
            preconditions=[parse_cond(s) for s in d.get("preconditions", [])],
            effects=[parse_assign(s) for s in d.get("effects", [])], constraints=cons,
            resources=d.get("resources", []),
            ops=OpProfile(cost.get("time_ms", 0), cost.get("money", 0), cost.get("resource", 0),
                          cost.get("risk", 0), cost.get("energy", 0), d.get("reliability", 1.0),
                          tuple(win) if win else None))


class Problem:
    """A formal application A = (S, C, S_I, G, R, K)."""
    def __init__(self, data: dict):
        self.name = data["name"]
        self.vocab = Vocabulary()
        self.resources = data.get("resources", [])
        self.policies = {n: parse_cond(s) for n, s in data.get("global_constraints", {}).items()}
        self.initial_state = dict(data["initial_state"])
        for r in self.resources:
            self.initial_state[f"Resource.{r}"] = "available"
        self.goal = [parse_cond(s) for s in data["goal"]]
        self.caps = [Capability.from_dict(c, self.policies) for c in data["capabilities"]]
        self.meta = {c["name"]: c.get("note", "") for c in data["capabilities"]}
        for c in self.goal: self.vocab.register(c)
        for cap in self.caps:
            for c in cap.preconditions + cap.constraints: self.vocab.register(c)

    @staticmethod
    def load(path) -> "Problem":
        return Problem(json.loads(Path(path).read_text()))

    def cap(self, name: str) -> Capability:
        return next(c for c in self.caps if c.name == name)

    # ---- atom semantics (shared by the embedding and the symbolic reference) ----
    def req_atoms(self, cap: Capability) -> set:
        a = {c.atom() for c in cap.preconditions + cap.constraints}
        a |= {("Data." + n, f"{t}|{d}") for n, t, d, req in cap.inputs if req}
        a |= {("Resource." + r, "available") for r in cap.resources}
        return a

    def prov_atoms(self, cap: Capability) -> set:
        a = set()
        for var, val in cap.effects:
            a |= self.vocab.ground(var, val)
        a |= {("Data." + n, f"{t}|{d}") for n, t, d in cap.outputs}
        return a

    def state_atoms(self, state: dict | None = None) -> set:
        return self.vocab.ground_state(self.initial_state if state is None else state)

    def goal_atoms(self) -> set:
        return {c.atom() for c in self.goal}
