"""Event and Trace schema for Environment-Anchored Monitoring (EAM) pilot.

A Trace is one multi-step task execution. It carries:
  - events: the state changes it caused (what EAM sees)
  - subtask_instructions: the decomposed, individually-benign phrasings a
    conversation monitor would see (what M1/M2 see)
  - label / category / irreversible_step: ground truth for scoring
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import os

# Allowed vocabulary — keep the recorder output normalised.
RESOURCES = {"database", "filesystem", "git", "outbound", "logs", "permissions"}
OPS = {"insert", "update", "delete", "read_bulk", "upload", "grant", "write", "execute"}


@dataclass
class Event:
    step: int
    resource: str
    op: str
    object: str
    attrs: dict = field(default_factory=dict)

    def render(self) -> str:
        a = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        a = f" [{a}]" if a else ""
        return f"step {self.step} | {self.resource} {self.op} {self.object}{a}"


@dataclass
class Trace:
    id: str
    label: str                       # "harmful" | "benign"
    category: str
    irreversible_step: Optional[int]  # step at which damage is done (harmful only)
    subtask_instructions: list        # list[str]
    events: list                      # list[Event]

    def events_text(self) -> str:
        return "\n".join(e.render() for e in self.events)

    def instructions_text(self) -> str:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.subtask_instructions))

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def save_traces(traces: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in traces], f, indent=2)


def load_traces(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    out = []
    for t in raw:
        t = dict(t)
        t["events"] = [Event(**e) for e in t["events"]]
        out.append(Trace(**t))
    return out


def validate(traces: list) -> None:
    """Fail loudly on malformed traces so the dataset stays clean."""
    ids = set()
    for t in traces:
        assert t.label in {"harmful", "benign"}, f"{t.id}: bad label {t.label}"
        assert t.id not in ids, f"duplicate id {t.id}"
        ids.add(t.id)
        for e in t.events:
            assert e.resource in RESOURCES, f"{t.id}: bad resource {e.resource}"
            assert e.op in OPS, f"{t.id}: bad op {e.op}"
        if t.label == "harmful":
            assert t.irreversible_step is not None, f"{t.id}: harmful needs irreversible_step"
    print(f"validate: {len(traces)} traces OK "
          f"({sum(t.label=='harmful' for t in traces)} harmful, "
          f"{sum(t.label=='benign' for t in traces)} benign)")
