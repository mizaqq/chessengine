"""Mixed finishing slot: each finishing board's start comes from one of several samplers
(e.g. generated technique starts and rewound human checkmates) drawn by weight. Reports
are routed to the sampler that produced the start: labelled (technique) results carry a
`config`, human records do not. Owner 2026-09-19: bring the human finishing boards back
next to the technique boards for the overnight run, since LONG2 had no human finishing
board at all and the conversion dial had nothing to train on.
"""
import random
from typing import Dict, List, Optional, Sequence


class MixedFinishSampler:
    def __init__(self, samplers: Sequence, weights: Sequence[float], seed: int = 0):
        if len(samplers) != len(weights) or not samplers or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("finish mix needs one non-negative weight per sampler with a positive sum")
        self.samplers = list(samplers)
        self.weights = [float(w) for w in weights]
        self._rng = random.Random(seed)
        self._labelled = [s for s in self.samplers if hasattr(s, "report_label")]
        self._human = [s for s in self.samplers if not hasattr(s, "report_label")]

    @property
    def current_depth(self) -> int:
        return max((getattr(s, "current_depth", 0) for s in self._human), default=0)

    def sample(self):
        s = self._rng.choices(self.samplers, weights=self.weights, k=1)[0]
        return s.sample()

    def report(self, depth: int, success: bool) -> None:
        for s in self._human:
            s.report(depth, success)

    def report_label(self, label: str, success: bool, clean: bool = True, depth: Optional[int] = None) -> None:
        for s in self._labelled:
            s.report_label(label, success, clean=clean, depth=depth)

    def dials(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.samplers:
            if hasattr(s, "dials"):
                out.update(s.dials())
        return out

    def state(self) -> Optional[dict]:
        states = [s.state() for s in self.samplers if hasattr(s, "state")]
        return {"mix": [st for st in states if st]} if states else None

    def load_state(self, state: dict) -> None:
        for s, st in zip([x for x in self.samplers if hasattr(x, "state")], (state or {}).get("mix", [])):
            if hasattr(s, "load_state") and st:
                s.load_state(st)
