"""A one-token speculative sampler over finite categorical distributions."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from random import Random


@dataclass(frozen=True)
class Sample:
    token: int
    accepted: bool


def normalize(weights: Sequence[float]) -> tuple[float, ...]:
    if not weights or any(not isfinite(w) or w < 0 for w in weights):
        raise ValueError('weights must be finite, nonnegative and nonempty')
    scale = max(weights)
    if not scale:
        raise ValueError('weights must have positive total mass')
    scaled = [w / scale for w in weights]
    total = fsum(scaled)
    return tuple(w / total for w in scaled)


class Sampler:
    def __init__(self, draft: Sequence[float], target: Sequence[float]):
        if len(draft) != len(target):
            raise ValueError('draft and target must have the same vocabulary')
        self.draft = normalize(draft)
        self.target = normalize(target)
        self.residual = tuple(max(0.0, p - q) for q, p in zip(self.draft, self.target))

    def sample(self, rng: Random) -> Sample:
        ids = range(len(self.draft))
        candidate = rng.choices(ids, weights=self.draft)[0]
        if rng.random() < min(1.0, self.target[candidate] / self.draft[candidate]):
            return Sample(candidate, True)
        return Sample(rng.choices(ids, weights=self.residual)[0], False)

    @property
    def expected_acceptance(self) -> float:
        return fsum(min(q, p) for q, p in zip(self.draft, self.target))
