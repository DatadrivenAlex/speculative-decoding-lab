from collections import Counter
from random import Random

import pytest

from lab import Sampler, normalize


@pytest.mark.parametrize('draft,target', [([1, 2, 7], [6, 3, 1]), ([0, 2, 2], [3, 1, 1])])
def test_recovers_target_distribution(draft, target):
    sampler = Sampler(draft, target)
    rng = Random(91)
    n = 20_000
    samples = [sampler.sample(rng) for _ in range(n)]
    counts = Counter(s.token for s in samples)
    assert [counts[i] / n for i in range(3)] == pytest.approx(normalize(target), abs=.02)
    assert sum(s.accepted for s in samples) / n == pytest.approx(sampler.expected_acceptance, abs=.02)


def test_equal_distributions_always_accept():
    sampler = Sampler([2, 0, 4], [1, 0, 2])
    rng = Random(11)
    assert all(sampler.sample(rng).accepted for _ in range(100))


def test_disjoint_support():
    sampler = Sampler([1, 0], [0, 9])
    result = sampler.sample(Random(0))
    assert (result.token, result.accepted, sampler.expected_acceptance) == (1, False, 0.0)


@pytest.mark.parametrize('weights', [[], [0, 0], [-1, 2], [1, float('inf')], [float('nan')]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        normalize(weights)


def test_large_finite_weights():
    assert normalize([1e308, 1e308]) == (0.5, 0.5)
