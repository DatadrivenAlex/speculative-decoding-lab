from collections import Counter
from random import Random

from lab import Sampler


def main():
    trials = 30_000
    for name, draft in [('aligned', [5, 3, 2]), ('misaligned', [1, 2, 7])]:
        sampler = Sampler(draft, [6, 3, 1])
        rng = Random(17)
        samples = [sampler.sample(rng) for _ in range(trials)]
        counts = Counter(s.token for s in samples)
        observed = [round(counts[i] / trials, 4) for i in range(3)]
        accepted = sum(s.accepted for s in samples) / trials
        print(f'{name}: tokens={observed}; accepted={accepted:.3f}; '
              f'expected={sampler.expected_acceptance:.3f}')


if __name__ == '__main__':
    main()
