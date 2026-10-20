from sparks.training.cyclers import LongCycler


def test_long_cycler_round_robin_until_longest_is_exhausted():
    long_loader = ['a0', 'a1', 'a2']
    short_loader = ['b0']

    cycler = LongCycler([long_loader, short_loader])

    assert len(cycler) == 6
    # Each batch comes with the index of the loader it was drawn from.
    assert list(cycler) == [(0, 'a0'), (1, 'b0'), (0, 'a1'), (1, 'b0'), (0, 'a2'), (1, 'b0')]
