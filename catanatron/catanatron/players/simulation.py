import random


def simulate(game, action, **kwargs):
    """Executes action on a copy of game, leaving the RNG stream untouched.

    Some actions (discarding, stealing, rolling) draw from the global RNG when
    their outcome is not given. Restoring the stream keeps every sibling
    simulation on equal footing, makes look-ahead independent of the order in
    which candidate actions are visited, and stops bots from altering the dice
    of the real game by thinking.
    """
    rng_state = random.getstate()
    try:
        copy = game.copy()
        copy.execute(action, validate_action=False, **kwargs)
        return copy
    finally:
        random.setstate(rng_state)
