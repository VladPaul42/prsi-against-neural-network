"""Two players who pick any legal move, for thousands of games.

Nothing should crash, get stuck, or lose a card.
"""

import random

from prsi.actions import NUM_ACTIONS
from prsi.cards import full_deck
from prsi.game import Game, IllegalAction

GAMES = 5000
STEP_LIMIT = 10000


def assert_conserved(game: Game) -> None:
    cards = [card for hand in game.hands for card in hand]
    cards.extend(game.draw)
    cards.extend(game.discard)
    assert len(cards) == 32
    assert len({card.id for card in cards}) == 32
    assert {card.id for card in cards} == {card.id for card in full_deck()}
    assert game.discard
    assert not (game.pending_sevens and game.ace_active)
    if not game.is_over:
        assert game.legal_actions()


def test_random_games_stay_consistent():
    master = random.Random(42)
    for _ in range(GAMES):
        seed = master.randrange(2**31)
        starter = master.randrange(2)
        game = Game(seed, starter)
        picker = random.Random(seed)
        steps = 0
        while not game.is_over:
            assert_conserved(game)
            legal = game.legal_actions()
            bad = next(action for action in range(NUM_ACTIONS) if action not in legal)
            played = len(game.actions)
            try:
                game.step(bad)
            except IllegalAction:
                pass
            else:
                raise AssertionError(f"illegal action accepted, seed={seed}")
            # A rejected move must not be recorded or change whose turn it is.
            assert len(game.actions) == played
            assert not game.is_over
            game.step(picker.choice(legal))
            steps += 1
            if steps > STEP_LIMIT:
                raise AssertionError(f"stuck seed={seed} starter={starter}")
        assert_conserved(game)
        result = game.result
        assert result is not None
        sizes = (len(game.hands[0]), len(game.hands[1]))
        if result["reason"] == "empty_hand":
            assert result["winner"] is not None
            assert sizes[result["winner"]] == 0
        elif result["reason"] == "stuck":
            winner = result["winner"]
            if winner is None:
                assert sizes[0] == sizes[1]
            elif winner == 0:
                assert sizes[0] < sizes[1]
            else:
                assert sizes[1] < sizes[0]
        else:
            raise AssertionError(result)
