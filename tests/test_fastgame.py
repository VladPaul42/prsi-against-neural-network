"""The fast engine must play exactly the same game as ``prsi.Game``."""

import random
from pathlib import Path

import pytest

from bot.fastgame import (
    DONT_PLAY,
    REASON_LIMIT,
    FastGame,
    from_prsi_game,
)
from prsi.cards import Card, Rank, Suit
from prsi.game import Game
from prsi.record import load_record, opening_hand_of

GAMES = 2000
GAMES_DIR = Path(__file__).resolve().parent.parent / "games"


def reference_state(game: Game) -> dict:
    state = game.debug_state()
    state["hands"] = [sorted(hand) for hand in state["hands"]]
    del state["actions"]
    return state


def assert_same(ref: Game, fast: FastGame, where: str) -> None:
    assert fast.debug_state() == reference_state(ref), where
    assert fast.legal_actions() == ref.legal_actions(), where
    assert fast.available_to_draw() == ref.available_to_draw(), where


def play_lockstep(seed: int, starter: int, picker: random.Random) -> int:
    ref = Game(seed, starter)
    fast = FastGame(seed, starter)
    assert_same(ref, fast, f"seed={seed} start")
    steps = 0
    while not ref.is_over:
        action = picker.choice(ref.legal_actions())
        ref.step(action)
        fast.step(action)
        steps += 1
        assert_same(ref, fast, f"seed={seed} step={steps} action={action}")
    assert fast.over
    return steps


def test_random_games_match_reference_engine():
    master = random.Random(42)
    total = 0
    for _ in range(GAMES):
        seed = master.randrange(2**63)
        starter = master.randrange(2)
        total += play_lockstep(seed, starter, random.Random(seed))
    assert total > GAMES * 10


def test_draw_happy_games_match_reference_engine():
    # Players who mostly refuse to play burn through the deck and reshuffle often.
    master = random.Random(42)
    reshuffles = 0
    for _ in range(300):
        seed = master.randrange(2**31)
        ref = Game(seed, master.randrange(2))
        fast = FastGame(seed, ref.starting_player)
        picker = random.Random(seed)
        while not ref.is_over and len(ref.actions) < 2000:
            legal = ref.legal_actions()
            action = DONT_PLAY if DONT_PLAY in legal and picker.random() < 0.8 else picker.choice(legal)
            ref.step(action)
            fast.step(action)
            assert_same(ref, fast, f"seed={seed} step={len(ref.actions)}")
        reshuffles += sum(1 for event in ref.log if event["kind"] == "shuffle")
    assert reshuffles > 300


def test_saved_games_replay_identically():
    paths = sorted(GAMES_DIR.glob("*.json"))
    if not paths:
        pytest.skip("no saved games")
    for path in paths:
        record = load_record(path)
        dealt = opening_hand_of(record)
        ref = Game(record["seed"], record["starting_player"], dealt)
        fast = FastGame(record["seed"], record["starting_player"], opening_hand=dealt)
        for action in record["actions"]:
            assert action in fast.legal_actions()
            ref.step(action)
            fast.step(action)
            assert_same(ref, fast, path.name)
        # Reset and resign are written over the hand. They are not moves, so
        # replaying the action list stops before the game is over.
        if record["result"]["reason"] in ("reset", "resign"):
            assert not ref.is_over and not fast.over
            assert_same(ref, fast, path.name)
            continue
        assert fast.over
        assert {"winner": fast.winner, "reason": fast.reason} == record["result"]


def test_clone_is_independent_and_replays_the_same_shuffles():
    game = FastGame(42, 0)
    picker = random.Random(42)
    for _ in range(15):
        game.step(picker.choice(game.legal_actions()))
    copy = game.clone()
    before = game.debug_state()
    picker_a = random.Random(42)
    picker_b = random.Random(42)
    while not copy.over:
        copy.step(picker_a.choice(copy.legal_actions()))
    assert game.debug_state() == before
    while not game.over:
        game.step(picker_b.choice(game.legal_actions()))
    assert game.debug_state() == copy.debug_state()


def test_clone_with_shared_rng_does_not_touch_the_original():
    game = FastGame(42, 1)
    shared = random.Random(0)
    copy = game.clone(shared)
    state = game.rng.getstate()
    while not copy.over:
        copy.step(shared.choice(copy.legal_actions()))
    assert game.rng.getstate() == state
    assert not game.over


def test_from_prsi_game_continues_identically():
    ref = Game(42, 1)
    picker = random.Random(42)
    for _ in range(20):
        ref.step(picker.choice(ref.legal_actions()))
    fast = from_prsi_game(ref)
    assert_same(ref, fast, "converted")
    while not ref.is_over:
        action = picker.choice(ref.legal_actions())
        ref.step(action)
        fast.step(action)
        assert_same(ref, fast, "after conversion")


def test_action_limit_ends_in_a_draw():
    game = FastGame(42, 0, max_actions=5)
    for _ in range(5):
        assert not game.over
        game.step(game.legal_actions()[-1])
    assert game.over
    assert game.winner is None
    assert game.reason == REASON_LIMIT
    assert game.legal_actions() == []


def test_last_card_seven_wins_without_penalty():
    seven = Card(Suit.HEARTS, Rank.SEVEN)
    ref = Game.from_position(
        [[seven], [Card(Suit.BELLS, Rank.KING)]],
        [Card(Suit.HEARTS, Rank.NINE)],
        [Card(Suit.LEAVES, Rank.EIGHT)],
    )
    fast = from_prsi_game(ref)
    ref.step(0)
    fast.step(0)
    assert_same(ref, fast, "last seven")
    assert fast.winner == 0 and fast.pending == 0
