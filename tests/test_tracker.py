"""The card tracker must never claim a card is somewhere it isn't."""

import random

from bot.fastgame import (
    DONT_PLAY,
    EV_DRAW,
    EV_SHUFFLE,
    EV_STAND,
    OVERS,
    FastGame,
    cards_mask,
)
from bot.tracker import HISTORY, Tracker, events_from_log, tracker_from_observation
from prsi.game import Game

GAMES = 2000


def check_against_hidden_state(tracker: Tracker, game: FastGame) -> None:
    me = tracker.me
    opp_hand = game.hands[1 - me]
    draw = cards_mask(game.draw)
    assert tracker.my_hand == game.hands[me]
    assert tracker.discard == game.discard
    assert tracker.active_suit == game.active_suit
    assert tracker.passes == game.passes
    assert tracker.n_actions == game.n_actions
    definite = tracker.definite_opp
    pool = tracker.pool
    assert definite & opp_hand == definite, "a 'definite' card is not in the opponent's hand"
    assert definite | pool == opp_hand | draw, "unseen cards are not exactly hand + draw pile"
    assert definite & pool == 0
    assert (pool & opp_hand).bit_count() == tracker.pool_in_opp
    assert pool.bit_count() - tracker.pool_in_opp == len(game.draw)
    assert tracker.excluded & opp_hand == 0, "an excluded card is in the opponent's hand"
    assert tracker.possible_opp & opp_hand == opp_hand


def test_random_games_tracker_is_always_right():
    master = random.Random(42)
    reshuffles = definite_seen = excluded_seen = 0
    for game_index in range(GAMES):
        seed = master.randrange(2**63)
        game = FastGame(seed, master.randrange(2), max_actions=600, record_events=True)
        trackers = [Tracker(0, game.discard[0]), Tracker(1, game.discard[0])]
        picker = random.Random(seed)
        # Half the games lean towards drawing, which forces many reshuffles.
        lean = 0.7 if game_index % 2 else 0.0
        while True:
            for tracker in trackers:
                tracker.catch_up(game)
                check_against_hidden_state(tracker, game)
                definite_seen += tracker.definite_opp != 0
                excluded_seen += tracker.excluded != 0
            if game.over:
                break
            legal = game.legal_actions()
            if lean and DONT_PLAY in legal and picker.random() < lean:
                game.step(DONT_PLAY)
            else:
                game.step(picker.choice(legal))
        reshuffles += sum(1 for event in game.events if event[0] == EV_SHUFFLE)
    assert reshuffles > 1000
    assert definite_seen > 1000
    assert excluded_seen > 0


def test_observation_and_engine_paths_agree():
    master = random.Random(42)
    for _ in range(120):
        seed = master.randrange(2**63)
        ref = Game(seed, master.randrange(2))
        fast = FastGame(seed, ref.starting_player, record_events=True)
        trackers = [Tracker(0, fast.discard[0]), Tracker(1, fast.discard[0])]
        picker = random.Random(seed)
        while True:
            assert events_from_log(ref.log) == fast.events
            for seat in (0, 1):
                assert ref.drawn[seat] == fast.drawn[seat]
                from_obs = tracker_from_observation(ref.observation(seat))
                from_engine = trackers[seat].catch_up(fast)
                assert from_obs.snapshot() == from_engine.snapshot()
            if ref.is_over or len(ref.actions) > 200:
                break
            legal = ref.legal_actions()
            action = DONT_PLAY if picker.random() < 0.5 and DONT_PLAY in legal else picker.choice(legal)
            ref.step(action)
            fast.step(action)


def position(mine, theirs, draw, top, *, rest="discard", **state) -> FastGame:
    """A full 32-card position; cards not listed go to ``rest`` (discard, mine or theirs)."""
    used = set(mine) | set(theirs) | set(draw) | {top}
    others = [card for card in range(32) if card not in used]
    mine, theirs, discard = list(mine), list(theirs), []
    {"discard": discard, "mine": mine, "theirs": theirs}[rest].extend(others)
    discard.append(top)
    state.setdefault("active_suit", top >> 3)
    return FastGame.from_state(
        [cards_mask(mine), cards_mask(theirs)],
        list(draw),
        discard,
        rng=random.Random(0),
        record_events=True,
        **state,
    )


def _tracked(game: FastGame, me: int = 0) -> Tracker:
    return Tracker.from_discard(me, game.discard, game.active_suit).catch_up(game)


def test_reshuffle_makes_the_opponent_hand_exact():
    # Seat 1 draws the last card; the discard pile (minus its top) is reshuffled.
    game = position([0, 9], [18, 27], [20], 17, current=1)
    tracker = _tracked(game)
    assert tracker.definite_opp == 0
    check_against_hidden_state(tracker, game)
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert game.events[-1][0] == EV_SHUFFLE
    assert tracker.definite_opp == game.hands[1] == cards_mask([18, 27, 20])
    assert tracker.pool_in_opp == 0
    assert tracker.pool == cards_mask(game.draw)
    check_against_hidden_state(tracker, game)


def test_reshuffle_in_the_middle_of_a_penalty_draw():
    # Two 7s pending (4 cards), only 2 in the draw pile: draw 2, reshuffle, draw 2.
    game = position([1], [10, 11], [21, 22], 8, pending=2, current=1)
    tracker = _tracked(game)
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert [event[0] for event in game.events] == [EV_SHUFFLE, EV_DRAW]
    assert game.events[-1][2] == 4
    assert tracker.definite_opp == cards_mask([10, 11, 21, 22])
    assert tracker.pool_in_opp == 2
    check_against_hidden_state(tracker, game)


def test_penalty_bigger_than_the_cards_left():
    # Four 7s pending (8 cards) but only 1 + 2 cards exist.
    game = position([1], [10], [30], 24, rest="mine", pending=4, current=1)
    game.discard[:0] = [12, 13]
    game.hands[0] &= ~cards_mask([12, 13])
    tracker = _tracked(game)
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert game.hands[1].bit_count() == 4
    assert len(game.draw) == 0 and game.discard == [24]
    assert tracker.definite_opp == cards_mask([10, 30])
    assert tracker.pool == cards_mask([12, 13])
    assert tracker.pool_in_opp == 2
    check_against_hidden_state(tracker, game)


def test_pass_deduction_and_when_it_is_cleared():
    # Nothing to draw, Hearts 9 on top, the opponent holds only Bells 7 and 8.
    game = position([3], [24, 25], [], 2, rest="mine", current=1)
    tracker = _tracked(game)
    assert game.legal_actions() == [DONT_PLAY]
    game.step(DONT_PLAY)  # a real pass
    tracker.catch_up(game)
    hearts_and_nines = 0xFF | cards_mask([2, 10, 18, 26])
    assert tracker.excluded == hearts_and_nines | OVERS
    check_against_hidden_state(tracker, game)
    game.step(3)  # I play the Hearts 10
    tracker.catch_up(game)
    assert tracker.excluded != 0  # playing does not clear it
    check_against_hidden_state(tracker, game)
    game.step(DONT_PLAY)  # the opponent draws the Hearts 9 after a reshuffle
    tracker.catch_up(game)
    assert tracker.excluded == 0
    check_against_hidden_state(tracker, game)


def test_history_clears_when_the_hand_is_known_and_restarts_on_a_draw():
    game = position([0, 9], [18, 27], [20], 17, current=1)
    tracker = _tracked(game)
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert tracker.opponent_hand_certain()
    assert tracker.recent == []
    assert game.current == 0
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert tracker.opponent_hand_certain()
    assert tracker.recent == []
    assert game.current == 1
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert not tracker.opponent_hand_certain()
    assert len(tracker.recent) == 1
    assert tracker.recent[0][0] == EV_DRAW
    assert tracker.recent[0][1] == 1


def test_history_keeps_only_the_latest_window_while_uncertain():
    tracker = Tracker(0, 0)
    hand = cards_mask(range(1, 7))
    stands = [(EV_STAND, 1, 0, 0)] * (HISTORY + 5)
    tracker._replay(stands, hand, 4, 10, [], 0)
    assert len(tracker.recent) == HISTORY
    assert tracker.recent[0] == stands[5]
    assert tracker.recent[-1] == stands[-1]
    assert not tracker.opponent_hand_certain()


def test_my_own_pass_teaches_nothing():
    game = position([24], [], [], 2, rest="theirs", current=0)
    tracker = _tracked(game)
    assert game.legal_actions() == [DONT_PLAY]
    game.step(DONT_PLAY)
    tracker.catch_up(game)
    assert tracker.excluded == 0
