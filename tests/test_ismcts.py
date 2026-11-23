"""Plain ISMCTS: legal moves, budgets, repeatability, and no cheating."""

import random
import time

from bot.agents import PublicState, RandomAgent
from bot.fastgame import DONT_PLAY, FastGame, cards_mask, mask_cards
from bot.ismcts import ISMCTSAgent, Node, backup_best
from bot.match import make_deals, play_game
from bot.tracker import Tracker


def mid_game(seed: int, moves: int) -> tuple[FastGame, Tracker]:
    game = FastGame(seed, 0, record_events=True)
    picker = random.Random(seed)
    for _ in range(moves):
        if game.over:
            break
        game.step(picker.choice(game.legal_actions()))
    return game, Tracker(game.current, game.discard[0])


def open_position(minimum: int = 2) -> PublicState:
    """A mid-game seat with at least ``minimum`` legal moves."""
    for seed in range(80):
        game, tracker = mid_game(seed, 3 + seed % 20)
        if game.over:
            continue
        state = PublicState.from_fast(game, tracker)
        if len(state.legal) >= minimum:
            return state
    raise AssertionError("no position with several legal moves")


def known_position(moves: int = 2) -> PublicState:
    """A seat with exactly ``moves`` legal actions and a known opponent hand.

    Two root moves that are legal in every imagined deal are visited equally.
    More than two can stay a visit apart because a move is counted as available
    while a later sibling is tried for the first time.
    """
    for seed in range(200):
        game = FastGame(seed, 0, record_events=True)
        picker = random.Random(seed)
        for _ in range(80):
            if game.over:
                break
            tracker = Tracker(game.current, game.discard[0])
            state = PublicState.from_fast(game, tracker)
            allowed = (tracker.pool & ~tracker.excluded).bit_count()
            known = tracker.pool_in_opp == 0 or tracker.pool_in_opp == allowed
            if known and len(state.legal) == moves:
                return state
            game.step(picker.choice(state.legal))
    raise AssertionError("no known position with that many legal moves")


def other_hidden_deal(game: FastGame, tracker: Tracker, rng: random.Random) -> FastGame:
    """The same public history, but the unseen pool cards dealt differently."""
    tracker.catch_up(game)
    copy = game.clone(events=True)
    me = tracker.me
    pool = mask_cards(tracker.pool)
    rng.shuffle(pool)
    k = tracker.pool_in_opp
    copy.hands[1 - me] = tracker.definite_opp | cards_mask(pool[:k])
    copy.draw = pool[k:]
    return copy


def test_same_observation_different_hidden_cards_same_decision():
    checked = 0
    for seed in range(40):
        game, _ = mid_game(seed, 6 + seed % 30)
        if game.over:
            continue
        me = game.current
        tracker = Tracker(me, game.discard[0]).catch_up(game)
        if tracker.pool_in_opp == 0 or len(game.draw) < 2:
            continue
        twin = other_hidden_deal(game, tracker, random.Random(seed))
        assert twin.hands[1 - me] != game.hands[1 - me] or twin.draw != game.draw
        results = []
        for world in (game, twin):
            agent = ISMCTSAgent(iterations=300, seed=42)
            agent.new_game(42)
            state = PublicState.from_fast(world, Tracker(me, world.discard[0]))
            results.append(agent.analyse(state))
        assert results[0]["action"] == results[1]["action"]
        assert results[0]["moves"] == results[1]["moves"]
        checked += 1
    assert checked >= 10


def test_always_legal_including_odd_positions():
    agent = ISMCTSAgent(iterations=25, seed=42)
    rng = random.Random(42)
    for seed in range(25):
        game = FastGame(seed, seed % 2, record_events=True, max_actions=250)
        trackers = (Tracker(0, game.discard[0]), Tracker(1, game.discard[0]))
        while not game.over:
            state = PublicState.from_fast(game, trackers[game.current])
            if rng.random() < 0.3:
                action = agent.act(state)
            else:
                # Mostly draw, to reach empty piles, big penalties and stuck players.
                action = DONT_PLAY if DONT_PLAY in state.legal and rng.random() < 0.7 else rng.choice(state.legal)
            assert action in state.legal
            game.step(action)


def test_time_budget_is_respected():
    game, tracker = mid_game(3, 4)
    agent = ISMCTSAgent(seconds=0.3)
    state = PublicState.from_fast(game, tracker)
    start = time.perf_counter()
    result = agent.analyse(state)
    elapsed = time.perf_counter() - start
    assert 0.25 < elapsed < 0.6
    assert result["iterations"] > 50
    assert result["action"] in state.legal


def test_progress_reports_and_stats_shape():
    for seed in range(50):
        game, tracker = mid_game(seed, 2)
        state = PublicState.from_fast(game, tracker)
        if len(state.legal) > 2:
            break
    seen = []
    result = ISMCTSAgent(seconds=0.6).analyse(state, progress=seen.append)
    assert seen
    assert [move["action"] for move in result["moves"]] == state.legal
    assert sum(move["visits"] for move in result["moves"]) == result["iterations"]
    assert 0.0 <= result["win_chance"] <= 1.0


def test_plain_rollout_can_decline_and_specs_can_pin_the_old_habit():
    agent = ISMCTSAgent.from_spec({"kind": "ismcts", "iterations": 1})
    assert agent.search_engine.evaluator.draw_prob is None
    pinned = ISMCTSAgent.from_spec({"kind": "ismcts", "iterations": 1, "draw_prob": 0.0})
    assert pinned.search_engine.evaluator.draw_prob == 0.0


def test_root_moves_are_tried_evenly_and_the_best_chance_is_played():
    state = known_position()
    result = ISMCTSAgent(iterations=240, seed=42).analyse(state)
    visits = [move["visits"] for move in result["moves"]]
    assert min(visits) > 0
    assert max(visits) - min(visits) <= 1
    assert sum(visits) == result["iterations"]
    best = max(move["win_chance"] for move in result["moves"])
    chosen = next(move for move in result["moves"] if move["action"] == result["action"])
    assert chosen["win_chance"] == best


def test_backup_uses_the_best_tried_reply():
    """A tried loss for the opponent counts, and so does the bot's best reply.

    The average of the two replies would be 0.5. Best play for the opponent
    keeps the loss.
    """
    me = 0
    opponent = 1
    good = Node(opponent)
    good.visits = 1
    good.reward = 1.0
    bad = Node(opponent)
    bad.visits = 1
    bad.reward = 0.0
    played = Node(me)
    played.children = {1: good, 2: bad}
    backup_best([played, bad], [[1, 2], None], 0.0, me)
    assert played.visits == 1
    assert played.reward == 0.0

    weak = Node(me)
    weak.visits = 1
    weak.reward = 0.2
    strong = Node(me)
    strong.visits = 1
    strong.reward = 0.9
    reply = Node(opponent)
    reply.children = {7: weak, 8: strong}
    backup_best([reply, strong], [[7, 8], None], 0.9, me)
    assert reply.visits == 1
    assert reply.reward == 0.9


class _StopAfter:
    """Becomes set after ``calls`` checks, so the search stops on its cadence."""

    def __init__(self, calls: int):
        self.calls = calls
        self.seen = 0

    def is_set(self) -> bool:
        self.seen += 1
        return self.seen > self.calls


def forced_stand() -> PublicState:
    """Ace is active and this seat has no Ace, so the only legal move is to stand."""
    hand = cards_mask([0, 1, 2, 3])
    opp = cards_mask([8, 9, 10, 11])
    discard = [16]
    used = hand | opp | (1 << discard[0])
    draw = [card for card in range(32) if not (used >> card) & 1]
    game = FastGame.from_state(
        [hand, opp],
        draw,
        discard,
        active_suit=discard[0] >> 3,
        ace=True,
        current=0,
        record_events=True,
    )
    tracker = Tracker.from_discard(0, discard, discard[0] >> 3)
    tracker.sync(hand, opp.bit_count(), len(draw))
    state = PublicState.from_fast(game, tracker)
    assert state.legal == [DONT_PLAY]
    return state


def test_stop_ends_the_search_before_the_budget():
    state = open_position()
    stopped = ISMCTSAgent(iterations=500, seed=42).analyse(state, stop=_StopAfter(1), iterations=500)
    # The clock is checked every 8 iterations, and the first check does not stop.
    assert stopped["iterations"] == 8
    assert stopped["action"] in state.legal
    immediate = ISMCTSAgent(iterations=500, seed=42).analyse(state, stop=_StopAfter(0), iterations=500)
    assert immediate["iterations"] == 0
    assert immediate["action"] == state.legal[0]


def test_forced_move_keeps_searching_until_stop_or_the_budget():
    state = forced_stand()
    short = ISMCTSAgent(iterations=500, seed=42).analyse(state)
    assert short["iterations"] == 500
    assert short["action"] == DONT_PLAY
    start = time.perf_counter()
    timed = ISMCTSAgent(seconds=0.2, seed=42).analyse(state)
    elapsed = time.perf_counter() - start
    assert 0.15 < elapsed < 0.5
    assert timed["iterations"] > 64
    assert timed["action"] == DONT_PLAY
    full = ISMCTSAgent(seconds=30, seed=42).analyse(state, stop=_StopAfter(10_000), iterations=200)
    assert full["iterations"] == 200
    assert full["action"] == DONT_PLAY


def test_iteration_override_ignores_the_clock():
    state = open_position()
    start = time.perf_counter()
    result = ISMCTSAgent(seconds=30, seed=42).analyse(state, iterations=40)
    assert time.perf_counter() - start < 5
    assert result["iterations"] == 40
    assert result["action"] in state.legal


def test_iteration_cap_and_clock_both_apply():
    state = open_position()
    start = time.perf_counter()
    capped = ISMCTSAgent(seconds=30, seed=42).analyse(state, iterations=40, seconds=30)
    assert time.perf_counter() - start < 5
    assert capped["iterations"] == 40
    assert capped["action"] in state.legal

    start = time.perf_counter()
    timed = ISMCTSAgent(iterations=100_000, seed=42).analyse(state, iterations=100_000, seconds=0.15)
    elapsed = time.perf_counter() - start
    assert 0.1 < elapsed < 0.6
    assert timed["iterations"] < 100_000
    assert timed["action"] in state.legal


def test_beats_random_on_mirrored_deals():
    agents = [ISMCTSAgent(iterations=200, seed=42), RandomAgent(0)]
    score = 0.0
    deals = make_deals(15, 42)
    for seed, starter in deals:
        for bot_seat in (0, 1):
            pair = agents if bot_seat == 0 else agents[::-1]
            game = play_game(pair, seed, starter)
            score += 0.5 if game.winner is None else float(game.winner == bot_seat)
    assert score / 30 >= 0.7
