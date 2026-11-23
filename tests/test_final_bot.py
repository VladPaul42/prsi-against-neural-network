"""The final bot (network-guided ISMCTS) and the network alone: legal moves, no cheating."""

import random

import pytest

pytest.importorskip("torch")

from bot.agents import PublicState  # noqa: E402
from bot.fastgame import DONT_PLAY, FastGame  # noqa: E402
from bot.guided import GuidedAgent  # noqa: E402
from bot.ismcts import ISMCTSAgent  # noqa: E402
from bot.net import PolicyAgent, build_model, load_numpy_net, save_network  # noqa: E402
from bot.tracker import Tracker  # noqa: E402
from prsi.game import Game  # noqa: E402
from tests.test_ismcts import known_position, mid_game, other_hidden_deal  # noqa: E402


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    folder = tmp_path_factory.mktemp("net")
    import torch

    torch.manual_seed(0)
    save_network(build_model(hidden=64), folder)
    return load_numpy_net(folder)


def test_guided_bot_same_observation_different_hidden_cards(net):
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
        results = []
        for world in (game, twin):
            agent = GuidedAgent(net, iterations=150, seed=42)
            agent.new_game(42)
            results.append(agent.analyse(PublicState.from_fast(world, Tracker(me, world.discard[0]))))
        assert results[0]["action"] == results[1]["action"]
        assert results[0]["moves"] == results[1]["moves"]
        assert results[0]["win_chance"] == results[1]["win_chance"]
        checked += 1
    assert checked >= 10


def test_guided_tries_moves_evenly_when_the_network_is_sure(net):
    state = known_position()
    agent = GuidedAgent(net, iterations=180, seed=42)

    def skewed(seat, tracker, pending, ace, suit_named, legal):
        probs = [0.01] * len(legal)
        probs[0] = 1.0 - 0.01 * (len(legal) - 1)
        return probs, 0.25

    agent.search_engine._evaluate = skewed
    result = agent.analyse(state)
    visits = [move["visits"] for move in result["moves"]]
    priors = [move["prior"] for move in result["moves"]]
    assert priors[0] > 0.5
    assert min(priors[1:]) < 0.05
    assert min(visits) > 0
    assert max(visits) - min(visits) <= 1
    assert sum(visits) == result["iterations"]
    best = max(move["win_chance"] for move in result["moves"])
    chosen = next(move for move in result["moves"] if move["action"] == result["action"])
    assert chosen["win_chance"] == best


def test_guided_bot_reports_network_opinion(net):
    game, tracker = mid_game(42, 3)
    state = PublicState.from_fast(game, tracker)
    result = GuidedAgent(net, iterations=100).analyse(state)
    priors = [move["prior"] for move in result["moves"]]
    assert all(p is not None for p in priors)
    assert abs(sum(priors) - 1.0) < 1e-6
    assert result["action"] in state.legal


def test_every_bot_plays_legal_moves_from_web_observations(net):
    """Real games through ``prsi.Game`` and its observation, the way the web app feeds bots."""
    bots = [
        ISMCTSAgent(iterations=20, seed=42),
        PolicyAgent(net, greedy=True),
        PolicyAgent(net, greedy=False, seed=42),
        GuidedAgent(net, iterations=20, seed=42),
    ]
    rng = random.Random(42)
    seen = {"penalty_too_big": 0, "nothing_to_draw": 0, "stuck": 0}
    for seed in range(30):
        game = Game(seed, seed % 2)
        while not game.is_over and len(game.actions) < 300:
            view = game.observation(game.current)
            if game.pending_sevens and game.pending_sevens * 2 > game.available_to_draw():
                seen["penalty_too_big"] += 1
            if game.available_to_draw() == 0:
                seen["nothing_to_draw"] += 1
            if game.consecutive_passes == 1:
                # The opponent just passed: one more pass and both players are stuck.
                seen["stuck"] += 1
            state = PublicState.from_observation(view)
            if rng.random() < 0.35:
                bot = bots[rng.randrange(len(bots))]
                action = bot.act(state)
                assert action in view["legal_actions"], type(bot).__name__
            else:
                legal = view["legal_actions"]
                action = DONT_PLAY if DONT_PLAY in legal and rng.random() < 0.75 else rng.choice(legal)
            game.step(action)
    assert all(count > 0 for count in seen.values()), seen


def test_policy_agent_analysis_shape(net):
    game = FastGame(42, 0, record_events=True)
    state = PublicState.from_fast(game, Tracker(0, game.discard[0]))
    result = PolicyAgent(net).analyse(state)
    assert [m["action"] for m in result["moves"]] == state.legal
    assert 0.0 <= result["win_chance"] <= 1.0
