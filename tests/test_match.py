"""Match runner: mirrored deals, scores, CSV files and reuse."""

import csv

from bot.agents import Agent, PublicState, RandomAgent, make_agent
from bot.match import make_deals, play_game, run_match, run_tournament
from bot.fastgame import DONT_PLAY


class AlwaysDraw(Agent):
    name = "always-draw"

    def act(self, state: PublicState) -> int:
        return DONT_PLAY if DONT_PLAY in state.legal else state.legal[0]


def test_mirrored_match_writes_csv(tmp_path):
    summary = run_match({"kind": "random"}, {"kind": "random", "seed": 42}, 40,
                        workers=1, out_dir=tmp_path, progress=False)
    assert summary["games"] == 40
    assert summary["a_wins"] + summary["b_wins"] + summary["draws"] == 40
    rows = list(csv.DictReader(open(next(tmp_path.glob("*_vs_*.csv")), encoding="utf-8")))
    assert len(rows) == 40
    # Every deal appears twice, once with each seat.
    seats = {}
    for row in rows:
        seats.setdefault(row["seed"], set()).add(row["a_seat"])
    assert all(value == {"0", "1"} for value in seats.values())
    assert (tmp_path / "summary.csv").exists()


def test_parallel_match_gives_the_same_result_as_serial(tmp_path):
    serial = run_match({"kind": "random"}, {"kind": "random", "seed": 42}, 20,
                       workers=1, out_dir=None, progress=False)
    parallel = run_match({"kind": "random"}, {"kind": "random", "seed": 42}, 20,
                         workers=2, out_dir=None, progress=False)
    for key in ("a_wins", "b_wins", "draws", "avg_length"):
        assert serial[key] == parallel[key]


def test_tournament_reuses_earlier_results(tmp_path):
    specs = [{"kind": "random"}, {"kind": "random", "seed": 42, "name": "random-42"}]
    first = run_tournament(specs, 10, workers=1, out_dir=tmp_path, progress=False)
    second = run_tournament(specs, 10, workers=1, out_dir=tmp_path, progress=False)
    assert first["table"] == second["table"]
    assert len(list(tmp_path.glob("*_vs_*.csv"))) == 1


def test_the_same_deal_is_the_same_game():
    (seed, starter), = make_deals(1, 42)
    one = play_game((RandomAgent(), RandomAgent()), seed, starter)
    two = play_game((RandomAgent(), RandomAgent()), seed, starter)
    assert one.debug_state() == two.debug_state()


def test_a_player_who_never_plays_loses_to_random():
    agents = (AlwaysDraw(), make_agent({"kind": "random"}))
    deals = make_deals(30, 42)
    losses = 0
    for seed, starter in deals:
        game = play_game(agents, seed, starter, max_actions=2000)
        losses += game.winner == 1
    assert losses >= 20
