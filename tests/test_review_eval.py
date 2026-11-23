"""The review bar: seat 0's win chance from the real hands."""

import json
import random
import time

import pytest
from fastapi.testclient import TestClient

from prsi.actions import DONT_PLAY, play_card_action
from prsi.cards import Card, Rank, Suit
from prsi.game import Game
from prsi.record import save_record
from server.app import create_app
from server.review import (
    EVAL_SAMPLES,
    analysis_paths,
    bar_snapshot,
    fill_eval,
    position_chances,
    read_status,
    review_spec,
    score_snapshot,
    seat0_chance,
    spec_version,
)
from server.room import Room

BOT = {"kind": "ismcts", "label": "test-bot", "iterations": 1}


def _finished_chance(winner):
    if winner is None:
        return 0.5
    return 1.0 if winner == 0 else 0.0


class Play:
    def __init__(self, action, greedy=False):
        self.action = action
        self.greedy = greedy
        self.calls = 0

    def new_game(self, seed):
        pass

    def act(self, state):
        self.calls += 1
        assert self.action in state.legal
        return self.action


def test_last_card_is_a_win_and_one_greedy_playout_is_enough():
    card = Card(Suit.HEARTS, Rank.KING)
    game = Game.from_position(
        hands=[[card], [Card(Suit.LEAVES, Rank.EIGHT), Card(Suit.ACORNS, Rank.TEN)]],
        discard=[Card(Suit.HEARTS, Rank.NINE)],
        draw=[],
        current_player=0,
    )
    agent = Play(play_card_action(card), greedy=True)
    assert seat0_chance(game, agent, random.Random(0), samples=5) == 1.0
    assert agent.calls == 1
    game.step(play_card_action(card))
    assert seat0_chance(game, agent, random.Random(0), samples=5) == 1.0
    assert agent.calls == 1


def test_finished_loss_and_draw_are_exact():
    card = Card(Suit.HEARTS, Rank.KING)
    loss = Game.from_position(
        hands=[[Card(Suit.LEAVES, Rank.EIGHT)], [card]],
        discard=[Card(Suit.HEARTS, Rank.NINE)],
        draw=[],
        current_player=1,
    )
    loss.step(play_card_action(card))
    assert seat0_chance(loss, Play(DONT_PLAY), random.Random(0), samples=3) == 0.0

    drawn = Game.from_position(
        hands=[[Card(Suit.LEAVES, Rank.EIGHT)], [Card(Suit.ACORNS, Rank.TEN)]],
        discard=[Card(Suit.HEARTS, Rank.NINE)],
        draw=[],
        current_player=0,
    )
    drawn.step(DONT_PLAY)
    drawn.step(DONT_PLAY)
    assert drawn.result["winner"] is None
    assert seat0_chance(drawn, Play(DONT_PLAY), random.Random(0), samples=3) == 0.5


def test_same_seed_scores_the_same_opening(monkeypatch):
    monkeypatch.setattr("server.review.EVAL_SAMPLES", 3)
    record = {"seed": 42, "starting_player": 0, "actions": []}
    spec = {"kind": "random", "seed": 42}
    first = position_chances(record, spec, lambda done, total: None)
    second = position_chances(record, spec, lambda done, total: None)
    assert first == second
    assert len(first) == 1
    assert 0.0 <= first[0] <= 1.0


def test_fill_eval_keeps_the_move_list(tmp_path, monkeypatch):
    monkeypatch.setattr("server.review.EVAL_SAMPLES", 2)
    game = Game(42, 1)
    rng = random.Random(42)
    while not game.is_over:
        game.step(rng.choice(game.legal_actions()))
    save_record(game, tmp_path / "sample.json", names=["Ada", "Bo"])
    version = spec_version(BOT)
    result_path, progress_path = analysis_paths(tmp_path, "sample.json", version)
    result_path.parent.mkdir(parents=True)
    moves = [{"index": 0, "kept": True}]
    result_path.write_text(
        json.dumps({"file": "sample.json", "moves": moves, "version": version, "seconds": 1.5}),
        encoding="utf-8",
    )

    assert read_status(tmp_path, "sample.json", version)["status"] == "done"
    progress_path.write_text(
        json.dumps({"status": "running", "done": 1, "total": 4, "phase": "bar"}),
        encoding="utf-8",
    )
    running = read_status(tmp_path, "sample.json", version)
    assert running["status"] == "running"
    assert running["phase"] == "bar"
    assert running["result"]["moves"] == moves
    progress_path.unlink()

    assert fill_eval(str(tmp_path), "sample.json", BOT) == {"status": "done"}
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["moves"] == moves
    assert saved["seconds"] == 1.5
    assert saved["eval_kind"] == "random"
    assert saved["eval_samples"] == 2
    assert len(saved["eval"]) == len(game.actions) + 1
    assert saved["eval"][-1] == _finished_chance(game.result["winner"])
    assert all(0.0 <= chance <= 1.0 for chance in saved["eval"])
    assert not progress_path.exists()
    assert read_status(tmp_path, "sample.json", version)["status"] == "done"


def test_analyse_endpoint_scores_only_the_bar_when_moves_exist(tmp_path, monkeypatch):
    called = {}

    def fake_fill(games_dir, name, spec):
        called["job"] = "fill"
        called["name"] = name
        return {"status": "done"}

    def fake_analyse(games_dir, name, spec):
        called["job"] = "analyse"
        return {"status": "done"}

    monkeypatch.setattr("server.app.fill_eval", fake_fill)
    monkeypatch.setattr("server.app.analyse_game", fake_analyse)
    monkeypatch.setattr("server.bot.executor", lambda: None)

    (tmp_path / "sample.json").write_text(
        json.dumps({"version": 2, "seed": 42, "starting_player": 0, "actions": [], "names": ["A", "B"]}),
        encoding="utf-8",
    )
    version = spec_version(BOT)
    result_path, _ = analysis_paths(tmp_path, "sample.json", version)
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"moves": [{"index": 0}], "version": version}), encoding="utf-8")
    room = Room("TEST01", tmp_path, host_token="secret-host", bot_spec=BOT, bot_note="test bot")

    with TestClient(create_app(room)) as client:
        started = client.post("/api/review/analyse", params={"host": "secret-host", "file": "sample.json"})
        assert started.status_code == 200
        assert started.json()["status"] == "running"
        assert started.json()["phase"] == "bar"
        future = client.app.state.analyses[("sample.json", version)]
        deadline = time.time() + 5
        while not future.done():
            if time.time() > deadline:
                raise AssertionError("the bar job did not finish")
            time.sleep(0.05)
        assert future.result() == {"status": "done"}
    assert called == {"job": "fill", "name": "sample.json"}

    result_path.write_text(
        json.dumps({"moves": [{"index": 0}], "eval": [0.5], "version": version}),
        encoding="utf-8",
    )
    called.clear()
    with TestClient(create_app(room)) as client:
        again = client.post("/api/review/analyse", params={"host": "secret-host", "file": "sample.json"})
        assert again.json()["status"] == "done"
    assert called == {}


def test_finished_snapshot_is_the_real_result():
    card = Card(Suit.HEARTS, Rank.KING)
    game = Game.from_position(
        hands=[[card], [Card(Suit.LEAVES, Rank.EIGHT)]],
        discard=[Card(Suit.HEARTS, Rank.NINE)],
        draw=[],
        current_player=0,
    )
    game.step(play_card_action(card))
    scored = score_snapshot(bar_snapshot(game), {"kind": "random"}, 42)
    assert scored == {"chance": 1.0, "kind": "random", "samples": 1, "cap": 1}


def test_snapshot_can_score_a_dealt_game(monkeypatch):
    monkeypatch.setattr("server.review.EVAL_SAMPLES", 2)
    game = Game(42, 0)
    scored = score_snapshot(bar_snapshot(game), {"kind": "random", "seed": 42}, 42)
    assert scored["kind"] == "random"
    assert 0.0 <= scored["chance"] <= 1.0


def test_default_sample_count_is_the_review_budget():
    assert EVAL_SAMPLES == 1000


def test_review_spec_uses_the_requested_budget():
    spec = review_spec(
        {
            "kind": "guided",
            "path": "runs/league/gen_003",
            "seconds": 3.0,
            "label": "gen_003",
            "review_line_limit": 50_000,
            "bar_samples": 40,
        }
    )
    assert spec["line_limit"] == 50_000
    assert spec["bar_samples"] == 40
    assert spec["seed"] == 42
    assert "seconds" not in spec
    assert "iterations" not in spec
    assert "review_line_limit" not in spec


def test_review_spec_keeps_the_default_budget():
    spec = review_spec({"kind": "ismcts", "label": "plain ISMCTS", "seconds": 5.0})
    assert spec["line_limit"] == 10_000
    assert spec["bar_samples"] == 1000
    assert "seconds" not in spec
    assert "iterations" not in spec


def test_spec_version_keeps_the_default_cache_name():
    assert spec_version({"kind": "ismcts", "label": "gen_003"}) == "gen_003-line-one"
    assert (
        spec_version(
            {"kind": "ismcts", "label": "gen_003", "review_line_limit": 10_000, "bar_samples": 1000}
        )
        == "gen_003-line-one"
    )


def test_spec_version_includes_a_changed_budget():
    assert (
        spec_version({"kind": "guided", "label": "gen_003", "review_line_limit": 50_000})
        == "gen_003-line-one-50000-1000"
    )
    assert spec_version({"kind": "guided", "label": "gen_003", "bar_samples": 40}) == "gen_003-line-one-10000-40"


def test_live_bar_request_carries_the_sample_count():
    from server.room import _bar_spec

    carried = _bar_spec({"kind": "guided", "path": "runs/x", "bar_samples": 40})
    assert carried["bar_samples"] == 40
    assert carried["path"] == "runs/x"
    assert "bar_samples" not in _bar_spec({"kind": "random"})


def test_score_snapshot_uses_the_spec_sample_count(monkeypatch):
    seen = {}

    def fake_chance(game, agent, rng, samples, progress=None, stop=None):
        seen["samples"] = samples
        return 0.25

    monkeypatch.setattr("server.review.seat0_chance", fake_chance)
    game = Game(42, 0)
    scored = score_snapshot(bar_snapshot(game), {"kind": "random", "seed": 42, "bar_samples": 7}, 42)
    assert seen["samples"] == 7
    assert scored == {"chance": 0.25, "kind": "random", "samples": 7, "cap": 7}


def _one_move_win():
    card = Card(Suit.HEARTS, Rank.KING)
    game = Game.from_position(
        hands=[[card], [Card(Suit.LEAVES, Rank.EIGHT)]],
        discard=[Card(Suit.HEARTS, Rank.NINE)],
        draw=[],
        current_player=0,
    )
    return game, Play(play_card_action(card))


def test_running_mean_reports_every_finish_and_matches_the_return(monkeypatch):
    clock = {"t": 0.0}

    def perf():
        clock["t"] += 0.3
        return clock["t"]

    monkeypatch.setattr("server.review.time.perf_counter", perf)
    game, agent = _one_move_win()
    seen = []
    value = seat0_chance(game, agent, random.Random(0), 4, progress=lambda *row: seen.append(row))
    assert [row[1] for row in seen] == [1, 2, 3, 4]
    assert all(row[2] == 4 for row in seen)
    assert seen[-1][0] == value == 1.0


def test_bar_stops_between_finishes():
    game, agent = _one_move_win()

    class Halt:
        def __init__(self):
            self.n = 0

        def is_set(self):
            return self.n >= 1

    halt = Halt()

    def progress(chance, played, cap):
        halt.n = played
        assert cap == 5

    value = seat0_chance(game, agent, random.Random(0), 5, progress=progress, stop=halt)
    assert agent.calls == 1
    assert value == 1.0


def test_bar_stopped_before_a_finish_returns_nothing():
    game, agent = _one_move_win()

    class Halt:
        def is_set(self):
            return True

    assert seat0_chance(game, agent, random.Random(0), 4, stop=Halt()) is None
    assert agent.calls == 0


def test_score_snapshot_streams_until_the_cap():
    import queue

    game, _agent = _one_move_win()
    updates = queue.Queue()
    token = (3, 9)
    spec = {"kind": "random", "seed": 42, "bar_samples": 4}
    result = score_snapshot(bar_snapshot(game), spec, 42, updates, None, token)
    items = []
    while True:
        try:
            items.append(updates.get_nowait())
        except queue.Empty:
            break
    assert items
    assert all(item[0] == token for item in items)
    assert items[0][2] == 1
    assert items[-1][2] == result["samples"] == result["cap"] == 4
    assert items[-1][1] == result["chance"]
    assert all(items[i][2] < items[i + 1][2] for i in range(len(items) - 1))
    assert result["kind"] == "random"

    class Halt:
        def is_set(self):
            return True

    assert score_snapshot(bar_snapshot(game), spec, 5, queue.Queue(), Halt(), token) is None


def test_server_rejects_a_budget_below_one():
    from server.__main__ import main

    with pytest.raises(SystemExit, match="review line limit"):
        main(["--no-bot", "--review-line-limit", "0"])
    with pytest.raises(SystemExit, match="bar sample count"):
        main(["--no-bot", "--bar-samples", "0"])
