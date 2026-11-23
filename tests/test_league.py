"""A tiny league: generations, evaluation, best network, and stop/resume."""

import csv
import json

import pytest

torch = pytest.importorskip("torch")

import bot.league as league_module  # noqa: E402
from bot.league import League, best_generation_dir, run_league  # noqa: E402
from bot.ppo import LeagueConfig, PPOConfig  # noqa: E402


def tiny_config(run_dir, **changes) -> LeagueConfig:
    settings = dict(
        run_dir=str(run_dir),
        workers=2,
        torch_threads=1,
        max_generations=2,
        updates_per_gen=2,
        train_opponent={"kind": "ismcts", "iterations": 10},
        eval_opponent={"kind": "ismcts", "iterations": 20},
        eval_games=8,
        checkpoint_minutes=0.0,
        ppo=PPOConfig(games_per_update=8, games_per_job=4, minibatch=64, hidden=32),
    )
    settings.update(changes)
    return LeagueConfig(**settings)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_two_generations_and_a_best(tmp_path):
    summary = run_league(tiny_config(tmp_path))
    assert [g["generation"] for g in summary["generations"]] == [1, 2]
    assert summary["best"] in (1, 2)
    assert summary["generations"][0]["became_best"] is True
    for generation in (1, 2):
        folder = tmp_path / f"gen_{generation:03d}"
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        assert (folder / "model.pt").exists() and (folder / "weights.npz").exists()
        assert meta["evaluation"]["games_per_opponent"] == 8
    # Generation 2 trained against ISMCTS and generation 1, never itself.
    meta2 = json.loads((tmp_path / "gen_002" / "meta.json").read_text(encoding="utf-8"))
    assert meta2["opponents"] == ["baseline", "gen001"]
    assert set(meta2["evaluation"]["scores"]) == {"gen001", "ismcts-20it"}
    training = read_csv(tmp_path / "logs" / "training.csv")
    assert len(training) == 4
    opponents = read_csv(tmp_path / "logs" / "opponents.csv")
    assert opponents and "games_this_update" in opponents[0]
    played = {}
    for row in opponents:
        key = (row["generation"], row["update"])
        played[key] = played.get(key, 0) + int(row["games_this_update"])
    assert set(played.values()) == {8}
    assert "pool_score" not in training[0]
    assert {row["reference"] for row in training if row["generation"] == "1"} == {"baseline"}
    assert {row["reference"] for row in training if row["generation"] == "2"} == {"gen001"}
    generations = read_csv(tmp_path / "logs" / "generations.csv")
    assert len(generations) == 2
    assert "pool_score" not in generations[0]
    assert [row["reference"] for row in generations] == ["baseline", "gen001"]
    assert best_generation_dir(tmp_path) == tmp_path / f"gen_{summary['best']:03d}"
    # The league is over: running again does nothing new.
    again = run_league(tiny_config(tmp_path))
    assert len(again["generations"]) == 2


def test_only_n_generations_per_call(tmp_path):
    first = run_league(tiny_config(tmp_path, max_generations=5), generations=1)
    assert len(first["generations"]) == 1
    second = run_league(tiny_config(tmp_path, max_generations=5), generations=1)
    assert len(second["generations"]) == 2


def test_stop_and_resume_continues_from_the_checkpoint(tmp_path, monkeypatch):
    real_update = league_module.ppo_update
    calls = {"n": 0}

    def interrupt_on_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_update(*args, **kwargs)

    monkeypatch.setattr(league_module, "ppo_update", interrupt_on_third)
    config = tiny_config(tmp_path, updates_per_gen=4, max_generations=1)
    League(config).run()
    checkpoints = sorted((tmp_path / "checkpoints").glob("*.pt"))
    assert checkpoints and checkpoints[-1].name.endswith("u000002.pt")
    saved = torch.load(checkpoints[-1], weights_only=False)
    assert saved["progress"]["update"] == 2
    state = json.loads((tmp_path / "league_state.json").read_text(encoding="utf-8"))
    assert state["current"] == {"generation": 1, "phase": "training"}

    # Resume: the model starts exactly from the checkpoint and the update count continues.
    seen = {}

    def remember_start(model, *args, **kwargs):
        if "first" not in seen:
            seen["first"] = {k: v.clone() for k, v in model.state_dict().items()}
        return real_update(model, *args, **kwargs)

    monkeypatch.setattr(league_module, "ppo_update", remember_start)
    summary = League(config).run()
    for key, value in saved["model"].items():
        assert torch.equal(seen["first"][key], value)
    updates = [int(row["update"]) for row in read_csv(tmp_path / "logs" / "training.csv")]
    assert updates == [1, 2, 3, 4]
    assert len(summary["generations"]) == 1
    # Only the last few checkpoints are kept.
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) <= config.keep_checkpoints


def _generation_ready_to_evaluate(run_dir) -> League:
    """Generations 1 and 2 are finished, and 2 is the reigning best."""
    league = League(tiny_config(run_dir, max_generations=3))
    league.state = {
        "generations": [
            {"generation": 1, "became_best": True, "score_vs_best": None, "scores": {}},
            {"generation": 2, "became_best": True, "score_vs_best": 0.6, "scores": {"gen001": 0.6}},
        ],
        "best": 2,
        "no_improvement": 0,
        "current": {"generation": 3, "phase": "evaluating"},
    }
    meta = league.gen_dir(3) / "meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"updates": 2, "games": 8, "training_hours": 0.0, "pool_score": 0.5}),
                    encoding="utf-8")
    return league


def _scripted_tournament(scores_by_name):
    def run(specs, games, **kwargs):
        from bot.agents import spec_name
        names = [spec_name(spec) for spec in specs]
        table = [[None] * len(names) for _ in names]
        for index, name in enumerate(names):
            if index:
                table[0][index] = scores_by_name[name]
        return {"names": names, "table": table}
    return run


def test_old_stop_keys_are_ignored():
    config = LeagueConfig.from_dict({
        "min_updates": 10,
        "max_updates": 20,
        "patience": 5,
        "improvement_margin": 0.01,
        "updates_per_gen": 7,
    })
    assert config.updates_per_gen == 7


def test_old_entropy_coef_float_is_one_stage():
    config = LeagueConfig.from_dict({"ppo": {"entropy_coef": 0.02}})
    assert config.ppo.entropy_coef == [0.02]
    assert config.ppo.entropy_at(0, 500) == 0.02
    assert config.ppo.entropy_at(499, 500) == 0.02
    staged = PPOConfig(entropy_coef=[0.5, 0.25, 0.1])
    assert [staged.entropy_at(update, 500) for update in (0, 499, 500, 999, 1000, 1499)] == [
        0.5, 0.5, 0.25, 0.25, 0.1, 0.1,
    ]
    with pytest.raises(ValueError):
        PPOConfig(entropy_coef=[])


def test_each_entropy_stage_lasts_updates_per_gen(tmp_path, monkeypatch):
    real_update = league_module.ppo_update
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["entropy_coef"] if "entropy_coef" in kwargs else args[5])
        return real_update(*args, **kwargs)

    monkeypatch.setattr(league_module, "ppo_update", capture)
    ppo = PPOConfig(
        games_per_update=8, games_per_job=4, minibatch=64, hidden=32,
        entropy_coef=[0.5, 0.1],
    )
    summary = run_league(tiny_config(tmp_path, updates_per_gen=1, max_generations=1, ppo=ppo))
    assert seen == [0.5, 0.1]
    assert [int(row["update"]) for row in read_csv(tmp_path / "logs" / "training.csv")] == [1, 2]
    assert len(summary["generations"]) == 1


def test_new_best_must_beat_every_earlier_generation(tmp_path, monkeypatch):
    lost = _generation_ready_to_evaluate(tmp_path / "lost")
    monkeypatch.setattr(league_module, "run_tournament", _scripted_tournament({
        "gen001": 0.4,
        "gen002": 0.6,
        "ismcts-20it": 0.9,
    }))
    lost.evaluate_generation(3)
    assert lost.state["best"] == 2
    last = lost.state["generations"][-1]
    assert last["became_best"] is False
    assert last["score_vs_best"] == 0.6
    assert lost.state["no_improvement"] == 1

    # Beating every earlier generation crowns it, even if the eval opponent wins.
    won = _generation_ready_to_evaluate(tmp_path / "won")
    monkeypatch.setattr(league_module, "run_tournament", _scripted_tournament({
        "gen001": 0.7,
        "gen002": 0.6,
        "ismcts-20it": 0.1,
    }))
    won.evaluate_generation(3)
    assert won.state["best"] == 3
    assert won.state["generations"][-1]["became_best"] is True

    # A tie with an older generation is not enough.
    tied = _generation_ready_to_evaluate(tmp_path / "tied")
    monkeypatch.setattr(league_module, "run_tournament", _scripted_tournament({
        "gen001": 0.5,
        "gen002": 0.8,
        "ismcts-20it": 0.9,
    }))
    tied.evaluate_generation(3)
    assert tied.state["best"] == 2
    assert tied.state["generations"][-1]["became_best"] is False


def test_old_log_header_is_rewritten(tmp_path):
    from bot.league import TRAINING_FIELDS, _append_csv

    path = tmp_path / "training.csv"
    path.write_text(
        "time,generation,update,pool_score,best_pool_score\n2020,1,1,0.5,0.6\n",
        encoding="utf-8",
    )
    _append_csv(path, TRAINING_FIELDS, {
        "generation": 1,
        "update": 2,
        "reference": "baseline",
        "reference_score": 0.25,
        "best_reference_score": 0.25,
    })
    rows = read_csv(path)
    assert list(rows[0]) == TRAINING_FIELDS
    assert "pool_score" not in rows[0]
    assert rows[0]["reference"] == ""
    assert rows[0]["reference_score"] == ""
    assert rows[0]["update"] == "1"
    assert rows[1]["reference"] == "baseline"
    assert rows[1]["reference_score"] == "0.25"
