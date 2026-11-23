"""The PPO league: generations of networks trained against a baseline and their predecessors.

- Generation 1 trains against ``train_opponent``. Every later generation trains
  against that same baseline plus all earlier generations, never against itself.
  Opponents the learner loses to more often are picked more often.
- A generation trains for ``updates_per_gen`` updates at each entropy
  coefficient in ``ppo.entropy_coef``, then stops.
- Then it plays every earlier generation and ``eval_opponent`` on mirrored
  deals (``eval_games`` each; old results are reused). It becomes the new best
  only if it scores more than 50% against every earlier generation. Generation
  1 is the best by default. The eval opponent is not a generation.

The published run in ``runs/league`` sets both opponents to random play.

- The league stops after ``max_generations``, or after
  ``stop_after_no_improvement`` generations in a row without a new best.

Everything lives in ``run_dir``:

    league_state.json         generations, the best one, what is in progress
    config.json               the settings used last
    gen_001/                  model.pt, weights.npz, meta.json
    checkpoints/              the last few checkpoints (every 15 minutes)
    logs/training.csv         one row per update
    logs/opponents.csv        score, pick weight, and games this update, per opponent
    logs/tournament.csv       every evaluation result
    logs/generations.csv      one row per finished generation
    matches/                  match files from the evaluation tournaments

Stop at any time (interrupt the notebook cell or Ctrl+C). A checkpoint is
written on the way out, and ``run_league`` continues from it next time.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from bot.agents import spec_name
from bot.match import run_tournament
from bot.net import build_model, load_model, model_to_arrays, save_network
from bot.ppo import LeagueConfig, collect, ppo_update

ROOT = Path(__file__).resolve().parent.parent

TRAINING_FIELDS = [
    "time", "generation", "update", "games", "samples",
    "reference", "reference_score", "best_reference_score",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
    "avg_game_length", "games_per_hour", "update_seconds", "elapsed_hours",
]
OPPONENT_FIELDS = [
    "time", "generation", "update", "opponent",
    "games_seen", "games_this_update", "score", "weight",
]
TOURNAMENT_FIELDS = ["time", "generation", "opponent", "games", "score"]
GENERATION_FIELDS = [
    "time", "generation", "updates", "games", "training_hours",
    "reference", "reference_score",
    "previous_best", "score_vs_best", "became_best", "best_after",
]


def run_league(config: LeagueConfig | None = None, generations: int | None = None) -> dict:
    """Train (or continue training). ``generations`` limits how many finish in this call."""
    league = League(config or LeagueConfig())
    return league.run(generations)


class League:
    def __init__(self, config: LeagueConfig):
        self.config = config
        run_dir = Path(config.run_dir)
        self.dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self.state_path = self.dir / "league_state.json"
        self.state = self._load_state()
        (self.dir / "config.json").write_text(config.to_json() + "\n", encoding="utf-8")

    # ------------------------------------------------------------ state

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"generations": [], "best": None, "no_improvement": 0, "current": None}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    def gen_dir(self, generation: int) -> Path:
        return self.dir / f"gen_{generation:03d}"

    def gen_spec(self, generation: int, greedy: bool) -> dict:
        weights = (self.gen_dir(generation) / "weights.npz").relative_to(ROOT)
        return {
            "kind": "ppo",
            "path": weights.as_posix(),
            "greedy": greedy,
            "label": f"gen{generation:03d}",
            "name": f"gen{generation:03d}",
        }

    def finished(self) -> list[int]:
        return [entry["generation"] for entry in self.state["generations"]]

    def should_stop(self) -> str | None:
        cfg = self.config
        if self.state["current"] is not None:
            return None
        if len(self.finished()) >= cfg.max_generations:
            return f"reached {cfg.max_generations} generations"
        if self.state["no_improvement"] >= cfg.stop_after_no_improvement:
            return f"{self.state['no_improvement']} generations in a row without improvement"
        return None

    # ------------------------------------------------------------ main loop

    def run(self, generations: int | None = None) -> dict:
        finished_now = 0
        while True:
            reason = self.should_stop()
            if reason:
                print(f"League finished: {reason}.", flush=True)
                break
            if generations is not None and finished_now >= generations:
                print(f"Stopping after {finished_now} generation(s), as asked.", flush=True)
                break
            current = self.state["current"]
            if current is None:
                generation = len(self.finished()) + 1
                current = {"generation": generation, "phase": "training"}
                self.state["current"] = current
                self._save_state()
            if current["phase"] == "training":
                if not self.train_generation(current["generation"]):
                    break  # interrupted
                current["phase"] = "evaluating"
                self._save_state()
            self.evaluate_generation(current["generation"])
            finished_now += 1
        return self.summary()

    def summary(self) -> dict:
        return {
            "generations": self.state["generations"],
            "best": self.state["best"],
            "no_improvement": self.state["no_improvement"],
            "in_progress": self.state["current"],
        }

    # ------------------------------------------------------------ training

    def opponents(self, generation: int) -> list[dict]:
        specs = [{**self.config.train_opponent, "name": "baseline"}]
        for previous in self.finished():
            if previous < generation:
                specs.append(self.gen_spec(previous, self.config.opponent_greedy))
        return specs

    def _initial_model(self, generation: int):
        cfg = self.config
        source = None
        if cfg.init_from == "best" and self.state["best"] is not None:
            source = self.state["best"]
        elif cfg.init_from == "previous" and self.finished():
            source = self.finished()[-1]
        if source is not None:
            model = load_model(self.gen_dir(source))
            if model.hidden == cfg.ppo.hidden and model.layers == cfg.ppo.layers:
                return model, source
        return build_model(cfg.ppo.hidden, cfg.ppo.layers), None

    def _checkpoints(self, generation: int) -> list[Path]:
        return sorted((self.dir / "checkpoints").glob(f"ckpt_gen{generation:03d}_u*.pt"))

    def train_generation(self, generation: int) -> bool:
        """Train one generation. Returns False if interrupted (a checkpoint is saved)."""
        import torch

        cfg = self.config
        torch.set_num_threads(cfg.torch_threads)
        checkpoints = self._checkpoints(generation)
        opponents = self.opponents(generation)
        names = [spec_name(spec) for spec in opponents]
        reference = self._reference_name(generation, names)
        window = {name: deque(maxlen=cfg.score_window) for name in names}
        if checkpoints:
            data = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
            model = build_model(data["hidden"], data["layers"])
            model.load_state_dict(data["model"])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)
            optimizer.load_state_dict(data["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = cfg.ppo.learning_rate
            progress = data["progress"]
            for name, scores in progress.pop("window").items():
                if name in window:
                    window[name].extend(scores)
            if "best_reference_score" not in progress:
                progress["best_reference_score"] = None
                progress["last_improvement"] = 0
            progress.pop("best_pool_score", None)
            rng = random.Random()
            rng.setstate(_tuple_state(data["rng"]))
            np_rng = np.random.default_rng()
            np_rng.bit_generator.state = data["np_rng"]
            print(f"Generation {generation}: resuming from update {progress['update']} "
                  f"({checkpoints[-1].name}).", flush=True)
        else:
            model, source = self._initial_model(generation)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)
            progress = {
                "update": 0,
                "games": 0,
                "samples": 0,
                "best_reference_score": None,
                "last_improvement": 0,
                "elapsed": 0.0,
                "started_from": source,
            }
            rng = random.Random(cfg.seed * 1000 + generation)
            np_rng = np.random.default_rng(cfg.seed * 1000 + generation)
            print(f"Generation {generation}: training against {', '.join(names)}"
                  + (f", starting from generation {source}." if source else ", starting from scratch."),
                  flush=True)

        def save_checkpoint() -> Path:
            path = self.dir / "checkpoints" / f"ckpt_gen{generation:03d}_u{progress['update']:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "hidden": model.hidden,
                    "layers": model.layers,
                    "progress": {**progress, "window": {k: list(v) for k, v in window.items()}},
                    "rng": rng.getstate(),
                    "np_rng": np_rng.bit_generator.state,
                    "config": cfg.to_dict(),
                },
                path,
            )
            for old in self._checkpoints(generation)[: -cfg.keep_checkpoints]:
                old.unlink(missing_ok=True)
            return path

        last_checkpoint = time.time()
        session_start = time.time()
        elapsed_before = progress["elapsed"]
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(cfg.workers)
        interrupted = False
        try:
            total_updates = cfg.updates_per_gen * len(cfg.ppo.entropy_coef)
            while True:
                update = progress["update"]
                if update >= total_updates:
                    print(f"Generation {generation}: finished {total_updates} updates.", flush=True)
                    break
                entropy_coef = cfg.ppo.entropy_at(update, cfg.updates_per_gen)
                if update % cfg.updates_per_gen == 0:
                    print(
                        f"Generation {generation}: entropy coef {entropy_coef:g} "
                        f"for the next {cfg.updates_per_gen} updates.",
                        flush=True,
                    )
                started = time.time()
                weights = model_to_arrays(model)
                picks = self._pick_opponents(opponents, names, window, rng)
                jobs = [
                    {
                        "weights": weights,
                        "opponent": opponents[index],
                        "opponent_name": names[index],
                        "games": cfg.ppo.games_per_job,
                        "seed": rng.randrange(2**62),
                        "gamma": cfg.ppo.gamma,
                        "gae_lambda": cfg.ppo.gae_lambda,
                        "max_actions": cfg.max_actions,
                    }
                    for index in picks
                ]
                batches = pool.map(collect, jobs, chunksize=1)
                games = sum(len(b["scores"]) for b in batches)
                lengths = [length for b in batches for length in b["lengths"]]
                played = {name: 0 for name in names}
                for batch in batches:
                    played[batch["opponent"]] += len(batch["scores"])
                    window[batch["opponent"]].extend(batch["scores"])
                stats = ppo_update(model, optimizer, batches, cfg.ppo, np_rng, entropy_coef)
                progress["update"] = update = update + 1
                progress["games"] += games
                progress["samples"] += stats["samples"]
                reference_score = self._mean_score(window[reference])
                best_score = progress["best_reference_score"]
                if reference_score is not None and (best_score is None or reference_score > best_score):
                    progress["best_reference_score"] = best_score = reference_score
                    progress["last_improvement"] = update
                now = time.time()
                progress["elapsed"] = elapsed_before + (now - session_start)
                seconds = now - started
                self._log_update(generation, progress, stats, reference, reference_score,
                                 lengths, games, seconds, window, names, played)
                score_text = "n/a" if reference_score is None else f"{reference_score:.1%}"
                best_text = "n/a" if best_score is None else f"{best_score:.1%}"
                print(
                    f"gen {generation} update {update}: score vs {reference} {score_text} "
                    f"(best {best_text} at {progress['last_improvement']}), "
                    f"{games / seconds * 3600:.0f} games/h, entropy {stats['entropy']:.2f} "
                    f"(coef {entropy_coef:g})",
                    flush=True,
                )
                if now - last_checkpoint >= cfg.checkpoint_minutes * 60:
                    save_checkpoint()
                    last_checkpoint = now
        except KeyboardInterrupt:
            interrupted = True
            pool.terminate()
            path = save_checkpoint()
            print(f"\nStopped. Saved {path.name}; run again to continue.", flush=True)
        finally:
            pool.terminate()
            pool.join()
        if interrupted:
            return False
        save_checkpoint()
        meta = {
            "generation": generation,
            "updates": progress["update"],
            "games": progress["games"],
            "samples": progress["samples"],
            "training_hours": round(progress["elapsed"] / 3600, 3),
            "started_from": progress.get("started_from"),
            "reference": reference,
            "reference_score": self._mean_score(window[reference]),
            "opponents": names,
            "scores_vs_opponents": {
                name: self._mean_score(scores) for name, scores in window.items()
            },
            "config": cfg.to_dict(),
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_network(model, self.gen_dir(generation), meta)
        return True

    def _pick_opponents(self, opponents, names, window, rng) -> list[int]:
        """Opponent index for every job. Harder opponents are picked more often."""
        cfg = self.config
        jobs = max(1, cfg.ppo.games_per_update // cfg.ppo.games_per_job)
        weights = []
        for name in names:
            scores = window[name]
            score = sum(scores) / len(scores) if scores else 0.5
            weights.append(max(cfg.pfsp_floor, (1.0 - score) ** cfg.pfsp_power))
        return rng.choices(range(len(opponents)), weights=weights, k=jobs)

    def _reference_name(self, generation: int, names: list[str]) -> str:
        """Reigning best generation, or the baseline opponent when there is none yet."""
        best = self.state.get("best")
        if isinstance(best, int) and best < generation:
            name = f"gen{best:03d}"
            if name in names:
                return name
        if "baseline" in names:
            return "baseline"
        return names[0]

    @staticmethod
    def _mean_score(scores) -> float | None:
        return sum(scores) / len(scores) if scores else None

    def _log_update(self, generation, progress, stats, reference, reference_score,
                    lengths, games, seconds, window, names, played) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        best_score = progress["best_reference_score"]
        _append_csv(
            self.dir / "logs" / "training.csv",
            TRAINING_FIELDS,
            {
                "time": stamp,
                "generation": generation,
                "update": progress["update"],
                "games": progress["games"],
                "samples": stats["samples"],
                "reference": reference,
                "reference_score": _score_cell(reference_score),
                "best_reference_score": _score_cell(best_score),
                "policy_loss": round(stats["policy_loss"], 5),
                "value_loss": round(stats["value_loss"], 5),
                "entropy": round(stats["entropy"], 4),
                "approx_kl": round(stats["approx_kl"], 5),
                "clip_frac": round(stats["clip_frac"], 4),
                "avg_game_length": round(sum(lengths) / max(len(lengths), 1), 1),
                "games_per_hour": round(games / max(seconds, 1e-6) * 3600),
                "update_seconds": round(seconds, 1),
                "elapsed_hours": round(progress["elapsed"] / 3600, 3),
            },
        )
        cfg = self.config
        for name in names:
            scores = window[name]
            score = sum(scores) / len(scores) if scores else None
            weight = max(cfg.pfsp_floor, (1.0 - (score if score is not None else 0.5)) ** cfg.pfsp_power)
            _append_csv(
                self.dir / "logs" / "opponents.csv",
                OPPONENT_FIELDS,
                {
                    "time": stamp,
                    "generation": generation,
                    "update": progress["update"],
                    "opponent": name,
                    "games_seen": len(scores),
                    "games_this_update": played.get(name, 0),
                    "score": "" if score is None else round(score, 4),
                    "weight": round(weight, 4),
                },
            )

    # ------------------------------------------------------------ evaluation

    def evaluation_specs(self, generation: int) -> list[dict]:
        cfg = self.config
        specs = [self.gen_spec(generation, cfg.eval_greedy)]
        for previous in self.finished():
            if previous < generation:
                specs.append(self.gen_spec(previous, cfg.eval_greedy))
        specs.append({**cfg.eval_opponent, "name": spec_name(cfg.eval_opponent)})
        return specs

    def evaluate_generation(self, generation: int) -> None:
        cfg = self.config
        specs = self.evaluation_specs(generation)
        print(f"Generation {generation}: tournament against "
              f"{', '.join(spec_name(s) for s in specs[1:])} ({cfg.eval_games} games each).", flush=True)
        result = run_tournament(
            specs,
            cfg.eval_games,
            pairs=[(0, j) for j in range(1, len(specs))],
            workers=cfg.workers,
            max_actions=1000,
            out_dir=self.dir / "matches",
        )
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        scores = {}
        for j in range(1, len(specs)):
            name = result["names"][j]
            scores[name] = result["table"][0][j]
            _append_csv(
                self.dir / "logs" / "tournament.csv",
                TOURNAMENT_FIELDS,
                {"time": stamp, "generation": generation, "opponent": name,
                 "games": cfg.eval_games, "score": round(scores[name], 4)},
            )
        previous_best = self.state["best"]
        earlier = [f"gen{previous:03d}" for previous in self.finished() if previous < generation]
        missed = [name for name in earlier if scores[name] <= 0.5]
        if previous_best is None:
            became_best = True
            vs_best = None
        else:
            vs_best = scores[f"gen{previous_best:03d}"]
            became_best = not missed
        if became_best:
            self.state["best"] = generation
            self.state["no_improvement"] = 0
        else:
            self.state["no_improvement"] += 1
        meta_path = self.gen_dir(generation) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["evaluation"] = {
            "games_per_opponent": cfg.eval_games,
            "scores": scores,
            "previous_best": previous_best,
            "score_vs_previous_best": vs_best,
            "became_best": became_best,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self.state["generations"].append(
            {"generation": generation, "became_best": became_best, "score_vs_best": vs_best,
             "scores": scores}
        )
        self.state["current"] = None
        self._save_state()
        _append_csv(
            self.dir / "logs" / "generations.csv",
            GENERATION_FIELDS,
            {
                "time": stamp,
                "generation": generation,
                "updates": meta.get("updates"),
                "games": meta.get("games"),
                "training_hours": meta.get("training_hours"),
                "reference": meta.get("reference") or "",
                "reference_score": _score_cell(meta.get("reference_score")),
                "previous_best": previous_best or "",
                "score_vs_best": "" if vs_best is None else round(vs_best, 4),
                "became_best": became_best,
                "best_after": self.state["best"],
            },
        )
        if became_best and vs_best is None:
            verdict = "is the new best."
        elif became_best:
            verdict = "is the new best (beat every earlier generation)."
        else:
            losses = ", ".join(f"{name} {scores[name]:.1%}" for name in missed)
            verdict = f"did not beat every earlier generation (lost to {losses})."
        print(f"Generation {generation} {verdict}", flush=True)

    def full_table(self, workers: int | None = None) -> dict:
        """Everyone against everyone, from saved results (missing pairs are played)."""
        cfg = self.config
        specs = [self.gen_spec(g, cfg.eval_greedy) for g in self.finished()]
        specs.append({**cfg.eval_opponent, "name": spec_name(cfg.eval_opponent)})
        return run_tournament(specs, cfg.eval_games, workers=workers or cfg.workers,
                              max_actions=1000, out_dir=self.dir / "matches")


def best_generation_dir(run_dir: str | Path = "runs/league") -> Path | None:
    """Folder of the best finished generation, or None if there is none yet."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    state_path = run_dir / "league_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    best = state.get("best")
    if best is None:
        return None
    folder = run_dir / f"gen_{best:03d}"
    return folder if (folder / "weights.npz").exists() else None


def _score_cell(score: float | None):
    return "" if score is None else round(score, 4)


def _append_csv(path: Path, fields: list[str], row: dict) -> None:
    """Append one row. An outdated header is rewritten so new columns line up."""
    stale = None
    if path.exists() and path.stat().st_size:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or []) != fields:
                stale = list(reader)
    if stale is not None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for old in stale:
                writer.writerow({key: old.get(key, "") for key in fields})
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _tuple_state(state):
    """``random.Random.setstate`` wants tuples; checkpoints may hold lists."""
    if isinstance(state, list):
        return tuple(_tuple_state(item) for item in state)
    return state
