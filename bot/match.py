"""Play bots against each other, fairly.

Prší has a lot of luck, so every deal is played twice with the bots swapping
seats ("mirrored deals"). 500 games = 250 deals. A draw counts as half a win.

Results go to ``runs/matches/``: one CSV per match with every game, and one
row per match in ``summary.csv``. Tournaments reuse rows from ``summary.csv``
when the same two bots already played the same deals.

    python -m bot.match random random --games 200
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import re
import time
from pathlib import Path

from bot.agents import Agent, PublicState, make_agent, spec_name
from bot.fastgame import FastGame
from bot.tracker import Tracker

ROOT = Path(__file__).resolve().parent.parent
MATCH_DIR = ROOT / "runs" / "matches"
DEFAULT_SEED = 42
DEFAULT_MAX_ACTIONS = 1000

SUMMARY_FIELDS = [
    "time",
    "a",
    "b",
    "games",
    "a_wins",
    "b_wins",
    "draws",
    "a_score",
    "avg_length",
    "seconds",
    "seed",
    "max_actions",
    "a_spec",
    "b_spec",
]


def play_game(
    agents: list[Agent] | tuple[Agent, Agent],
    seed: int,
    starter: int,
    max_actions: int | None = DEFAULT_MAX_ACTIONS,
) -> FastGame:
    """One game, ``agents[0]`` in seat 0. Returns the finished game."""
    game = FastGame(seed, starter, max_actions=max_actions, record_events=True)
    opening = game.discard[0]
    trackers = (Tracker(0, opening), Tracker(1, opening))
    for seat, agent in enumerate(agents):
        agent.new_game((seed * 2 + seat) & 0xFFFFFFFFFFFF)
    while not game.over:
        seat = game.current
        state = PublicState.from_fast(game, trackers[seat])
        action = agents[seat].act(state)
        if action not in state.legal:
            raise RuntimeError(f"{agents[seat].name} chose illegal action {action}")
        game.step(action)
    return game


def make_deals(count: int, seed: int = DEFAULT_SEED) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(2**62), rng.randrange(2)) for _ in range(count)]


_AGENTS: dict[str, Agent] = {}


def _cached_agent(spec: dict) -> Agent:
    key = json.dumps(spec, sort_keys=True)
    agent = _AGENTS.get(key)
    if agent is None:
        agent = _AGENTS[key] = make_agent(spec)
    return agent


def _play_deals(spec_a: dict, spec_b: dict, deals: list, max_actions) -> list[dict]:
    a = _cached_agent(spec_a)
    b = _cached_agent({**spec_b, "_instance": "b"}) if spec_a == spec_b else _cached_agent(spec_b)
    rows = []
    for deal_seed, starter in deals:
        for a_seat in (0, 1):
            agents = (a, b) if a_seat == 0 else (b, a)
            start = time.perf_counter()
            game = play_game(agents, deal_seed, starter, max_actions)
            if game.winner is None:
                outcome = 0.5
            else:
                outcome = 1.0 if game.winner == a_seat else 0.0
            rows.append(
                {
                    "seed": deal_seed,
                    "starter": starter,
                    "a_seat": a_seat,
                    "winner": "" if game.winner is None else game.winner,
                    "reason": game.reason,
                    "length": game.n_actions,
                    "a_score": outcome,
                    "seconds": round(time.perf_counter() - start, 3),
                }
            )
    return rows


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_match(
    spec_a: dict,
    spec_b: dict,
    games: int = 500,
    *,
    workers: int = 4,
    seed: int = DEFAULT_SEED,
    max_actions: int | None = DEFAULT_MAX_ACTIONS,
    out_dir: Path | str | None = MATCH_DIR,
    progress: bool = True,
    chunk: int = 5,
) -> dict:
    """Play ``games`` games (an even number) of ``spec_a`` against ``spec_b``."""
    if games < 2 or games % 2:
        raise ValueError("games must be an even number: every deal is played twice")
    deals = make_deals(games // 2, seed)
    name_a, name_b = spec_name(spec_a), spec_name(spec_b)
    start = time.perf_counter()
    rows: list[dict] = []
    if workers <= 1:
        for part in _chunks(deals, chunk):
            rows.extend(_play_deals(spec_a, spec_b, part, max_actions))
            if progress:
                _print_progress(name_a, name_b, rows, games, start)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            jobs = [(spec_a, spec_b, part, max_actions) for part in _chunks(deals, chunk)]
            for part_rows in pool.imap_unordered(_play_deals_star, jobs):
                rows.extend(part_rows)
                if progress:
                    _print_progress(name_a, name_b, rows, games, start)
    seconds = time.perf_counter() - start
    summary = summarise(rows)
    summary.update(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "a": name_a,
            "b": name_b,
            "seconds": round(seconds, 1),
            "seed": seed,
            "max_actions": max_actions,
            "a_spec": json.dumps(spec_a, sort_keys=True),
            "b_spec": json.dumps(spec_b, sort_keys=True),
        }
    )
    if out_dir is not None:
        _save(Path(out_dir), summary, rows)
    if progress:
        print(
            f"{name_a} vs {name_b}: {summary['a_score']:.1%} for {name_a} "
            f"({summary['a_wins']}-{summary['b_wins']}-{summary['draws']}) in {seconds:.0f}s",
            flush=True,
        )
    return summary


def _play_deals_star(job):
    return _play_deals(*job)


def _print_progress(name_a, name_b, rows, games, start) -> None:
    score = sum(row["a_score"] for row in rows) / len(rows)
    print(
        f"\r  {name_a} vs {name_b}: {len(rows)}/{games} games, {score:.1%} for {name_a}, "
        f"{time.perf_counter() - start:.0f}s   ",
        end="" if len(rows) < games else "\n",
        flush=True,
    )


def summarise(rows: list[dict]) -> dict:
    games = len(rows)
    a_wins = sum(1 for row in rows if row["a_score"] == 1.0)
    b_wins = sum(1 for row in rows if row["a_score"] == 0.0)
    draws = games - a_wins - b_wins
    return {
        "games": games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score": (a_wins + 0.5 * draws) / games if games else 0.0,
        "avg_length": round(sum(row["length"] for row in rows) / games, 1) if games else 0.0,
    }


def _save(out_dir: Path, summary: dict, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{summary['a']}_vs_{summary['b']}")
    with open(out_dir / f"{stamp}_{safe}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = out_dir / "summary.csv"
    new = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(summary)


def load_summaries(out_dir: Path | str = MATCH_DIR) -> list[dict]:
    path = Path(out_dir) / "summary.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_result(spec_a: dict, spec_b: dict, games: int, seed: int, max_actions, out_dir=MATCH_DIR):
    """The score of ``spec_a`` against ``spec_b`` from earlier matches, or None."""
    key_a = json.dumps(spec_a, sort_keys=True)
    key_b = json.dumps(spec_b, sort_keys=True)
    for row in reversed(load_summaries(out_dir)):
        if int(row["games"]) != games or int(row["seed"]) != seed:
            continue
        if str(row["max_actions"]) != str(max_actions):
            continue
        if row["a_spec"] == key_a and row["b_spec"] == key_b:
            return float(row["a_score"])
        if row["a_spec"] == key_b and row["b_spec"] == key_a:
            return 1.0 - float(row["a_score"])
    return None


def run_tournament(
    specs: list[dict],
    games: int = 500,
    *,
    pairs: list[tuple[int, int]] | None = None,
    reuse: bool = True,
    workers: int = 4,
    seed: int = DEFAULT_SEED,
    max_actions: int | None = DEFAULT_MAX_ACTIONS,
    out_dir: Path | str = MATCH_DIR,
    progress: bool = True,
) -> dict:
    """Everyone against everyone (or only ``pairs``). Returns names and a score table.

    ``table[i][j]`` is the score of bot ``i`` against bot ``j`` (None if not played).
    """
    names = [spec_name(spec) for spec in specs]
    n = len(specs)
    table: list[list[float | None]] = [[None] * n for _ in range(n)]
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        score = find_result(specs[i], specs[j], games, seed, max_actions, out_dir) if reuse else None
        if score is None:
            summary = run_match(
                specs[i], specs[j], games,
                workers=workers, seed=seed, max_actions=max_actions,
                out_dir=out_dir, progress=progress,
            )
            score = summary["a_score"]
        elif progress:
            print(f"{names[i]} vs {names[j]}: {score:.1%} (from earlier results)", flush=True)
        table[i][j] = score
        table[j][i] = 1.0 - score
    return {"names": names, "table": table}


def format_table(result: dict) -> str:
    names = result["names"]
    width = max(8, *(len(name) for name in names)) + 2
    lines = ["".ljust(width) + "".join(name[: width - 1].rjust(width) for name in names) + "   average"]
    for i, name in enumerate(names):
        cells = []
        scores = []
        for j in range(len(names)):
            value = result["table"][i][j]
            if i == j or value is None:
                cells.append("-".rjust(width))
            else:
                cells.append(f"{value:.1%}".rjust(width))
                scores.append(value)
        average = f"{sum(scores) / len(scores):.1%}" if scores else "-"
        lines.append(name.ljust(width) + "".join(cells) + average.rjust(10))
    return "\n".join(lines)


def _budget(text: str, default: dict) -> dict:
    if not text:
        return default
    if text.endswith("s"):
        return {"seconds": float(text[:-1])}
    return {"iterations": int(text)}


def parse_spec(text: str) -> dict:
    """``random``, ``ismcts:2000`` (imagined games), ``ismcts:1.0s``, ``ppo:PATH``,
    ``guided:PATH`` / ``guided:PATH:1.0s``, or a JSON dict."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    if text == "random":
        return {"kind": "random"}
    kind, _, rest = text.partition(":")
    if kind == "ismcts":
        return {"kind": "ismcts", **_budget(rest, {"iterations": 1000})}
    if kind == "ppo" and rest:
        return {"kind": "ppo", "path": rest, "label": Path(rest).name or rest}
    if kind == "guided" and rest:
        path, budget = rest, ""
        head, _, tail = rest.rpartition(":")
        if head and re.fullmatch(r"\d+(\.\d+)?s?", tail):
            path, budget = head, tail
        return {"kind": "guided", "path": path, "label": Path(path).name or path, **_budget(budget, {"seconds": 1.0})}
    raise ValueError(f"Unknown bot: {text}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play two bots against each other on mirrored deals.")
    parser.add_argument("a", help="random, ismcts:2000, ismcts:1.0s, or a JSON spec")
    parser.add_argument("b")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    run_match(parse_spec(args.a), parse_spec(args.b), args.games, workers=args.workers, seed=args.seed)


if __name__ == "__main__":
    main()
