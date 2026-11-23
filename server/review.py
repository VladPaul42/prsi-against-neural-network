"""Looking back at saved games (host only).

``game_frames`` replays a record and returns every position with both hands
(the game is over; nothing is secret any more). ``analyse_game`` runs in a
worker process: for every move it gives the bot exactly the observation the
player to move had at that moment, and stores the bot's evaluation next to
the move that was really played. It also stores a win-chance bar for the
bottom player (seat 0): both real hands, the remaining deck shuffled again,
both seats playing without seeing the other hand.

Results are cached per game and network in ``games/analysis/`` so a game is
analysed only once per network and search rule. An older file that has the
move list but no bar is filled in by ``fill_eval`` without searching again.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from prsi.actions import DONT_PLAY, describe_action
from prsi.game import Game
from prsi.record import load_record, opening_hand_of

# Imagined games the saved-game review gives each legal card.
REVIEW_LINE_LIMIT = 10_000
# Shuffles of the remaining deck for one position. Greedy play is certain
# once nothing is left to draw, so that position uses a single play-out.
EVAL_SAMPLES = 1000
# How often a running bar publishes while finishes are still being played.
_BAR_REPORT_EVERY = 0.25
# Play-outs that run this long past the position count as a draw.
EVAL_HORIZON = 300
SAFE_NAME = re.compile(r"^[\w.-]+\.json$")


def review_spec(bot_spec: dict) -> dict:
    """Same bot as at the table, but with a fixed per-line budget so results repeat."""
    spec = {
        key: value
        for key, value in bot_spec.items()
        if key not in ("seconds", "review_line_limit", "bar_samples")
    }
    spec["line_limit"] = int(bot_spec.get("review_line_limit", REVIEW_LINE_LIMIT))
    spec["bar_samples"] = int(bot_spec.get("bar_samples", EVAL_SAMPLES))
    spec["seed"] = 42
    return spec


def spec_version(bot_spec: dict) -> str:
    """Names the cache: the network folder (e.g. ``gen_003``), or the engine without one.

    The ``-line-one`` suffix is part of the name so an analysis saved by an
    older search is left unused and the next review runs again. That covers a
    shared total, the biased search before that, and a search that skipped a
    turn with one legal move. A line limit or bar budget other than the
    default is added too, so that analysis is run again and a default run
    still finds the caches already saved.
    """
    label = bot_spec.get("label") or bot_spec["kind"]
    name = re.sub(r"[^\w.-]+", "-", label)
    version = f"{name}-line-one"
    line_limit = int(bot_spec.get("review_line_limit", REVIEW_LINE_LIMIT))
    samples = int(bot_spec.get("bar_samples", EVAL_SAMPLES))
    if line_limit != REVIEW_LINE_LIMIT or samples != EVAL_SAMPLES:
        version = f"{version}-{line_limit}-{samples}"
    return version


def safe_game_path(games_dir: Path, name: str) -> Path | None:
    if not isinstance(name, str) or not SAFE_NAME.match(name):
        return None
    path = Path(games_dir) / name
    return path if path.is_file() else None


def analysis_paths(games_dir: Path, name: str, version: str) -> tuple[Path, Path]:
    folder = Path(games_dir) / "analysis"
    stem = Path(name).stem
    return folder / f"{stem}__{version}.json", folder / f"{stem}__{version}.progress.json"


def list_games(games_dir: Path) -> list[dict]:
    games = []
    for path in sorted(Path(games_dir).glob("*.json"), reverse=True):
        try:
            record = load_record(path)
        except (OSError, ValueError, KeyError):
            continue
        result = record.get("result") or {}
        games.append(
            {
                "file": path.name,
                "names": record.get("names"),
                "seats": record.get("seats"),
                "moves": len(record.get("actions", [])),
                "winner": result.get("winner"),
                "reason": result.get("reason"),
                "mode": record.get("mode"),
            }
        )
    return games


def _card_dicts(cards) -> list[dict]:
    return [card.to_dict() for card in sorted(cards, key=lambda c: c.id)]


def _label(action: int, game: Game) -> str:
    if action == DONT_PLAY:
        info = game._dont_play_info()
        return info["label"]
    text = describe_action(action)
    return text[0].upper() + text[1:]


def game_frames(path: Path) -> dict:
    record = load_record(path)
    game = Game(record["seed"], record["starting_player"], opening_hand_of(record))
    frames = []
    # How many public log lines exist after the action this frame is showing.
    # The frame itself is the position before that action.
    log_at = []
    for index, action in enumerate(record["actions"]):
        frames.append(_frame(game, index, action))
        game.step(action)
        log_at.append(len(game.log))
    # A resignation or a reset is stored on the record, not as a move.
    result = record.get("result")
    if result and (not game.log or game.log[-1].get("kind") != "game_over"):
        game.log.append(
            {"kind": "game_over", "winner": result.get("winner"), "reason": result.get("reason")}
        )
    frames.append(_frame(game, len(record["actions"]), None))
    log_at.append(len(game.log))
    return {"record": record, "frames": frames, "log": list(game.log), "log_at": log_at}


def _frame(game: Game, index: int, action: int | None) -> dict:
    frame = {
        "index": index,
        "hands": [_card_dicts(game.hands[0]), _card_dicts(game.hands[1])],
        "top_card": game.discard[-1].to_dict(),
        "under_cards": [card.to_dict() for card in game.discard[-3:-1]],
        "active_suit": game.active_suit.key,
        "active_suit_name": game.active_suit.label,
        "named_by_over": game.suit_named,
        "penalty_cards": game.pending_sevens * 2,
        "ace_active": game.ace_active,
        "draw_pile_size": len(game.draw),
        "current": None if game.is_over else game.current,
        "result": game.result,
    }
    if action is not None:
        frame["action"] = action
        frame["label"] = _label(action, game)
        frame["legal_count"] = len(game.legal_actions())
    return frame


def read_status(games_dir: Path, name: str, version: str) -> dict:
    result_path, progress_path = analysis_paths(games_dir, name, version)
    result = _read_json(result_path) if result_path.exists() else None
    # A finished move list can still be growing a bar. The progress file wins
    # until ``eval`` is stored, so the page keeps polling.
    if result is None or "eval" not in result:
        if progress_path.exists():
            progress = _read_json(progress_path) or {}
            status = {"status": progress.get("status", "running"), **progress}
            if result is not None:
                status["result"] = result
            return status
    if result is not None:
        return {"status": "done", "result": result}
    return {"status": "missing"}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def _rollout_agent(spec: dict):
    """Who plays inside a bar sample.

    The review search tries a hundred thousand deals per move. Doing that
    inside every sample would make the bar take hours, so a network plays
    its most likely move. With no network, both seats play a random legal move.
    """
    if spec.get("path"):
        from bot.net import PolicyAgent, load_numpy_net

        return PolicyAgent(load_numpy_net(spec["path"]), greedy=True, seed=spec.get("seed", 0))
    from bot.agents import RandomAgent

    return RandomAgent(spec.get("seed", 0))


def _public_trackers(game: Game):
    """What each seat knows at ``game``, with the other hand still hidden."""
    from bot.fastgame import cards_mask
    from bot.tracker import Tracker, tracker_from_observation

    log = game.log
    if log and log[0].get("kind") == "opening":
        return (
            tracker_from_observation(game.observation(0)),
            tracker_from_observation(game.observation(1)),
        )
    # A position built for a test has no public log. Before any reshuffle,
    # every unseen card is simply in the pool.
    discard = [card.id for card in game.discard]
    built = []
    for seat in (0, 1):
        tracker = Tracker.from_discard(seat, discard, game.active_suit.value)
        tracker.passes = game.consecutive_passes
        tracker.n_actions = len(game.actions)
        tracker.sync(
            cards_mask(card.id for card in game.hands[seat]),
            len(game.hands[1 - seat]),
            len(game.draw),
        )
        built.append(tracker)
    return tuple(built)


def _finished_chance(game: Game) -> float:
    winner = game.result["winner"]
    if winner is None:
        return 0.5
    return 1.0 if winner == 0 else 0.0


def _play_out(game: Game, agent, rng: random.Random, trackers) -> float:
    """One finish from ``game``. The draw pile is shuffled; future reshuffles use a fresh generator."""
    from bot.agents import PublicState
    from bot.fastgame import FastGame, cards_mask, shuffle_ids

    draw = [card.id for card in game.draw]
    shuffle_ids(rng, draw)
    fast = FastGame.from_state(
        [cards_mask(card.id for card in game.hands[0]), cards_mask(card.id for card in game.hands[1])],
        draw,
        [card.id for card in game.discard],
        active_suit=game.active_suit.value,
        pending=game.pending_sevens,
        ace=game.ace_active,
        suit_named=game.suit_named,
        current=game.current,
        passes=game.consecutive_passes,
        rng=random.Random(rng.randrange(2**62)),
        n_actions=len(game.actions),
        max_actions=len(game.actions) + EVAL_HORIZON,
        record_events=True,
    )
    views = (trackers[0].clone(), trackers[1].clone())
    while not fast.over:
        seat = fast.current
        state = PublicState.from_fast(fast, views[seat])
        action = agent.act(state)
        if action not in state.legal:
            raise RuntimeError(f"Rollout chose illegal action {action}")
        fast.step(action)
    if fast.winner is None:
        return 0.5
    return 1.0 if fast.winner == 0 else 0.0


def bar_snapshot(game: Game) -> dict:
    """A picklable copy of a position, including the public log the tracker needs."""
    return {
        "hands": [[card.id for card in hand] for hand in game.hands],
        "draw": [card.id for card in game.draw],
        "discard": [card.id for card in game.discard],
        "active_suit": game.active_suit.value,
        "pending": game.pending_sevens,
        "ace": game.ace_active,
        "suit_named": game.suit_named,
        "current": game.current if game.current in (0, 1) else 0,
        "passes": game.consecutive_passes,
        "log": [dict(event) for event in game.log],
        "drawn": [list(game.drawn[0]), list(game.drawn[1])],
        "over": game.is_over,
        "winner": None if game.result is None else game.result["winner"],
    }


def _chance_for_winner(winner: int | None) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == 0 else 0.0


def _bar_kind(spec: dict | None) -> str:
    return "network" if spec and spec.get("path") else "random"


def _game_from_snapshot(snapshot: dict) -> Game:
    from prsi.cards import Card, Rank, Suit

    def cards(ids: list[int]) -> list[Card]:
        return [Card(Suit(card_id // 8), Rank(card_id % 8)) for card_id in ids]

    game = Game.from_position(
        hands=[cards(snapshot["hands"][0]), cards(snapshot["hands"][1])],
        discard=cards(snapshot["discard"]),
        draw=cards(snapshot["draw"]),
        active_suit=Suit(snapshot["active_suit"]),
        pending_sevens=snapshot["pending"],
        ace_active=snapshot["ace"],
        suit_named=snapshot["suit_named"],
        current_player=snapshot["current"],
        consecutive_passes=snapshot["passes"],
        seed=0,
    )
    game.log = list(snapshot.get("log") or [])
    drawn = snapshot.get("drawn") or [[], []]
    game.drawn = [list(drawn[0]), list(drawn[1])]
    if snapshot.get("over"):
        game.result = {"winner": snapshot.get("winner"), "reason": None}
    return game


_LIVE_AGENT = None
_LIVE_AGENT_KEY = None


def score_snapshot(
    snapshot: dict,
    spec: dict | None,
    seed: int,
    progress=None,
    stop=None,
    token=None,
) -> dict | None:
    """Score one position. Safe to call in a worker process.

    ``progress`` is a queue. Each report is ``(token, chance, played, cap)``.
    ``stop`` (an event with ``is_set``) ends the run between finishes.
    A stop before any finish returns ``None``.
    """
    kind = _bar_kind(spec)
    if snapshot.get("over"):
        chance = _chance_for_winner(snapshot.get("winner"))
        _put_bar(progress, token, chance, 1, 1)
        return {"chance": chance, "kind": kind, "samples": 1, "cap": 1}
    global _LIVE_AGENT, _LIVE_AGENT_KEY
    key = spec.get("path") if spec else None
    if _LIVE_AGENT is None or _LIVE_AGENT_KEY != key:
        _LIVE_AGENT = _rollout_agent(spec or {})
        _LIVE_AGENT_KEY = key
    game = _game_from_snapshot(snapshot)
    frame_seed = int(seed) & 0xFFFFFFFFFFFF
    _LIVE_AGENT.new_game(frame_seed)
    heard: dict = {}

    def report(chance: float, played: int, cap: int) -> None:
        heard["samples"] = played
        heard["cap"] = cap
        _put_bar(progress, token, chance, played, cap)

    chance = seat0_chance(
        game,
        _LIVE_AGENT,
        random.Random(frame_seed),
        _bar_samples(spec),
        progress=report,
        stop=stop,
    )
    if chance is None:
        return None
    samples = int(heard.get("samples", _bar_samples(spec)))
    cap = int(heard.get("cap", samples))
    return {"chance": chance, "kind": kind, "samples": samples, "cap": cap}


def _put_bar(progress, token, chance: float, played: int, cap: int) -> None:
    """One running average for the live bar. A missing queue scores quietly."""
    if progress is None:
        return
    try:
        progress.put_nowait((token, chance, played, cap))
    except Exception:
        pass


def _bar_samples(spec: dict | None) -> int:
    """Finishes to play from one position. A missing count keeps the default."""
    if spec is None or spec.get("bar_samples") is None:
        return EVAL_SAMPLES
    return int(spec["bar_samples"])


def seat0_chance(
    game: Game,
    agent,
    rng: random.Random,
    samples: int,
    progress=None,
    stop=None,
) -> float | None:
    """Chance seat 0 wins from this position, seeing both hands but not the deck order.

    ``progress(chance, played, cap)`` receives the running mean: after the first
    finish, again whenever ``_BAR_REPORT_EVERY`` seconds have passed, and always
    after the last finish. ``stop`` (an event with ``is_set``) ends the run
    between finishes. The return is the mean of the finishes actually played,
    or ``None`` when stop fires before any finish.
    """
    if game.is_over:
        chance = _finished_chance(game)
        if progress is not None:
            progress(chance, 1, 1)
        return chance
    trackers = _public_trackers(game)
    count = samples
    if getattr(agent, "greedy", False) and game.available_to_draw() == 0:
        count = 1
    total = 0.0
    played = 0
    next_report = 0.0
    for _ in range(count):
        if stop is not None and stop.is_set():
            break
        total += _play_out(game, agent, rng, trackers)
        played += 1
        if progress is None:
            continue
        now = time.perf_counter()
        if played == 1 or played == count or now >= next_report:
            progress(round(total / played, 4), played, count)
            next_report = now + _BAR_REPORT_EVERY
    if played == 0:
        return None
    return round(total / played, 4)


def position_chances(record: dict, spec: dict, report) -> list[float]:
    """One seat-0 chance per frame, including the position after the last move."""
    agent = _rollout_agent(spec)
    game = Game(record["seed"], record["starting_player"], opening_hand_of(record))
    actions = record["actions"]
    total = len(actions) + 1
    samples = _bar_samples(spec)
    seed = int(spec.get("seed", 0))
    values = []
    for index in range(total):
        report(index, total)
        frame_seed = (seed + index) & 0xFFFFFFFFFFFF
        agent.new_game(frame_seed)
        values.append(seat0_chance(game, agent, random.Random(frame_seed), samples))
        if index < len(actions):
            game.step(actions[index])
    report(total, total)
    return values


def _score_bar(record: dict, spec: dict, report) -> dict:
    return {
        "eval": position_chances(record, spec, report),
        "eval_samples": _bar_samples(spec),
        "eval_kind": "network" if spec.get("path") else "random",
    }


def analyse_game(games_dir: str, name: str, bot_spec: dict) -> dict:
    """Runs in a worker process. Writes progress, then the cached result."""
    from bot.agents import PublicState, make_agent

    games_dir = Path(games_dir)
    spec = review_spec(bot_spec)
    version = spec_version(bot_spec)
    result_path, progress_path = analysis_paths(games_dir, name, version)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = load_record(games_dir / name)
    actions = record["actions"]
    total = len(actions)

    def progress(done: int, total_steps: int, phase: str, status: str = "running") -> None:
        _write_json(progress_path, {"status": status, "done": done, "total": total_steps, "phase": phase})

    progress(0, total, "moves")
    agent = make_agent(spec)
    game = Game(record["seed"], record["starting_player"], opening_hand_of(record))
    moves = []
    started = time.time()
    try:
        for index, action in enumerate(actions):
            mover = game.current
            view = game.observation(mover)
            legal = view["legal_actions"]
            entry = {"index": index, "player": mover, "actual": action, "actual_label": _label(action, game)}
            # One legal move is still searched, so the review has a win chance for it.
            agent.new_game(spec["seed"] + index)
            state = PublicState.from_observation(view)
            stats = agent.analyse(state, iterations=int(spec["line_limit"]) * len(legal))
            options = []
            for move in stats["moves"]:
                options.append(
                    {
                        "action": move["action"],
                        "label": _label(move["action"], game),
                        "win_chance": move["win_chance"],
                        "visits": move["visits"],
                        "prior": move.get("prior"),
                    }
                )
            options.sort(
                key=lambda m: (-(m["win_chance"] if m["win_chance"] is not None else -1.0), -(m["visits"] or 0))
            )
            by_action = {m["action"]: m for m in options}
            best = by_action[stats["action"]]
            actual = by_action.get(action, {})
            loss = None
            if best["win_chance"] is not None and actual.get("win_chance") is not None:
                loss = best["win_chance"] - actual["win_chance"]
            elif best["win_chance"] is not None and action != stats["action"]:
                # The search never tried the move: it did not look worth trying.
                loss = best["win_chance"]
            entry.update(
                {
                    "forced": len(legal) == 1,
                    "bot_action": stats["action"],
                    "bot_label": best["label"],
                    "win_chance": stats["win_chance"],
                    "actual_win_chance": actual.get("win_chance"),
                    "loss": loss,
                    "iterations": stats["iterations"],
                    "moves": options,
                }
            )
            moves.append(entry)
            game.step(action)
            progress(index + 1, total, "moves")
    except Exception as exc:  # report, don't leave the page waiting forever
        progress(len(moves), total, "moves", status=f"failed: {exc}")
        raise
    result = {
        "file": name,
        "version": version,
        "spec": spec,
        "names": record.get("names"),
        "seats": record.get("seats"),
        "seconds": round(time.time() - started, 1),
        "moves": moves,
    }
    try:
        result.update(_score_bar(record, spec, lambda done, total_frames: progress(done, total_frames, "bar")))
    except Exception as exc:
        # Keep the move list. The next request can score the bar on its own.
        _write_json(result_path, result)
        progress(0, len(actions) + 1, "bar", status=f"failed: {exc}")
        raise
    result["seconds"] = round(time.time() - started, 1)
    _write_json(result_path, result)
    progress_path.unlink(missing_ok=True)
    return {"status": "done"}


def fill_eval(games_dir: str, name: str, bot_spec: dict) -> dict:
    """Add the bar to a cached analysis that already has its move list."""
    games_dir = Path(games_dir)
    spec = review_spec(bot_spec)
    version = spec_version(bot_spec)
    result_path, progress_path = analysis_paths(games_dir, name, version)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "eval" in result:
        progress_path.unlink(missing_ok=True)
        return {"status": "done"}
    record = load_record(games_dir / name)
    frames = len(record["actions"]) + 1

    def progress(done: int, total_steps: int, status: str = "running") -> None:
        _write_json(progress_path, {"status": status, "done": done, "total": total_steps, "phase": "bar"})

    progress(0, frames)
    try:
        result.update(_score_bar(record, spec, progress))
    except Exception as exc:
        progress(0, frames, status=f"failed: {exc}")
        raise
    _write_json(result_path, result)
    progress_path.unlink(missing_ok=True)
    return {"status": "done"}
