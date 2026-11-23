"""The bot at the table.

``BotPlayer`` is a ``Player`` like a person in the browser: it gets the
observation for its seat and returns an action number. Thinking happens in a
separate process, so the server keeps answering while the bot searches.
While it thinks it sends progress snapshots (the evaluation panel) a few
times per second; the room passes them only to the host's spectator view.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import logging
import multiprocessing as mp
import queue as queue_module
import secrets
import threading
from pathlib import Path

from server.players import Player

log = logging.getLogger("prsi.server")

ROOT = Path(__file__).resolve().parent.parent

_EXECUTOR: concurrent.futures.ProcessPoolExecutor | None = None
_MANAGER = None


def executor() -> concurrent.futures.ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"))
    return _EXECUTOR


def manager():
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = mp.get_context("spawn").Manager()
    return _MANAGER


def shutdown() -> None:
    global _EXECUTOR, _MANAGER
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None
    if _MANAGER is not None:
        _MANAGER.shutdown()
        _MANAGER = None


def warm_up(spec: dict) -> concurrent.futures.Future:
    """Start the worker processes and load the bot, so the first move is quick."""
    return executor().submit(_load, spec)


# ---------------------------------------------------------------- choosing the bot


def choose_bot(network: str | Path | None = None, seconds: float = 1.0) -> tuple[dict, str]:
    """The strongest bot available, and a sentence for the lobby.

    With a trained network: network-guided ISMCTS. Without one: plain ISMCTS.
    """
    from bot.league import best_generation_dir

    folder = None
    if network:
        folder = Path(network)
        if not folder.is_absolute():
            folder = ROOT / folder
        if folder.is_file():
            folder = folder.parent
        if not (folder / "weights.npz").exists():
            raise SystemExit(f"No trained network in {folder} (weights.npz is missing).")
    else:
        folder = best_generation_dir()
    if folder is None:
        spec = {"kind": "ismcts", "seconds": seconds, "label": "plain ISMCTS"}
        note = "No trained network yet, so the bot is plain ISMCTS (search with random play-outs)."
        return spec, note
    label = folder.name
    spec = {"kind": "guided", "path": str(folder), "seconds": seconds, "label": label}
    note = f"The bot is ISMCTS guided by network {label}."
    return spec, note


# ---------------------------------------------------------------- worker side

_AGENTS: dict = {}


def _load(spec: dict):
    import json

    from bot.agents import make_agent

    key = json.dumps(spec, sort_keys=True)
    agent = _AGENTS.get(key)
    if agent is None:
        agent = _AGENTS[key] = make_agent(spec)
    return True


def _follow(stop):
    """A process-local event that mirrors ``stop``.

    ``stop`` is a manager event, so the search can pass it between processes.
    Checking that proxy on every iteration would be a round trip; one waiting
    thread copies it into a local event instead.
    """
    local = threading.Event()
    if stop.is_set():
        local.set()
        return local

    def watch() -> None:
        try:
            stop.wait()
        except (EOFError, OSError, BrokenPipeError):
            return
        local.set()

    threading.Thread(target=watch, daemon=True).start()
    return local


def think(
    spec: dict,
    view: dict,
    seed: int,
    progress_queue,
    decision: int,
    stop=None,
    iterations: int | None = None,
    seconds: float | None = None,
) -> dict:
    """Runs in a worker process: analyse the position and return the panel.

    ``iterations`` is the imagined-game cap. ``seconds``, when also given, is
    the think clock; the search stops when either limit is reached. The caller
    waits out any time left on the clock. ``stop`` (an event with ``is_set``)
    ends the search early; the panel then describes whatever was found,
    including when no imagined game has finished yet.
    """
    import json

    from bot.agents import PublicState

    _load(spec)
    agent = _AGENTS[json.dumps(spec, sort_keys=True)]
    agent.new_game(seed)
    state = PublicState.from_observation(view)

    def report(stats: dict) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait((decision, make_panel(stats, state, view, final=False)))
        except Exception:
            pass

    kwargs = {}
    if stop is not None:
        kwargs["stop"] = _follow(stop)
    if iterations is not None:
        kwargs["iterations"] = iterations
    if seconds is not None:
        kwargs["seconds"] = seconds
    stats = agent.analyse(state, progress=report, **kwargs)
    return make_panel(stats, state, view, final=True)


def make_panel(stats: dict, state, view: dict, final: bool) -> dict:
    """What the spectator panel shows. Plain data, safe to send as JSON."""
    from bot.fastgame import mask_cards
    from prsi.actions import short_label
    from prsi.cards import Card, Rank, Suit

    def card(card_id: int) -> dict:
        return Card(Suit(card_id // 8), Rank(card_id % 8)).to_dict()

    dont = view.get("dont_play") or {}
    moves = []
    for move in stats["moves"]:
        action = move["action"]
        label = short_label(action, dont)
        moves.append(
            {
                "action": action,
                "label": label,
                "win_chance": move.get("win_chance"),
                "visits": move.get("visits", 0),
                "prior": move.get("prior"),
            }
        )
    moves.sort(key=lambda m: (-(m["win_chance"] if m["win_chance"] is not None else -1.0), -(m["visits"] or 0)))
    tracker = state.tracker
    definite = tracker.definite_opp
    return {
        "action": stats["action"],
        "win_chance": stats.get("win_chance"),
        "iterations": stats.get("iterations", 0),
        "seconds": stats.get("seconds", 0.0),
        "moves": moves,
        "unseen": [{**card(c), "certain": bool((definite >> c) & 1)} for c in mask_cards(tracker.unseen)],
        "opponent_cards": tracker.opp_size,
        "draw_pile": tracker.draw_size,
        "final": final,
    }


# ---------------------------------------------------------------- the player


class BotPlayer(Player):
    """A seat played by the bot. ``on_panel(seat, panel)`` receives progress.

    Every move stops at ``search_limit`` imagined games. In "My friend vs the
    bot", ``confirm_moves`` is set: the search also stops when the host calls
    ``confirm``, and the move is played only after that. In the other bot
    modes the think clock still decides when the card is played. If the cap
    is reached first, the finished evaluation is published and the bot waits
    out the rest of the clock.
    """

    _decisions = itertools.count(1)
    SEARCH_LIMIT = 100_000

    def __init__(self, seat: int, spec: dict, on_panel=None):
        self.seat = seat
        self.spec = spec
        self.label = spec.get("label") or spec["kind"]
        self.name = "Bot"
        self.on_panel = on_panel
        self._queue = None
        self.thinking = False
        self.confirm_moves = False
        self.search_limit = self.SEARCH_LIMIT
        self.awaiting_confirm = False
        self._turn_confirmed = False
        self._stop = None
        self._confirm: asyncio.Event | None = None
        self._choice = None

    def begin_turn(self) -> None:
        """A new turn is armed. A confirm from the previous turn does not count."""
        self._turn_confirmed = False

    def confirm(self) -> None:
        """The host accepts the move. Safe to call before thinking has started."""
        self._turn_confirmed = True
        stop = self._stop
        if stop is not None:
            stop.set()
        event = self._confirm
        if event is not None:
            event.set()

    def describe(self) -> dict:
        info = {"kind": "bot", "name": self.name, "engine": self.spec["kind"]}
        info["network"] = self.label if self.spec["kind"] in ("guided", "ppo") else None
        return info

    async def choose_action(self, view: dict) -> int:
        legal = list(view["legal_actions"])
        decision = next(self._decisions)
        loop = asyncio.get_running_loop()
        if self._queue is None:
            self._queue = await loop.run_in_executor(None, lambda: manager().Queue())
        progress_queue = self._queue
        await loop.run_in_executor(None, _drain, progress_queue)
        seed = secrets.randbits(48)
        # Lets a cancelled turn's cleanup leave the next turn's stop event alone.
        call = object()
        self._choice = call
        stop = None
        event = None
        iterations = self.search_limit
        # The friend's game has no clock: the host's confirm releases the move.
        seconds = None if self.confirm_moves else self.spec.get("seconds")
        if self.confirm_moves:
            # A manager event can cross into the worker. A raw event cannot:
            # the pool passes arguments through a queue, not by inheritance.
            # Created before the worker starts, so a confirm that arrived as
            # the turn opened (or during the drain above) is not missed.
            stop = manager().Event()
            event = asyncio.Event()
            self._stop = stop
            self._confirm = event
            self.awaiting_confirm = True
            if self._turn_confirmed:
                stop.set()
                event.set()
        future = loop.run_in_executor(
            executor(), think, self.spec, view, seed, progress_queue, decision, stop, iterations, seconds
        )
        self.thinking = True
        try:
            while not future.done():
                if stop is not None and self._turn_confirmed:
                    stop.set()
                await asyncio.wait({future}, timeout=0.2)
                latest = await loop.run_in_executor(None, _latest, progress_queue, decision)
                if latest is not None and self.on_panel is not None and not future.done():
                    await self.on_panel(self.seat, latest)
            try:
                panel = future.result()
            except Exception:
                log.exception("The bot failed to think; playing the first legal move.")
                return legal[0]
            if stop is not None and not self._turn_confirmed:
                paused = dict(panel)
                paused["final"] = False
                paused["paused"] = True
                if self.on_panel is not None:
                    await self.on_panel(self.seat, paused)
                await event.wait()
                panel = dict(panel)
                panel["final"] = True
                panel["paused"] = False
            elif seconds:
                remain = float(seconds) - float(panel.get("seconds") or 0.0)
                if remain > 0:
                    # The search is finished. Publish that panel now, then
                    # hold the card until the think clock runs out.
                    deadline = loop.time() + remain
                    if self.on_panel is not None:
                        await self.on_panel(self.seat, panel)
                    delay = deadline - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
        finally:
            if stop is not None:
                stop.set()
            if self._choice is call:
                self.thinking = False
                self.awaiting_confirm = False
                if self._stop is stop:
                    self._stop = None
                if self._confirm is event:
                    self._confirm = None
        if self.on_panel is not None:
            await self.on_panel(self.seat, panel)
        action = panel["action"]
        if action not in legal:
            log.error("The bot chose an illegal move %s; playing the first legal move.", action)
            return legal[0]
        return action


def _drain(progress_queue) -> None:
    try:
        while True:
            progress_queue.get_nowait()
    except (queue_module.Empty, EOFError, OSError):
        pass


def _latest(progress_queue, decision: int):
    latest = None
    try:
        while True:
            item_decision, panel = progress_queue.get_nowait()
            if item_decision == decision:
                latest = panel
    except (queue_module.Empty, EOFError, OSError):
        pass
    return latest
