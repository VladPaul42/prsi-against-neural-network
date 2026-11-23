"""Information Set Monte Carlo Tree Search (single observer).

Every iteration imagines one deal of the hidden cards that fits what the card
tracker knows: the opponent's hand is the cards known to be there plus a
random sample from the pool, and the draw pile is the rest of the pool in a
fresh random order. The imagined game is played down the tree. At every
decision, among the moves legal in that deal, the one tried least often goes
next. A move the tree has never seen is tried before a repeat. A move that
has been winning gets no extra tries.

One new move is added per iteration. The rest of that deal is finished by an
``Evaluator``. On the way back, the player to move is treated as playing
their best card that has actually been tried: the highest win chance for the
searching player, the lowest for the opponent. Cards with no tries yet stay
out of that choice. A move's recorded chance is the average of those results
across the deals in which it was tried.

The tree is keyed by moves, not by deals. Only ``PublicState`` goes in. The
real game is never touched.
"""

from __future__ import annotations

import random
import time

from bot.agents import Agent, PublicState
from bot.fastgame import DONT_PLAY, FastGame, mask_cards
from bot.tracker import Tracker


class Node:
    __slots__ = ("player", "visits", "reward", "avail", "prior", "fpu", "children")

    def __init__(self, player: int, prior: float = 0.0):
        self.player = player  # who made the move into this node
        self.visits = 0
        self.reward = 0.0  # summed win chance for the searching player
        self.avail = 1  # times this move has been legal, including the visit that created it
        self.prior = prior  # network's instant opinion; selection does not read it
        self.fpu = 0.5  # network score of this position; selection does not read it
        self.children: dict[int, Node] | None = {}


# ------------------------------------------------------------------ deals


def determinize(state: PublicState, rng: random.Random, *, record_events: bool = False,
                max_actions: int | None = None) -> FastGame:
    """One imagined game that fits everything ``state`` knows."""
    tracker = state.tracker
    me = state.seat
    pool = mask_cards(tracker.pool)
    need = tracker.pool_in_opp
    allowed = [card for card in pool if not (tracker.excluded >> card) & 1]
    if len(allowed) < need:
        allowed = pool
    rng.shuffle(allowed)
    opp = tracker.definite_opp
    taken = 0
    for card in allowed[:need]:
        opp |= 1 << card
        taken |= 1 << card
    draw = [card for card in pool if not (taken >> card) & 1]
    rng.shuffle(draw)
    hands = [0, 0]
    hands[me] = tracker.my_hand
    hands[1 - me] = opp
    return FastGame.from_state(
        hands,
        draw,
        tracker.discard[:],
        active_suit=tracker.active_suit,
        pending=state.pending,
        ace=state.ace,
        suit_named=state.suit_named,
        current=me,
        passes=tracker.passes,
        rng=rng,
        n_actions=tracker.n_actions,
        max_actions=max_actions,
        record_events=record_events,
    )


# ------------------------------------------------------------------ evaluators


class Evaluator:
    """Scores a position reached in the tree for ``root`` (1 win, 0.5 draw, 0 loss)."""

    needs_events = False

    def evaluate(self, game: FastGame, root: int, rng: random.Random, tracker: Tracker | None) -> float:
        raise NotImplementedError


class RandomRollout(Evaluator):
    """Finish the imagined game with random moves.

    ``draw_prob`` is the chance of choosing "don't play" when a card could be
    played. ``None`` means uniform over all legal actions.
    """

    def __init__(self, draw_prob: float | None = 0.0, max_length: int = 300):
        self.draw_prob = draw_prob
        self.max_length = max_length

    def evaluate(self, game, root, rng, tracker=None) -> float:
        choice = rng.choice
        rand = rng.random
        draw_prob = self.draw_prob
        limit = game.n_actions + self.max_length
        while not game.over:
            if game.n_actions >= limit:
                return 0.5
            legal = game.legal_actions()
            if draw_prob is not None and len(legal) > 1 and legal[-1] == DONT_PLAY:
                if draw_prob <= 0.0 or rand() >= draw_prob:
                    legal.pop()
                else:
                    game.step(DONT_PLAY)
                    continue
            game.step(choice(legal))
        if game.winner is None:
            return 0.5
        return 1.0 if game.winner == root else 0.0


# ------------------------------------------------------------------ search


def choose_even(children: dict[int, Node], legal: list[int], rng: random.Random) -> int:
    """The legal move skipped most often.

    A move missing from the tree is tried before any repeat. Otherwise the
    largest gap between times legal (``avail``) and times tried (``visits``)
    wins. Ties are random. Existing moves have ``avail`` counted for this deal.
    """
    untried: list[int] = []
    best: list[int] = []
    best_deficit = -1
    for action in legal:
        child = children.get(action)
        if child is None:
            untried.append(action)
            continue
        child.avail += 1
        deficit = child.avail - child.visits
        if deficit > best_deficit:
            best_deficit = deficit
            best = [action]
        elif deficit == best_deficit:
            best.append(action)
    pool = untried or best
    return pool[0] if len(pool) == 1 else rng.choice(pool)


def best_reply(node: Node, legal: list[int], me: int) -> float | None:
    """Win chance for ``me`` if the player to move plays their best tried card.

    ``None`` when none of ``legal`` has been tried. Cards with no tries yet are left out.
    """
    children = node.children
    if not children:
        return None
    chosen = None
    chooser = None
    for action in legal:
        child = children.get(action)
        if child is None or child.visits == 0:
            continue
        value = child.reward / child.visits
        if chosen is None:
            chooser = child.player
            chosen = value
        elif chooser == me:
            if value > chosen:
                chosen = value
        elif value < chosen:
            chosen = value
    return chosen


def backup_best(path: list[Node], legal_at: list[list[int] | None], value: float, me: int) -> None:
    """Record ``value`` (a win chance for ``me``) under best play.

    ``path`` is every move stepped into, the current decision first.
    ``legal_at[i]`` is what was legal at ``path[i]`` in this deal, or ``None``
    when the visit stopped there. A stopped node keeps ``value``. A node the
    search played past keeps the best tried reply that was legal in this deal.
    """
    outcome = value
    for index in range(len(path) - 1, -1, -1):
        node = path[index]
        legal = legal_at[index]
        if legal is not None:
            reply = best_reply(node, legal, me)
            if reply is not None:
                outcome = reply
        node.visits += 1
        node.reward += outcome


class ISMCTS:
    def __init__(self, evaluator: Evaluator | None = None, c: float = 0.7):
        self.evaluator = evaluator or RandomRollout()
        self.c = c  # accepted so old specs still load; selection does not use it

    def search(
        self,
        state: PublicState,
        rng: random.Random,
        *,
        iterations: int | None = None,
        seconds: float | None = None,
        progress=None,
        progress_every: float = 0.25,
        stop=None,
    ) -> dict:
        """Think, then return statistics for every legal move (see ``stats``).

        ``stop`` is an event with ``is_set()``. The host uses it to end the
        search early and play whatever has been found. Checked on the same
        cadence as the clock.
        """
        if iterations is None and seconds is None:
            iterations = 1000
        root = Node(1 - state.seat)
        self.prepare_root(root, state)
        start = time.perf_counter()
        deadline = start + seconds if seconds is not None else None
        next_report = start + progress_every
        done = 0
        watch = deadline is not None or progress is not None or stop is not None
        while True:
            if iterations is not None and done >= iterations:
                break
            if watch and (done & 7) == 0:
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    break
                if stop is not None and stop.is_set():
                    break
                if progress is not None and now >= next_report:
                    progress(self.stats(root, state, done, now - start))
                    next_report = now + progress_every
            self.iterate(root, state, rng)
            done += 1
        return self.stats(root, state, done, time.perf_counter() - start, rng=rng)

    def prepare_root(self, root: Node, state: PublicState) -> None:
        """Hook for the guided search, which adds network priors here."""

    def iterate(self, root: Node, state: PublicState, rng: random.Random) -> None:
        evaluator = self.evaluator
        game = determinize(state, rng, record_events=evaluator.needs_events)
        me = state.seat
        node = root
        path: list[Node] = []
        legal_at: list[list[int] | None] = []
        while not game.over:
            legal = game.legal_actions()
            if path:
                legal_at[-1] = legal
            if node.children is None:
                node.children = {}
            action = choose_even(node.children, legal, rng)
            child = node.children.get(action)
            fresh = child is None
            if fresh:
                child = Node(game.current)
                node.children[action] = child
            game.step(action)
            path.append(child)
            legal_at.append(None)
            if fresh:
                break
            node = child
        tracker = None
        if evaluator.needs_events:
            tracker = state.tracker.clone()
            tracker.cursor = 0
            tracker.catch_up(game)
        if game.over:
            value = 0.5 if game.winner is None else (1.0 if game.winner == me else 0.0)
        else:
            value = evaluator.evaluate(game, me, rng, tracker)
        backup_best(path, legal_at, value, me)

    def stats(
        self,
        root: Node,
        state: PublicState,
        iterations: int,
        seconds: float,
        *,
        rng: random.Random | None = None,
    ) -> dict:
        """Per legal move: visits and win chance; the chosen move; the position's win chance.

        The chosen move is the highest win chance. Exact ties are random when
        ``rng`` is given, and the lowest action number otherwise, so a progress
        report does not draw from the search's generator.
        """
        moves = []
        for action in state.legal:
            child = root.children.get(action)
            visits = child.visits if child else 0
            moves.append(
                {
                    "action": action,
                    "visits": visits,
                    "win_chance": (child.reward / visits) if visits else None,
                    "prior": child.prior if child and child.prior else None,
                }
            )
        ranked = [move for move in moves if move["win_chance"] is not None]
        if not ranked:
            best = moves[0]
        else:
            best_chance = max(move["win_chance"] for move in ranked)
            tied = [move for move in ranked if move["win_chance"] == best_chance]
            if len(tied) == 1 or rng is None:
                best = min(tied, key=lambda move: move["action"])
            else:
                best = rng.choice(tied)
        return {
            "action": best["action"],
            "win_chance": best["win_chance"],
            "iterations": iterations,
            "seconds": round(seconds, 3),
            "moves": moves,
        }


class ISMCTSAgent(Agent):
    name = "ismcts"

    def __init__(
        self,
        iterations: int | None = None,
        seconds: float | None = None,
        c: float = 0.7,
        draw_prob: float | None = None,
        rollout_length: int = 300,
        seed: int = 0,
    ):
        if iterations is None and seconds is None:
            iterations = 1000
        self.iterations = iterations
        self.seconds = seconds
        self.search_engine = ISMCTS(RandomRollout(draw_prob, rollout_length), c)
        self.rng = random.Random(seed)

    @classmethod
    def from_spec(cls, spec: dict) -> ISMCTSAgent:
        return cls(
            iterations=spec.get("iterations"),
            seconds=spec.get("seconds"),
            c=spec.get("c", 0.7),
            draw_prob=spec.get("draw_prob", None),
            rollout_length=spec.get("rollout_length", 300),
            seed=spec.get("seed", 0),
        )

    def new_game(self, seed: int) -> None:
        self.rng.seed(seed)

    def analyse(self, state: PublicState, progress=None, *, stop=None, iterations=None, seconds=None) -> dict:
        """Search from this position.

        ``iterations`` alone replaces the agent's own budget, including a time
        budget, so a confirmed move can run a fixed number of imagined games.
        Passing ``seconds`` as well keeps the clock: the search stops when
        either limit is reached. ``stop`` ends the search early when the host
        confirms.
        """
        search_rng = random.Random(self.rng.getrandbits(64))
        if iterations is None and seconds is None:
            if self.seconds is not None:
                use_iterations, use_seconds = None, self.seconds
            else:
                use_iterations, use_seconds = self.iterations, None
        else:
            use_iterations, use_seconds = iterations, seconds
        return self.search_engine.search(
            state,
            search_rng,
            iterations=use_iterations,
            seconds=use_seconds,
            progress=progress,
            stop=stop,
        )

    def act(self, state: PublicState) -> int:
        if len(state.legal) == 1:
            return state.legal[0]
        return self.analyse(state)["action"]
