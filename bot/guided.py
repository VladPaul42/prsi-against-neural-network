"""The final bot: ISMCTS scored by the PPO network.

Same imagined deals, the same even tries, and the same best-card backup as
plain ISMCTS, with two changes:

- **Priors.** When a decision point is reached for the first time, the
  network gives a probability for every legal move from the view of the
  player to move there. Those probabilities are stored and shown. They do
  not decide which move is tried next. For the bot's own moves the view is
  its real tracker; for the imagined opponent it is the same public history
  with the opponent's imagined hand (the reshuffled-set bookkeeping is
  public, so both sides' trackers follow the same rules).
- **No random rollouts.** The new position is scored by the value head.

Output is the same as plain ISMCTS plus ``prior`` per move: the network's
instant opinion before searching.
"""

from __future__ import annotations

import random

import numpy as np

from bot.agents import Agent, PublicState
from bot.encode import INPUT_SIZE, encode_raw
from bot.fastgame import FastGame
from bot.ismcts import ISMCTS, Node, backup_best, choose_even, determinize
from bot.net import NumpyNet, load_numpy_net, resolve_weights
from bot.tracker import Tracker


class GuidedISMCTS(ISMCTS):
    def __init__(self, net: NumpyNet, c_puct: float = 1.5):
        super().__init__(evaluator=None, c=c_puct)
        self.net = net
        self.c_puct = c_puct  # accepted so old specs still load; selection does not use it
        self._x = np.zeros(INPUT_SIZE, dtype=np.float32)

    # ------------------------------------------------------------ network

    def _evaluate(self, seat, tracker, pending, ace, suit_named, legal) -> tuple[np.ndarray, float]:
        """Priors over ``legal`` and the win chance (0..1) for ``seat``."""
        x = encode_raw(seat, tracker, pending, ace, suit_named, self._x)
        logits, value = self.net.forward(x)
        chosen = logits[legal].astype(np.float64)
        chosen -= chosen.max()
        probs = np.exp(chosen)
        probs /= probs.sum()
        return probs, (float(value) + 1.0) / 2.0

    def prepare_root(self, root: Node, state: PublicState) -> None:
        probs, win = self._evaluate(
            state.seat, state.tracker, state.pending, state.ace, state.suit_named, state.legal
        )
        _expand(root, state.seat, state.legal, probs, win)

    # ------------------------------------------------------------ one iteration

    def iterate(self, root: Node, state: PublicState, rng: random.Random) -> None:
        me = state.seat
        game = determinize(state, rng, record_events=True)
        node = root
        path: list[Node] = []
        legal_at: list[list[int] | None] = []
        value = None
        while not game.over:
            if node.children is None:
                value = self._leaf_value(node, game, state.tracker, me)
                break
            legal = game.legal_actions()
            if path:
                legal_at[-1] = legal
            action = choose_even(node.children, legal, rng)
            child = node.children.get(action)
            if child is None:
                child = Node(game.current)
                child.children = None
                node.children[action] = child
            game.step(action)
            path.append(child)
            legal_at.append(None)
            node = child
        if value is None:
            value = 0.5 if game.winner is None else (1.0 if game.winner == me else 0.0)
        backup_best(path, legal_at, value, me)

    def _leaf_value(self, node: Node, game: FastGame, root_tracker: Tracker, me: int) -> float:
        """Expand ``node`` with priors and return the win chance for ``me``."""
        mover = game.current
        tracker = root_tracker.clone()
        tracker.cursor = 0
        tracker.catch_up(game)
        if mover != me:
            tracker = _other_side(tracker, game)
        legal = game.legal_actions()
        probs, win = self._evaluate(mover, tracker, game.pending, game.ace, game.suit_named, legal)
        _expand(node, mover, legal, probs, win)
        return win if mover == me else 1.0 - win


def _expand(node: Node, mover: int, legal: list[int], probs, win: float) -> None:
    node.children = {}
    node.fpu = win
    for action, p in zip(legal, probs):
        child = Node(mover, float(p))
        child.children = None
        node.children[action] = child


def _other_side(tracker: Tracker, game: FastGame) -> Tracker:
    """The same public history seen from the other seat of an imagined game."""
    other = tracker.clone()
    other.me = 1 - tracker.me
    other.excluded = 0
    other.sync(game.hands[other.me], game.hands[tracker.me].bit_count(), len(game.draw))
    return other


class GuidedAgent(Agent):
    name = "guided"

    def __init__(self, net: NumpyNet, iterations: int | None = None, seconds: float | None = None,
                 c_puct: float = 1.5, seed: int = 0, label: str | None = None):
        if iterations is None and seconds is None:
            seconds = 1.0
        self.iterations = iterations
        self.seconds = seconds
        self.search_engine = GuidedISMCTS(net, c_puct)
        self.rng = random.Random(seed)
        self.label = label

    @classmethod
    def from_spec(cls, spec: dict) -> GuidedAgent:
        return cls(
            load_numpy_net(spec["path"]),
            iterations=spec.get("iterations"),
            seconds=spec.get("seconds"),
            c_puct=spec.get("c_puct", 1.5),
            seed=spec.get("seed", 0),
            label=spec.get("label") or resolve_weights(spec["path"]).parent.name,
        )

    def new_game(self, seed: int) -> None:
        self.rng.seed(seed)

    def analyse(self, state: PublicState, progress=None, *, stop=None, iterations=None, seconds=None) -> dict:
        """Search from this position. See ``ISMCTSAgent.analyse``."""
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
