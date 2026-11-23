"""Players for training, matches and the web app.

An agent gets a ``PublicState`` - exactly what a person in that seat knows -
and returns an action number. A ``PublicState`` is built either from the fast
engine (training, matches) or from ``prsi.Game.observation`` (the web app);
neither path hands over the opponent's cards or the draw pile order.

Agents are described by small dicts ("specs") so that worker processes can
build their own copies:

    {"kind": "random"}
    {"kind": "ismcts", "iterations": 2000}
    {"kind": "ismcts", "seconds": 1.0}
    {"kind": "ppo", "path": "runs/league/gen_001/model.pt"}
    {"kind": "guided", "path": "...", "seconds": 1.0}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from bot.fastgame import FastGame
from bot.tracker import Tracker, tracker_from_observation


@dataclass(slots=True)
class PublicState:
    seat: int
    legal: list[int]
    tracker: Tracker
    pending: int
    ace: bool
    suit_named: bool

    @classmethod
    def from_fast(cls, game: FastGame, tracker: Tracker) -> PublicState:
        """The view of the player to move. ``tracker`` must belong to that seat."""
        tracker.catch_up(game)
        return cls(
            seat=game.current,
            legal=game.legal_actions(),
            tracker=tracker,
            pending=game.pending,
            ace=game.ace,
            suit_named=game.suit_named,
        )

    @classmethod
    def from_observation(cls, view: dict) -> PublicState:
        return cls(
            seat=view["you"],
            legal=list(view["legal_actions"]),
            tracker=tracker_from_observation(view),
            pending=view["penalty_cards"] // 2,
            ace=bool(view["ace_active"]),
            suit_named=bool(view["named_by_over"]),
        )


class Agent:
    """Base class. ``act`` must return one of ``state.legal``."""

    name = "agent"

    def new_game(self, seed: int) -> None:
        """Called before every game, so that random choices are repeatable."""

    def act(self, state: PublicState) -> int:
        raise NotImplementedError

    def analyse(self, state: PublicState, progress=None) -> dict:
        """Choose a move and explain it. Search agents override this."""
        action = self.act(state)
        return {"action": action, "moves": [], "win_chance": None, "iterations": 0}


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def new_game(self, seed: int) -> None:
        self.rng.seed(seed)

    def act(self, state: PublicState) -> int:
        return self.rng.choice(state.legal)


def spec_name(spec: dict) -> str:
    if "name" in spec:
        return str(spec["name"])
    kind = spec["kind"]
    if kind == "random":
        return "random"
    if kind == "ismcts":
        budget = f"{spec['iterations']}it" if "iterations" in spec else f"{spec['seconds']}s"
        return f"ismcts-{budget}"
    if kind in ("ppo", "guided"):
        label = spec.get("label") or spec["path"]
        return f"{kind}-{label}"
    return json.dumps(spec, sort_keys=True)


def make_agent(spec: dict) -> Agent:
    kind = spec["kind"]
    if kind == "random":
        agent: Agent = RandomAgent(spec.get("seed", 0))
    elif kind == "ismcts":
        from bot.ismcts import ISMCTSAgent

        agent = ISMCTSAgent.from_spec(spec)
    elif kind == "ppo":
        from bot.net import PolicyAgent

        agent = PolicyAgent.from_spec(spec)
    elif kind == "guided":
        from bot.guided import GuidedAgent

        agent = GuidedAgent.from_spec(spec)
    else:
        raise ValueError(f"Unknown agent kind: {kind!r}")
    agent.name = spec_name(spec)
    return agent
