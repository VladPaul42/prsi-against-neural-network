"""Prší rules, with no networking and no interface.

Import this package from a script, a test, or the web server and play a full
game through ``Game``. The bot uses ``Game.observation`` and the action
numbers from ``prsi.actions`` — nothing else about the position.
"""

from prsi.actions import DONT_PLAY, NUM_ACTIONS
from prsi.cards import Card, Rank, Suit, full_deck
from prsi.game import Game, IllegalAction

__all__ = [
    "Card",
    "DONT_PLAY",
    "Game",
    "IllegalAction",
    "NUM_ACTIONS",
    "Rank",
    "Suit",
    "full_deck",
]
