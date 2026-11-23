"""The 32-card German-suited deck, with English names."""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass


class Suit(enum.Enum):
    HEARTS = 0
    LEAVES = 1
    ACORNS = 2
    BELLS = 3

    @property
    def key(self) -> str:
        return ("hearts", "leaves", "acorns", "bells")[self.value]

    @property
    def label(self) -> str:
        return ("Hearts", "Leaves", "Acorns", "Bells")[self.value]


class Rank(enum.Enum):
    SEVEN = 0
    EIGHT = 1
    NINE = 2
    TEN = 3
    UNDER = 4
    OVER = 5
    KING = 6
    ACE = 7

    @property
    def key(self) -> str:
        return ("7", "8", "9", "10", "under", "over", "king", "ace")[self.value]

    @property
    def label(self) -> str:
        return ("7", "8", "9", "10", "Under", "Over", "King", "Ace")[self.value]

    @property
    def short(self) -> str:
        return ("7", "8", "9", "10", "U", "O", "K", "A")[self.value]


@dataclass(frozen=True, slots=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def id(self) -> int:
        return self.suit.value * 8 + self.rank.value

    @property
    def name(self) -> str:
        return f"{self.suit.label} {self.rank.label}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "suit": self.suit.key,
            "rank": self.rank.key,
            "name": self.name,
            "short": self.rank.short,
        }


def full_deck() -> list[Card]:
    return [Card(suit, rank) for suit in Suit for rank in Rank]


def shuffle(rng: random.Random, items: list) -> None:
    """Fisher–Yates driven by ``rng.randrange``.

    Python's ``Random.shuffle`` has changed between releases. This version
    stays the same, so a saved seed replays the same game later.
    """
    for i in range(len(items) - 1, 0, -1):
        j = rng.randrange(i + 1)
        items[i], items[j] = items[j], items[i]
