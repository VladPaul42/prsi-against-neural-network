"""Fixed 45-action encoding for Prší.

0-27   Play one non-Over card.
       Ranks, in order: 7, 8, 9, 10, Under, King, Ace (Over is left out).
       ``action = suit * 7 + rank_index``
       Suits, in order: Hearts, Leaves, Acorns, Bells.

28-43  Play one Over and name a suit.
       ``action = 28 + over_suit * 4 + named_suit``

44     Don't play. Depending on the position this draws one card, takes a
       7-penalty, stands on an Ace, or passes.

The numbers are part of the rules API. The bot is trained against them, so
they must not be rearranged.
"""

from __future__ import annotations

from prsi.cards import Card, Rank, Suit

NUM_ACTIONS = 45
DONT_PLAY = 44

# Rank order used by actions 0-27. Over is not in this list.
ACTION_RANKS: tuple[Rank, ...] = (
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.UNDER,
    Rank.KING,
    Rank.ACE,
)
_ACTION_RANK_INDEX = {rank: index for index, rank in enumerate(ACTION_RANKS)}


class ActionError(ValueError):
    """Raised when an action number does not encode a card play."""


def play_card_action(card: Card) -> int:
    """Action number for playing a non-Over card."""
    if card.rank is Rank.OVER:
        raise ActionError("An Over also needs the suit being named.")
    return card.suit.value * 7 + _ACTION_RANK_INDEX[card.rank]


def play_over_action(card: Card, named_suit: Suit) -> int:
    """Action number for playing an Over and naming ``named_suit``."""
    if card.rank is not Rank.OVER:
        raise ActionError("Only an Over names a suit.")
    return 28 + card.suit.value * 4 + named_suit.value


def action_to_play(action: int) -> tuple[Card, Suit | None]:
    """Decode a card action into ``(card, named_suit)``.

    ``named_suit`` is set only for an Over. ``DONT_PLAY`` is not a card.
    """
    if type(action) is not int or not 0 <= action < DONT_PLAY:
        raise ActionError(f"Not a card action: {action!r}")
    if action < 28:
        suit = Suit(action // 7)
        rank = ACTION_RANKS[action % 7]
        return Card(suit, rank), None
    over = action - 28
    return Card(Suit(over // 4), Rank.OVER), Suit(over % 4)


def actions_for_card(card: Card) -> list[tuple[int, Suit | None]]:
    """Every action that plays this card, ignoring whether it is legal now."""
    if card.rank is Rank.OVER:
        return [(play_over_action(card, suit), suit) for suit in Suit]
    return [(play_card_action(card), None)]


def describe_action(action: int) -> str:
    if action == DONT_PLAY:
        return "don't play"
    card, named = action_to_play(action)
    if named is None:
        return f"play {card.name}"
    return f"play {card.name}, name {named.label}"


def short_label(action: int, dont: dict | None = None) -> str:
    """Compact move name for the live spectator panel.

    ``a7`` is Acorns 7. ``aO=h`` is Acorns Over naming hearts.
    Don't-play is ``D``, ``S``, ``T2``, or ``P``.
    """
    if action == DONT_PLAY:
        info = dont or {}
        kind = info.get("kind")
        if kind == "draw":
            return "D"
        if kind == "stand":
            return "S"
        if kind == "penalty":
            count = next(
                (part for part in str(info.get("label") or "").split() if part.isdigit()),
                "",
            )
            return f"T{count}" if count else "T"
        if kind == "pass":
            return "P"
        return str(info.get("label") or "Don't play")
    card, named = action_to_play(action)
    letter = card.suit.key[0]
    if named is None:
        return f"{letter}{card.rank.short}"
    return f"{letter}O={named.key[0]}"
