"""The Prší rules again, written for speed instead of clarity.

Same seed and same actions give the same hands, piles, legal actions and
result as ``prsi.Game`` at every step; ``tests/test_fastgame.py`` checks this
in lockstep. Cards are the ids from ``prsi.cards`` (``suit * 8 + rank``).
Hands are 32-bit masks, because the order of a hand never matters. The draw
and discard piles are lists, bottom first, because their order does.

``step`` trusts its caller: pass only actions from ``legal_actions``.
"""

from __future__ import annotations

import random

from prsi.game import OPENING_HAND

NUM_ACTIONS = 45
DONT_PLAY = 44
NUM_CARDS = 32
ALL_CARDS = (1 << NUM_CARDS) - 1

SEVEN, EIGHT, NINE, TEN, UNDER, OVER, KING, ACE = range(8)

# Public events, as tuples (kind, player, a, b):
#   EV_PLAY    a = card, b = named suit or -1
#   EV_DRAW    a = count, b = 1 for a 7-penalty, else 0
#   EV_STAND
#   EV_PASS    a = 1 if a 7-penalty was forgiven, else 0
#   EV_SHUFFLE player = -1
EV_PLAY, EV_DRAW, EV_STAND, EV_PASS, EV_SHUFFLE = range(5)

REASON_EMPTY_HAND = "empty_hand"
REASON_STUCK = "stuck"
REASON_LIMIT = "limit"


def card_suit(card: int) -> int:
    return card >> 3


def card_rank(card: int) -> int:
    return card & 7


SUIT_MASK = [0xFF << (8 * suit) for suit in range(4)]
RANK_MASK = [sum(1 << (suit * 8 + rank) for suit in range(4)) for rank in range(8)]
SEVENS = RANK_MASK[SEVEN]
OVERS = RANK_MASK[OVER]
ACES = RANK_MASK[ACE]

# Cards other than Overs that may follow (active suit, top rank).
PLAYABLE = [
    (SUIT_MASK[suit] | RANK_MASK[rank]) & ~OVERS
    for suit in range(4)
    for rank in range(8)
]

_ACTION_RANK_INDEX = {SEVEN: 0, EIGHT: 1, NINE: 2, TEN: 3, UNDER: 4, KING: 5, ACE: 6}

# Action -> card, and action -> named suit (-1 for a plain card).
ACTION_CARD = [0] * DONT_PLAY
ACTION_NAMED = [-1] * DONT_PLAY
# Card -> its plain action, or -1 for an Over.
CARD_ACTION = [-1] * NUM_CARDS
for _card in range(NUM_CARDS):
    _suit, _rank = _card >> 3, _card & 7
    if _rank == OVER:
        for _named in range(4):
            _action = 28 + _suit * 4 + _named
            ACTION_CARD[_action] = _card
            ACTION_NAMED[_action] = _named
    else:
        _action = _suit * 7 + _ACTION_RANK_INDEX[_rank]
        CARD_ACTION[_card] = _action
        ACTION_CARD[_action] = _card


def _byte_actions(suit: int, byte: int) -> tuple[int, ...]:
    out = []
    for rank in range(8):
        if byte >> rank & 1 and rank != OVER:
            out.append(CARD_ACTION[suit * 8 + rank])
    return tuple(out)


def _byte_overs(suit: int, byte: int) -> tuple[int, ...]:
    if byte >> OVER & 1:
        return tuple(28 + suit * 4 + named for named in range(4))
    return ()


# Sorted actions for the plain cards / the Over in one suit's byte of a mask.
_PLAIN_BY_BYTE = [[_byte_actions(suit, byte) for byte in range(256)] for suit in range(4)]
_OVER_BY_BYTE = [[_byte_overs(suit, byte) for byte in range(256)] for suit in range(4)]
_P0, _P1, _P2, _P3 = _PLAIN_BY_BYTE
_O0, _O1, _O2, _O3 = _OVER_BY_BYTE


def mask_to_actions(plain: int, overs: int) -> list[int]:
    """Sorted actions playing the cards in ``plain`` and ``overs`` (all four named suits)."""
    out = [
        *_P0[plain & 255],
        *_P1[(plain >> 8) & 255],
        *_P2[(plain >> 16) & 255],
        *_P3[(plain >> 24) & 255],
    ]
    if overs:
        out += _O0[overs & 255]
        out += _O1[(overs >> 8) & 255]
        out += _O2[(overs >> 16) & 255]
        out += _O3[(overs >> 24) & 255]
    return out


def mask_cards(mask: int) -> list[int]:
    """Card ids in ``mask``, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def cards_mask(cards) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << card
    return mask


def shuffle_ids(rng: random.Random, items: list) -> None:
    """``prsi.cards.shuffle``, step for step.

    ``Random.randrange(n)`` with one positive argument returns
    ``Random._randbelow(n)``; calling it directly gives the same numbers.
    """
    below = rng._randbelow
    for i in range(len(items) - 1, 0, -1):
        j = below(i + 1)
        items[i], items[j] = items[j], items[i]


class FastGame:
    __slots__ = (
        "seed",
        "starting_player",
        "rng",
        "hands",
        "draw",
        "discard",
        "active_suit",
        "pending",
        "ace",
        "suit_named",
        "current",
        "passes",
        "over",
        "winner",
        "reason",
        "n_actions",
        "max_actions",
        "events",
        "drawn",
    )

    def __init__(
        self,
        seed: int,
        starting_player: int = 0,
        *,
        max_actions: int | None = None,
        record_events: bool = False,
        opening_hand: int = OPENING_HAND,
    ):
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        if starting_player not in (0, 1):
            raise ValueError("starting_player must be 0 or 1")
        self.seed = seed
        self.starting_player = starting_player
        self.rng = random.Random(seed)
        deck = list(range(NUM_CARDS))
        shuffle_ids(self.rng, deck)
        n = opening_hand
        self.hands = [cards_mask(deck[0:n]), cards_mask(deck[n : 2 * n])]
        top = deck[2 * n]
        self.discard = [top]
        self.draw = deck[2 * n + 1 :][::-1]
        self.active_suit = top >> 3
        self.pending = 0
        self.ace = False
        self.suit_named = False
        self.current = starting_player
        self.passes = 0
        self.over = False
        self.winner = None
        self.reason = None
        self.n_actions = 0
        self.max_actions = max_actions
        self.events = [] if record_events else None
        self.drawn = [[], []]

    @classmethod
    def from_state(
        cls,
        hands: tuple[int, int] | list[int],
        draw: list[int],
        discard: list[int],
        *,
        active_suit: int,
        pending: int = 0,
        ace: bool = False,
        suit_named: bool = False,
        current: int = 0,
        passes: int = 0,
        rng: random.Random | None = None,
        n_actions: int = 0,
        max_actions: int | None = None,
        record_events: bool = False,
    ) -> FastGame:
        """Build a position directly. The lists are used as given, not copied."""
        game = cls.__new__(cls)
        game.seed = None
        game.starting_player = current
        game.rng = rng if rng is not None else random.Random()
        game.hands = [hands[0], hands[1]]
        game.draw = draw
        game.discard = discard
        game.active_suit = active_suit
        game.pending = pending
        game.ace = ace
        game.suit_named = suit_named
        game.current = current
        game.passes = passes
        game.over = False
        game.winner = None
        game.reason = None
        game.n_actions = n_actions
        game.max_actions = max_actions
        game.events = [] if record_events else None
        game.drawn = [[], []]
        return game

    def clone(self, rng: random.Random | None = None, *, events: bool = False) -> FastGame:
        """An independent copy.

        Without ``rng`` the copy gets its own copy of the generator, so it
        reshuffles exactly as the original would. Search passes a shared
        generator instead, which is much cheaper. ``events`` keeps a copy of
        the event list; otherwise the copy records nothing.
        """
        game = FastGame.__new__(FastGame)
        game.seed = self.seed
        game.starting_player = self.starting_player
        if rng is None:
            rng = random.Random()
            rng.setstate(self.rng.getstate())
        game.rng = rng
        game.hands = [self.hands[0], self.hands[1]]
        game.draw = self.draw[:]
        game.discard = self.discard[:]
        game.active_suit = self.active_suit
        game.pending = self.pending
        game.ace = self.ace
        game.suit_named = self.suit_named
        game.current = self.current
        game.passes = self.passes
        game.over = self.over
        game.winner = self.winner
        game.reason = self.reason
        game.n_actions = self.n_actions
        game.max_actions = self.max_actions
        game.events = list(self.events) if events and self.events is not None else None
        game.drawn = [row[:] for row in self.drawn] if events and self.events is not None else [[], []]
        return game

    # ------------------------------------------------------------------ queries

    @property
    def top(self) -> int:
        return self.discard[-1]

    def available_to_draw(self) -> int:
        n = len(self.discard)
        return len(self.draw) + (n - 1 if n > 1 else 0)

    def hand_size(self, player: int) -> int:
        return self.hands[player].bit_count()

    def legal_actions(self) -> list[int]:
        if self.over:
            return []
        hand = self.hands[self.current]
        if self.pending:
            out = mask_to_actions(hand & SEVENS, 0)
            out.append(DONT_PLAY)
            return out
        if self.ace:
            out = mask_to_actions(hand & ACES, 0)
            out.append(DONT_PLAY)
            return out
        top = self.discard[-1]
        plain = hand & PLAYABLE[self.active_suit * 8 + (top & 7)]
        overs = hand & OVERS
        out = mask_to_actions(plain, overs)
        if not out or len(self.draw) or len(self.discard) > 1:
            out.append(DONT_PLAY)
        return out

    def playable_mask(self) -> int:
        """Cards the player to move could play right now (Overs included)."""
        hand = self.hands[self.current]
        if self.pending:
            return hand & SEVENS
        if self.ace:
            return hand & ACES
        top = self.discard[-1]
        return hand & (PLAYABLE[self.active_suit * 8 + (top & 7)] | OVERS)

    def result_for(self, player: int) -> float:
        """+1 win, -1 loss, 0 draw, from ``player``'s side. Only after the end."""
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # ------------------------------------------------------------------ moves

    def step(self, action: int) -> None:
        self.n_actions += 1
        if action == DONT_PLAY:
            self._dont_play()
        else:
            self._play(action)
        if not self.over and self.max_actions is not None and self.n_actions >= self.max_actions:
            self.over = True
            self.winner = None
            self.reason = REASON_LIMIT

    def _play(self, action: int) -> None:
        player = self.current
        card = ACTION_CARD[action]
        named = ACTION_NAMED[action]
        hand = self.hands[player] & ~(1 << card)
        self.hands[player] = hand
        self.discard.append(card)
        if self.events is not None:
            self.events.append((EV_PLAY, player, card, named))
        rank = card & 7
        self._after_card(rank, card, named)
        if not hand:
            self.over = True
            self.winner = player
            self.reason = REASON_EMPTY_HAND
            if rank == SEVEN or rank == ACE:
                self.pending = 0
                self.ace = False
            return
        self.passes = 0
        self.current = 1 - player

    def _after_card(self, rank: int, card: int, named: int) -> None:
        if rank == SEVEN:
            self.pending += 1
            self.ace = False
            self.suit_named = False
            self.active_suit = card >> 3
        elif rank == ACE:
            self.pending = 0
            self.ace = True
            self.suit_named = False
            self.active_suit = card >> 3
        elif rank == OVER:
            self.pending = 0
            self.ace = False
            self.suit_named = True
            self.active_suit = named
        else:
            self.pending = 0
            self.ace = False
            self.suit_named = False
            self.active_suit = card >> 3

    def _dont_play(self) -> None:
        player = self.current
        if self.pending:
            owed = self.pending * 2
            got = self._draw_into(player, owed)
            self.pending = 0
            if got == 0:
                self._pass(player, 1)
            else:
                self.passes = 0
                if self.events is not None:
                    self.events.append((EV_DRAW, player, got, 1))
                self._reshuffle_if_needed()
                self.current = 1 - player
            return
        if self.ace:
            self.ace = False
            self.passes = 0
            if self.events is not None:
                self.events.append((EV_STAND, player, 0, 0))
            self.current = 1 - player
            return
        if self.draw or len(self.discard) > 1:
            got = self._draw_into(player, 1)
            self.passes = 0
            if self.events is not None:
                self.events.append((EV_DRAW, player, got, 0))
            self._reshuffle_if_needed()
            self.current = 1 - player
            return
        self._pass(player, 0)

    def _draw_into(self, player: int, count: int) -> int:
        got = 0
        hand = self.hands[player]
        draw = self.draw
        for _ in range(count):
            if not draw:
                self._reshuffle_if_needed()
                draw = self.draw
                if not draw:
                    break
            card = draw.pop()
            hand |= 1 << card
            self.drawn[player].append(card)
            got += 1
        self.hands[player] = hand
        return got

    def _reshuffle_if_needed(self) -> None:
        if self.draw or len(self.discard) <= 1:
            return
        top = self.discard[-1]
        rest = self.discard[:-1]
        shuffle_ids(self.rng, rest)
        self.discard = [top]
        self.draw = rest
        if self.events is not None:
            self.events.append((EV_SHUFFLE, -1, 0, 0))

    def _pass(self, player: int, forgiven: int) -> None:
        self.passes += 1
        if self.events is not None:
            self.events.append((EV_PASS, player, forgiven, 0))
        if self.passes >= 2:
            a = self.hands[0].bit_count()
            b = self.hands[1].bit_count()
            self.over = True
            self.winner = 0 if a < b else 1 if b < a else None
            self.reason = REASON_STUCK
            return
        self.current = 1 - player

    # ------------------------------------------------------------------ helpers

    def debug_state(self) -> dict:
        """Hidden state in the same shape as ``prsi.Game.debug_state`` (hands sorted)."""
        result = None
        if self.over:
            result = {"winner": self.winner, "reason": self.reason}
        return {
            "hands": [mask_cards(self.hands[0]), mask_cards(self.hands[1])],
            "draw": list(self.draw),
            "discard": list(self.discard),
            "active_suit": self.active_suit,
            "pending_sevens": self.pending,
            "ace_active": self.ace,
            "suit_named": self.suit_named,
            "current": self.current,
            "consecutive_passes": self.passes,
            "result": result,
        }

    def __repr__(self) -> str:
        return (
            f"FastGame(turn={self.current}, top={self.discard[-1]}, suit={self.active_suit}, "
            f"sevens={self.pending}, ace={self.ace}, over={self.over})"
        )


def from_prsi_game(game) -> FastGame:
    """Copy a ``prsi.Game`` position, including its random generator state."""
    rng = random.Random()
    rng.setstate(game.rng.getstate())
    fast = FastGame.from_state(
        [cards_mask(c.id for c in game.hands[0]), cards_mask(c.id for c in game.hands[1])],
        [c.id for c in game.draw],
        [c.id for c in game.discard],
        active_suit=game.active_suit.value,
        pending=game.pending_sevens,
        ace=game.ace_active,
        suit_named=game.suit_named,
        current=game.current,
        passes=game.consecutive_passes,
        rng=rng,
        n_actions=len(game.actions),
    )
    fast.seed = game.seed
    fast.starting_player = game.starting_player
    if game.result is not None:
        fast.over = True
        fast.winner = game.result["winner"]
        fast.reason = game.result["reason"]
    return fast
