"""Card counting from one seat's point of view.

The tracker only ever sees public events (plays, draw counts, stands, passes,
reshuffles) plus its own seat's hand and the public pile sizes. It can be fed
from the fast engine's event list or from ``observation["log"]``; both paths
produce the same events.

Where the cards are, from seat ``me``:

- my hand (exact),
- the discard pile, in order (exact),
- ``definite_opp``: cards certainly in the opponent's hand,
- ``pool``: cards that are either in the draw pile or in the opponent's hand;
  exactly ``pool_in_opp`` of them are in the opponent's hand,
- ``excluded``: cards the opponent cannot hold (the pass deduction).

Why ``definite_opp`` exists: a reshuffle only happens when the draw pile is
empty, so at that moment every unseen card is in the opponent's hand. After
it, the new draw pile is the old discard pile minus its top card - a known
set (``universe``) in an unknown order. Any unseen card outside that set must
still be in the opponent's hand. Before the first reshuffle the universe is the
whole deck, so every unseen card is in the pool.
"""

from __future__ import annotations

from bot.fastgame import (
    ALL_CARDS,
    EV_DRAW,
    EV_PASS,
    EV_PLAY,
    EV_SHUFFLE,
    EV_STAND,
    NUM_CARDS,
    OVERS,
    RANK_MASK,
    SUIT_MASK,
    FastGame,
    cards_mask,
)

HISTORY = 30

SUIT_KEYS = ("hearts", "leaves", "acorns", "bells")
_SUIT_INDEX = {key: index for index, key in enumerate(SUIT_KEYS)}


class Tracker:
    __slots__ = (
        "me",
        "discard",
        "discard_mask",
        "universe",
        "excluded",
        "active_suit",
        "my_hand",
        "opp_size",
        "draw_size",
        "passes",
        "n_actions",
        "recent",
        "cursor",
        "synced",
    )

    def __init__(self, me: int, opening: int | None = None):
        self.me = me
        self.discard: list[int] = []
        self.discard_mask = 0
        self.universe = ALL_CARDS
        self.excluded = 0
        self.active_suit = 0
        self.my_hand = 0
        self.opp_size = 0
        self.draw_size = 0
        self.passes = 0
        self.n_actions = 0
        # Actions since the opponent's hand was last known exactly, oldest first.
        self.recent: list[tuple[int, int, int, int]] = []
        # How many fast-engine events have been read.
        self.cursor = 0
        # False until my_hand and the pile sizes match a real position.
        self.synced = False
        if opening is not None:
            self.discard = [opening]
            self.discard_mask = 1 << opening
            self.active_suit = opening >> 3

    @classmethod
    def from_discard(cls, me: int, discard: list[int], active_suit: int) -> Tracker:
        """A tracker for a position built mid-game, assuming the whole discard pile is known."""
        tracker = cls(me)
        tracker.discard = list(discard)
        tracker.discard_mask = cards_mask(discard)
        tracker.active_suit = active_suit
        return tracker

    def clone(self) -> Tracker:
        copy = Tracker.__new__(Tracker)
        copy.me = self.me
        copy.discard = self.discard[:]
        copy.discard_mask = self.discard_mask
        copy.universe = self.universe
        copy.excluded = self.excluded
        copy.active_suit = self.active_suit
        copy.my_hand = self.my_hand
        copy.opp_size = self.opp_size
        copy.draw_size = self.draw_size
        copy.passes = self.passes
        copy.n_actions = self.n_actions
        copy.recent = self.recent[:]
        copy.cursor = self.cursor
        copy.synced = self.synced
        return copy

    # ------------------------------------------------------------ feeding

    def apply(self, event: tuple[int, int, int, int]) -> None:
        kind, player, a, b = event
        if kind == EV_SHUFFLE:
            top = self.discard[-1]
            self.universe = self.discard_mask & ~(1 << top)
            self.discard = [top]
            self.discard_mask = 1 << top
            return
        self.n_actions += 1
        if kind == EV_PLAY:
            self.discard.append(a)
            self.discard_mask |= 1 << a
            self.active_suit = b if b >= 0 else a >> 3
            self.passes = 0
        elif kind == EV_DRAW:
            if player != self.me:
                self.excluded = 0
            self.passes = 0
        elif kind == EV_STAND:
            self.passes = 0
        elif kind == EV_PASS:
            self.passes += 1
            if player != self.me and not a:
                # Nothing left to draw and no 7 or Ace active: passing was
                # only allowed because nothing in that hand could be played.
                top = self.discard[-1]
                self.excluded |= SUIT_MASK[self.active_suit] | RANK_MASK[top & 7] | OVERS
        self.recent.append(event)
        if len(self.recent) > HISTORY:
            del self.recent[0]

    def sync(self, my_hand: int, opp_size: int, draw_size: int) -> None:
        self.my_hand = my_hand
        self.opp_size = opp_size
        self.draw_size = draw_size

    def opponent_hand_certain(self) -> bool:
        """True when only one set of cards can be the opponent's hand."""
        need = self.pool_in_opp
        options = (self.pool & ~self.excluded).bit_count()
        return need == 0 or need == options

    def _forget_if_certain(self) -> None:
        if self.opponent_hand_certain():
            self.recent.clear()

    def _replay(
        self,
        events: list[tuple[int, int, int, int]],
        my_hand: int,
        opp_size: int,
        draw_size: int,
        drawn: list[int],
        draw_at: int,
    ) -> None:
        """Apply ``events``, then clear the memory whenever the hand is certain.

        ``drawn`` is this seat's drawn card ids. ``draw_at`` is how many of them
        belong to events that were already applied.
        """
        me = self.me
        for event in events:
            kind, player, count, _flag = event
            if kind == EV_SHUFFLE:
                draw_size += max(len(self.discard) - 1, 0)
                self.apply(event)
                continue
            self.apply(event)
            if kind == EV_PLAY:
                if player == me:
                    my_hand &= ~(1 << count)
                else:
                    opp_size -= 1
            elif kind == EV_DRAW:
                if player == me:
                    taken = drawn[draw_at : draw_at + count]
                    if len(taken) != count:
                        raise RuntimeError("The draw record is missing cards this seat drew.")
                    for card in taken:
                        my_hand |= 1 << card
                    draw_at += count
                else:
                    opp_size += count
                draw_size -= count
            self.sync(my_hand, opp_size, draw_size)
            self._forget_if_certain()

    def replay_all(
        self,
        events: list[tuple[int, int, int, int]],
        final_hand: int,
        final_opp: int,
        final_draw: int,
        drawn: list[int],
    ) -> None:
        """Rebuild from the opening deal. ``drawn`` is this seat's cards, in order.

        A card can be drawn again after it is played and the discard pile is
        reshuffled, so the opening hand is recovered by undoing events from
        the end, not by cancelling every drawn id.
        """
        me = self.me
        my_draws: list[list[int]] = []
        opp_plays = 0
        opp_draws = 0
        at = 0
        for kind, player, count, _flag in events:
            if kind == EV_PLAY and player != me:
                opp_plays += 1
            elif kind == EV_DRAW:
                if player == me:
                    taken = drawn[at : at + count]
                    if len(taken) != count:
                        raise RuntimeError("The draw record is missing cards this seat drew.")
                    my_draws.append(taken)
                    at += count
                else:
                    opp_draws += count
        if at != len(drawn):
            raise RuntimeError("The draw record has cards that were never drawn.")
        my_hand = final_hand
        draw_index = len(my_draws)
        for kind, player, count, _flag in reversed(events):
            if kind == EV_PLAY and player == me:
                my_hand |= 1 << count
            elif kind == EV_DRAW and player == me:
                draw_index -= 1
                for card in my_draws[draw_index]:
                    my_hand &= ~(1 << card)
        opp_size = final_opp + opp_plays - opp_draws
        draw_size = NUM_CARDS - my_hand.bit_count() - opp_size - len(self.discard)
        self._replay(events, my_hand, opp_size, draw_size, drawn, 0)
        self.sync(final_hand, final_opp, final_draw)
        self._forget_if_certain()
        self.synced = True

    def catch_up(self, game: FastGame) -> Tracker:
        """Read new events from a fast game (recorded with ``record_events``)."""
        events = game.events
        if events is None:
            raise RuntimeError("catch_up needs a game recorded with record_events")
        me = self.me
        final_hand = game.hands[me]
        final_opp = game.hands[1 - me].bit_count()
        final_draw = len(game.draw)
        new = events[self.cursor :]
        if not self.synced and new:
            self.replay_all(events, final_hand, final_opp, final_draw, game.drawn[me])
        else:
            if new:
                used = _own_draws_before(events, me, self.cursor)
                self._replay(new, self.my_hand, self.opp_size, self.draw_size, game.drawn[me], used)
            self.sync(final_hand, final_opp, final_draw)
            self._forget_if_certain()
            self.synced = True
        self.cursor = len(events)
        return self

    # ------------------------------------------------------------ answers

    @property
    def top(self) -> int:
        return self.discard[-1]

    @property
    def unseen(self) -> int:
        return ALL_CARDS & ~self.my_hand & ~self.discard_mask

    @property
    def definite_opp(self) -> int:
        return self.unseen & ~self.universe

    @property
    def pool(self) -> int:
        return self.unseen & self.universe

    @property
    def pool_in_opp(self) -> int:
        return self.opp_size - self.definite_opp.bit_count()

    @property
    def possible_opp(self) -> int:
        """Cards the opponent might hold."""
        return self.definite_opp | (self.pool & ~self.excluded)

    @property
    def under_top(self) -> int:
        return self.discard_mask & ~(1 << self.discard[-1])

    def snapshot(self) -> tuple:
        """Everything the tracker knows, for comparing two trackers."""
        return (
            self.me,
            tuple(self.discard),
            self.universe & self.unseen,
            self.excluded & self.unseen,
            self.active_suit,
            self.my_hand,
            self.opp_size,
            self.draw_size,
            self.passes,
            self.n_actions,
            tuple(self.recent),
        )


# ---------------------------------------------------------------- observations


def event_from_log(entry: dict) -> tuple[int, int, int, int] | None:
    """One ``prsi.Game`` log entry as a fast-engine event (None for opening / game over)."""
    kind = entry.get("kind")
    if kind == "play":
        named = entry.get("suit")
        return (EV_PLAY, entry["player"], entry["card"]["id"], _SUIT_INDEX[named] if named else -1)
    if kind == "draw":
        return (EV_DRAW, entry["player"], entry["count"], 1 if entry.get("penalty") else 0)
    if kind == "stand":
        return (EV_STAND, entry["player"], 0, 0)
    if kind == "pass":
        return (EV_PASS, entry["player"], 1 if entry.get("forgiven") else 0, 0)
    if kind == "shuffle":
        return (EV_SHUFFLE, -1, 0, 0)
    return None


def events_from_log(log: list[dict]) -> list[tuple[int, int, int, int]]:
    out = []
    for entry in log:
        event = event_from_log(entry)
        if event is not None:
            out.append(event)
    return out


def _own_draws_before(events: list[tuple[int, int, int, int]], seat: int, stop: int) -> int:
    total = 0
    for index in range(stop):
        kind, player, count, _flag = events[index]
        if kind == EV_DRAW and player == seat:
            total += count
    return total


def tracker_from_observation(view: dict) -> Tracker:
    """Build the tracker for ``view["you"]`` from a ``prsi.Game.observation``."""
    log = view["log"]
    if not log or log[0].get("kind") != "opening":
        raise ValueError("The observation log must start with the opening card.")
    tracker = Tracker(view["you"], log[0]["card"]["id"])
    events = []
    for entry in log[1:]:
        event = event_from_log(entry)
        if event is not None:
            events.append(event)
    tracker.replay_all(
        events,
        cards_mask(card["id"] for card in view["hand"]),
        view["opponent_hand_size"],
        view["draw_pile_size"],
        list(view.get("drawn") or []),
    )
    tracker.cursor = len(events)
    return tracker
