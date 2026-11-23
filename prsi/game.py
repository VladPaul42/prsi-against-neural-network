"""Prší rules.

Nothing in this module knows about sockets, browsers, or bots. Call
``legal_actions`` to learn which of the 45 action numbers are allowed, then
``step`` to play one. ``observation`` is everything one player is allowed to
know — never the opponent's cards, and never the order of the draw pile.

The opening card has no special effect. A 7 or an Ace only becomes active
when somebody plays one. Playing your last card wins immediately.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from prsi.actions import (
    DONT_PLAY,
    action_to_play,
    actions_for_card,
)
from prsi.cards import Card, Rank, Suit, full_deck, shuffle

# Cards dealt to each seat before the face-up card. A 32-card deck then
# leaves 19 in the draw pile.
OPENING_HAND = 6


class IllegalAction(Exception):
    """A move that ``legal_actions`` does not currently allow."""


class Game:
    def __init__(self, seed: int, starting_player: int = 0, opening_hand: int = OPENING_HAND):
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        if starting_player not in (0, 1):
            raise ValueError("starting_player must be 0 or 1")
        self.seed = seed
        self.starting_player = starting_player
        self.opening_hand = opening_hand
        self.rng = random.Random(seed)
        self.actions: list[int] = []
        self.result: dict | None = None
        self._deal()

    @classmethod
    def from_position(
        cls,
        hands: Sequence[Sequence[Card]],
        discard: Sequence[Card],
        draw: Sequence[Card] = (),
        *,
        active_suit: Suit | None = None,
        pending_sevens: int = 0,
        ace_active: bool = False,
        current_player: int = 0,
        consecutive_passes: int = 0,
        seed: int = 0,
        suit_named: bool = False,
    ) -> Game:
        """Build a position for tests or debugging.

        ``draw`` and ``discard`` are ordered from bottom to top: the last
        card of ``draw`` is the next one drawn, and the last card of
        ``discard`` is the face-up card. ``seed`` is used only if this
        position later has to reshuffle.
        """
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        if len(hands) != 2:
            raise ValueError("Prší is a two-player game.")
        if current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if pending_sevens < 0 or consecutive_passes < 0:
            raise ValueError("Counts cannot be negative.")
        if pending_sevens and ace_active:
            raise ValueError("A 7 penalty and an Ace cannot both be active.")
        if not discard:
            raise ValueError("The discard pile needs a top card.")

        game = cls.__new__(cls)
        game.seed = seed
        game.starting_player = current_player
        game.opening_hand = OPENING_HAND
        game.rng = random.Random(seed)
        game.actions = []
        game.result = None
        game.hands = [list(hands[0]), list(hands[1])]
        game.draw = list(draw)
        game.discard = list(discard)
        game.played = list(game.discard)
        game.active_suit = game.discard[-1].suit if active_suit is None else active_suit
        game.pending_sevens = pending_sevens
        game.ace_active = ace_active
        game.suit_named = bool(suit_named)
        game.current = current_player
        game.consecutive_passes = consecutive_passes
        game.log = []
        game.drawn = [[], []]
        game._check_cards()
        if not isinstance(game.active_suit, Suit):
            raise TypeError("active_suit must be a Suit.")
        return game

    def _deal(self) -> None:
        deck = full_deck()
        shuffle(self.rng, deck)
        # First n cards to seat 0, next n to seat 1, the next card face up.
        # The rest of the deck is the draw pile, last card drawn first.
        n = self.opening_hand
        self.hands = [deck[0:n], deck[n : 2 * n]]
        top = deck[2 * n]
        self.discard = [top]
        self.draw = deck[2 * n + 1 :][::-1]
        self.played = [top]
        self.active_suit = top.suit
        self.pending_sevens = 0
        self.ace_active = False
        # The turned-up card never counts as a named Over, even if it is one.
        self.suit_named = False
        self.current = self.starting_player
        self.consecutive_passes = 0
        self.drawn = [[], []]
        self.log = [{"kind": "opening", "player": None, "card": top.to_dict()}]

    def _check_cards(self) -> None:
        cards = [card for hand in self.hands for card in hand]
        cards.extend(self.draw)
        cards.extend(self.discard)
        ids = [card.id for card in cards]
        if len(ids) != len(set(ids)):
            raise ValueError("The same card appears twice in this position.")
        for card in cards:
            if not isinstance(card, Card):
                raise TypeError("Positions are made of Card objects.")

    @property
    def is_over(self) -> bool:
        return self.result is not None

    def available_to_draw(self) -> int:
        """Cards that can still be drawn, including a future reshuffle."""
        return len(self.draw) + max(0, len(self.discard) - 1)

    def legal_actions(self) -> list[int]:
        """The action numbers the current player may take, sorted."""
        if self.result is not None:
            return []
        hand = self.hands[self.current]
        actions: list[int] = []
        if self.pending_sevens > 0:
            for card in hand:
                if card.rank is Rank.SEVEN:
                    actions.append(actions_for_card(card)[0][0])
            actions.append(DONT_PLAY)
            return sorted(set(actions))
        if self.ace_active:
            for card in hand:
                if card.rank is Rank.ACE:
                    actions.append(actions_for_card(card)[0][0])
            actions.append(DONT_PLAY)
            return sorted(set(actions))

        top = self.discard[-1]
        can_play = False
        for card in hand:
            if card.rank is Rank.OVER:
                actions.extend(action_id for action_id, _suit in actions_for_card(card))
                can_play = True
            elif card.suit is self.active_suit or card.rank is top.rank:
                actions.append(actions_for_card(card)[0][0])
                can_play = True
        # Drawing is always allowed when a card exists. Passing is allowed
        # only when nothing can be drawn and nothing can be played.
        if self.available_to_draw() > 0 or not can_play:
            actions.append(DONT_PLAY)
        return sorted(set(actions))

    def step(self, action: int) -> None:
        """Play one action. Raises ``IllegalAction`` and changes nothing if it is illegal."""
        if self.result is not None:
            raise IllegalAction("The game is over.")
        if type(action) is not int or action not in self.legal_actions():
            raise IllegalAction("That move isn't allowed.")
        self.actions.append(action)
        if action == DONT_PLAY:
            self._dont_play()
        else:
            self._play(action)

    def observation(self, player: int) -> dict:
        """What ``player`` is allowed to see. Safe to send to that player's browser."""
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        your_turn = self.result is None and self.current == player
        legal = self.legal_actions() if your_turn else []
        legal_set = set(legal)
        hand_view = []
        for card in sorted(self.hands[player], key=lambda item: item.id):
            plays = []
            for action_id, named in actions_for_card(card):
                entry: dict = {"id": action_id}
                if named is not None:
                    entry["suit"] = named.key
                    entry["suit_name"] = named.label
                plays.append(entry)
            hand_view.append({**card.to_dict(), "actions": plays})
        if your_turn and DONT_PLAY in legal_set:
            dont = self._dont_play_info()
            dont["id"] = DONT_PLAY
            dont["legal"] = True
        else:
            dont = {"id": DONT_PLAY, "kind": None, "label": None, "legal": False}
        return {
            "you": player,
            "your_turn": your_turn,
            "hand": hand_view,
            "top_card": self.discard[-1].to_dict(),
            "under_cards": [card.to_dict() for card in self.discard[-3:-1]],
            "active_suit": self.active_suit.key,
            "active_suit_name": self.active_suit.label,
            "penalty_cards": self.pending_sevens * 2,
            "ace_active": self.ace_active,
            "named_by_over": self.suit_named,
            "played": [card.to_dict() for card in self.played],
            "opponent_hand_size": len(self.hands[1 - player]),
            "draw_pile_size": len(self.draw),
            "discard_size": len(self.discard),
            "current": self.current,
            "legal_actions": legal,
            "dont_play": dont,
            "log": [dict(event) for event in self.log],
            "drawn": list(self.drawn[player]),
            "result": None if self.result is None else dict(self.result),
        }

    def debug_state(self) -> dict:
        """Full hidden state, for tests and replay checks. Not for players."""
        return {
            "hands": [[card.id for card in hand] for hand in self.hands],
            "draw": [card.id for card in self.draw],
            "discard": [card.id for card in self.discard],
            "active_suit": self.active_suit.value,
            "pending_sevens": self.pending_sevens,
            "ace_active": self.ace_active,
            "suit_named": self.suit_named,
            "current": self.current,
            "consecutive_passes": self.consecutive_passes,
            "result": None if self.result is None else dict(self.result),
            "actions": list(self.actions),
        }

    def _dont_play_info(self) -> dict:
        if self.pending_sevens > 0:
            actual = min(self.pending_sevens * 2, self.available_to_draw())
            if actual <= 0:
                return {"kind": "pass", "label": "Pass"}
            noun = "card" if actual == 1 else "cards"
            return {"kind": "penalty", "label": f"Take {actual} {noun}"}
        if self.ace_active:
            return {"kind": "stand", "label": "Stand"}
        if self.available_to_draw() > 0:
            return {"kind": "draw", "label": "Draw a card"}
        return {"kind": "pass", "label": "Pass"}

    def _play(self, action: int) -> None:
        player = self.current
        card, named = action_to_play(action)
        self._remove(player, card)
        self.discard.append(card)
        self.played.append(card)
        event: dict = {
            "kind": "play",
            "player": player,
            "card": card.to_dict(),
            "suit": None,
            "suit_name": None,
        }
        if named is not None:
            event["suit"] = named.key
            event["suit_name"] = named.label
        self.log.append(event)
        if not self.hands[player]:
            # The last card wins outright. A 7 or Ace on it never takes effect.
            if named is not None:
                self.active_suit = named
            else:
                self.active_suit = card.suit
            self.pending_sevens = 0
            self.ace_active = False
            self.suit_named = named is not None
            self._set_result(player, "empty_hand")
            return
        if card.rank is Rank.SEVEN:
            self.pending_sevens += 1
            self.ace_active = False
            self.suit_named = False
            self.active_suit = card.suit
        elif card.rank is Rank.ACE:
            self.pending_sevens = 0
            self.ace_active = True
            self.suit_named = False
            self.active_suit = card.suit
        elif card.rank is Rank.OVER:
            self.pending_sevens = 0
            self.ace_active = False
            self.suit_named = True
            self.active_suit = named
        else:
            self.pending_sevens = 0
            self.ace_active = False
            self.suit_named = False
            self.active_suit = card.suit
        self.consecutive_passes = 0
        self._advance()

    def _dont_play(self) -> None:
        player = self.current
        if self.pending_sevens > 0:
            owed = self.pending_sevens * 2
            got = self._draw_into(player, owed)
            self.pending_sevens = 0
            if got == 0:
                self._pass(player, forgiven=True)
            else:
                self.consecutive_passes = 0
                self.log.append(
                    {"kind": "draw", "player": player, "count": got, "penalty": True}
                )
                self._reshuffle_if_needed()
                self._advance()
            return
        if self.ace_active:
            self.ace_active = False
            self.consecutive_passes = 0
            self.log.append({"kind": "stand", "player": player})
            self._advance()
            return
        if self.available_to_draw() > 0:
            got = self._draw_into(player, 1)
            self.consecutive_passes = 0
            self.log.append(
                {"kind": "draw", "player": player, "count": got, "penalty": False}
            )
            self._reshuffle_if_needed()
            self._advance()
            return
        self._pass(player, forgiven=False)

    def _draw_into(self, player: int, count: int) -> int:
        got = 0
        for _ in range(count):
            self._reshuffle_if_needed()
            if not self.draw:
                break
            card = self.draw.pop()
            self.hands[player].append(card)
            self.drawn[player].append(card.id)
            got += 1
        return got

    def _reshuffle_if_needed(self) -> None:
        if self.draw or len(self.discard) <= 1:
            return
        top = self.discard[-1]
        rest = self.discard[:-1]
        shuffle(self.rng, rest)
        self.discard = [top]
        self.draw = rest
        self.log.append({"kind": "shuffle", "player": None})

    def _pass(self, player: int, *, forgiven: bool) -> None:
        self.consecutive_passes += 1
        event: dict = {"kind": "pass", "player": player}
        if forgiven:
            event["forgiven"] = True
        self.log.append(event)
        if self.consecutive_passes >= 2:
            self._finish_stuck()
            return
        self._advance()

    def _advance(self) -> None:
        self.current = 1 - self.current

    def _finish_stuck(self) -> None:
        sizes = (len(self.hands[0]), len(self.hands[1]))
        if sizes[0] < sizes[1]:
            winner: int | None = 0
        elif sizes[1] < sizes[0]:
            winner = 1
        else:
            winner = None
        self._set_result(winner, "stuck")

    def concede(self, loser: int) -> None:
        """``loser`` gives up the hand. The other seat wins. A finished hand stays as it is."""
        if self.result is not None:
            return
        if loser not in (0, 1):
            raise ValueError("loser must be 0 or 1")
        self._set_result(1 - loser, "resign")

    def _set_result(self, winner: int | None, reason: str) -> None:
        self.result = {"winner": winner, "reason": reason}
        self.log.append({"kind": "game_over", "winner": winner, "reason": reason})

    def _remove(self, player: int, card: Card) -> None:
        hand = self.hands[player]
        for index, held in enumerate(hand):
            if held == card:
                del hand[index]
                return
        raise IllegalAction("You don't have that card.")

    def __repr__(self) -> str:
        top = self.discard[-1].name if self.discard else "-"
        return (
            f"Game(seed={self.seed}, turn={self.current}, top={top}, "
            f"suit={self.active_suit.label}, sevens={self.pending_sevens}, "
            f"ace={self.ace_active}, done={self.result is not None})"
        )
