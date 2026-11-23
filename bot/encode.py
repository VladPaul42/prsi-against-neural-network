"""Turn what one seat knows into the network's input vector.

Everything comes from a ``PublicState``: the seat's own hand, the public pile
sizes and flags, and the card tracker. Layout (1564 numbers):

    0    my hand                         32
    32   top card                        32
    64   suit in play                     4
    68   pending 7s (0, 1, 2, 3, 4+)      5
    73   Ace active, suit named by an Over, opponent passed last
    76   my hand size, opponent hand size, draw pile, discard pile (scaled)
    80   share of pool cards in the opponent's hand, that count scaled
    82   32 cards x {mine, top, under top, opponent for sure, pool, excluded}
    274  last 30 moves, newest first, 43 each:
         mine / theirs (2), play / draw / stand / pass (4), card (32),
         named suit (4), cards drawn (1)
"""

from __future__ import annotations

import numpy as np

from bot.fastgame import EV_DRAW, EV_PASS, EV_PLAY, EV_STAND, NUM_ACTIONS, mask_cards
from bot.tracker import HISTORY

OFF_HAND = 0
OFF_TOP = 32
OFF_SUIT = 64
OFF_PENDING = 68
OFF_ACE = 73
OFF_NAMED = 74
OFF_PASSED = 75
OFF_SIZES = 76
OFF_POOL = 80
OFF_LOC = 82
LOC_WIDTH = 6
OFF_HIST = OFF_LOC + 32 * LOC_WIDTH
HIST_WIDTH = 2 + 4 + 32 + 4 + 1
INPUT_SIZE = OFF_HIST + HISTORY * HIST_WIDTH

_KIND_SLOT = {EV_PLAY: 0, EV_DRAW: 1, EV_STAND: 2, EV_PASS: 3}


def encode_parts(seat, tracker, pending, ace, suit_named) -> tuple[list[int], list[tuple[int, float]]]:
    """Indices set to 1, and (index, value) pairs for the scaled numbers."""
    ones = []
    append = ones.append
    top = tracker.discard[-1]
    for card in mask_cards(tracker.my_hand):
        append(OFF_HAND + card)
        append(OFF_LOC + card * LOC_WIDTH)
    append(OFF_TOP + top)
    append(OFF_LOC + top * LOC_WIDTH + 1)
    for card in tracker.discard[:-1]:
        append(OFF_LOC + card * LOC_WIDTH + 2)
    unseen = tracker.unseen
    universe = tracker.universe
    excluded = tracker.excluded
    for card in mask_cards(unseen):
        if (universe >> card) & 1:
            append(OFF_LOC + card * LOC_WIDTH + 4)
            if (excluded >> card) & 1:
                append(OFF_LOC + card * LOC_WIDTH + 5)
        else:
            append(OFF_LOC + card * LOC_WIDTH + 3)
    append(OFF_SUIT + tracker.active_suit)
    append(OFF_PENDING + min(pending, 4))
    if ace:
        append(OFF_ACE)
    if suit_named:
        append(OFF_NAMED)
    if tracker.passes:
        append(OFF_PASSED)
    recent = tracker.recent
    base = OFF_HIST
    floats = []
    for kind, player, a, b in reversed(recent):
        append(base + (0 if player == seat else 1))
        append(base + 2 + _KIND_SLOT[kind])
        if kind == EV_PLAY:
            append(base + 6 + a)
            if b >= 0:
                append(base + 38 + b)
        elif kind == EV_DRAW:
            floats.append((base + 42, a / 8.0))
        base += HIST_WIDTH
    pool = tracker.pool.bit_count()
    in_opp = tracker.pool_in_opp
    floats.extend(
        (
            (OFF_SIZES, tracker.my_hand.bit_count() / 16.0),
            (OFF_SIZES + 1, tracker.opp_size / 16.0),
            (OFF_SIZES + 2, tracker.draw_size / 32.0),
            (OFF_SIZES + 3, len(tracker.discard) / 32.0),
            (OFF_POOL, in_opp / pool if pool else 0.0),
            (OFF_POOL + 1, in_opp / 16.0),
        )
    )
    return ones, floats


def encode(state, out: np.ndarray | None = None) -> np.ndarray:
    """``state`` is a ``PublicState``."""
    return encode_raw(state.seat, state.tracker, state.pending, state.ace, state.suit_named, out)


def encode_raw(seat, tracker, pending, ace, suit_named, out: np.ndarray | None = None) -> np.ndarray:
    ones, floats = encode_parts(seat, tracker, pending, ace, suit_named)
    if out is None:
        out = np.zeros(INPUT_SIZE, dtype=np.float32)
    else:
        out[:] = 0.0
    out[ones] = 1.0
    for index, value in floats:
        out[index] = value
    return out


def legal_mask(legal: list[int]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[legal] = True
    return mask
