"""Rules the two players agreed on. Positions are built by hand so each case is obvious."""

import json
import random
import subprocess
import sys

import pytest

from prsi.actions import (
    DONT_PLAY,
    NUM_ACTIONS,
    action_to_play,
    play_card_action,
    play_over_action,
    short_label,
)
from prsi.cards import Card, Rank, Suit, full_deck, shuffle
from prsi.game import OPENING_HAND, Game, IllegalAction
from prsi.record import format_event, load_record, opening_hand_of, replay, save_record

H, L, A, B = Suit.HEARTS, Suit.LEAVES, Suit.ACORNS, Suit.BELLS
S7, S8, S9, S10 = Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN
UN, OV, KI, AC = Rank.UNDER, Rank.OVER, Rank.KING, Rank.ACE

VIEW_KEYS = {
    "you",
    "your_turn",
    "hand",
    "top_card",
    "under_cards",
    "active_suit",
    "active_suit_name",
    "penalty_cards",
    "ace_active",
    "named_by_over",
    "played",
    "opponent_hand_size",
    "draw_pile_size",
    "discard_size",
    "current",
    "legal_actions",
    "dont_play",
    "log",
    "drawn",
    "result",
}


def C(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


def test_action_numbers_are_fixed():
    numbers = [play_card_action(card) for card in full_deck() if card.rank is not OV]
    numbers += [
        play_over_action(card, suit)
        for card in full_deck()
        if card.rank is OV
        for suit in Suit
    ]
    numbers.append(DONT_PLAY)
    assert sorted(numbers) == list(range(NUM_ACTIONS))
    assert NUM_ACTIONS == 45
    assert DONT_PLAY == 44
    assert play_card_action(C(H, S7)) == 0
    assert play_card_action(C(B, AC)) == 27
    assert play_over_action(C(H, OV), H) == 28
    assert play_over_action(C(B, OV), B) == 43
    card, named = action_to_play(0)
    assert card == C(H, S7) and named is None
    card, named = action_to_play(43)
    assert card == C(B, OV) and named is B


def test_short_labels_for_the_spectator_panel():
    assert short_label(play_card_action(C(A, S7))) == "a7"
    assert short_label(play_card_action(C(A, S8))) == "a8"
    assert short_label(play_card_action(C(A, S9))) == "a9"
    assert short_label(play_card_action(C(A, S10))) == "a10"
    assert short_label(play_card_action(C(H, UN))) == "hU"
    assert short_label(play_card_action(C(L, KI))) == "lK"
    assert short_label(play_card_action(C(B, AC))) == "bA"
    assert short_label(play_over_action(C(A, OV), H)) == "aO=h"
    assert short_label(DONT_PLAY, {"kind": "draw", "label": "Draw a card"}) == "D"
    assert short_label(DONT_PLAY, {"kind": "stand", "label": "Stand"}) == "S"
    assert short_label(DONT_PLAY, {"kind": "penalty", "label": "Take 1 card"}) == "T1"
    assert short_label(DONT_PLAY, {"kind": "penalty", "label": "Take 2 cards"}) == "T2"
    assert short_label(DONT_PLAY, {"kind": "penalty", "label": "Take 4 cards"}) == "T4"
    assert short_label(DONT_PLAY, {"kind": "penalty", "label": "Take 6 cards"}) == "T6"
    assert short_label(DONT_PLAY, {"kind": "pass", "label": "Pass"}) == "P"


def test_deal_follows_the_seed():
    game = Game(42, 1)
    rng = random.Random(42)
    deck = full_deck()
    shuffle(rng, deck)
    assert game.hands[0] == deck[0:6]
    assert game.hands[1] == deck[6:12]
    assert game.discard == [deck[12]]
    assert game.draw == deck[13:][::-1]
    assert game.current == 1
    assert game.pending_sevens == 0
    assert game.ace_active is False
    assert game.suit_named is False
    assert game.active_suit is deck[12].suit
    assert len(game.draw) == 19
    assert game.opening_hand == 6


def test_opening_card_has_no_effect():
    saw = {"7": False, "ace": False, "over": False}
    for seed in range(300):
        game = Game(seed, 0)
        top = game.discard[-1]
        assert game.pending_sevens == 0
        assert game.ace_active is False
        assert game.suit_named is False
        assert game.active_suit is top.suit
        assert game.observation(game.current)["named_by_over"] is False
        if top.rank is S7:
            saw["7"] = True
        elif top.rank is AC:
            saw["ace"] = True
        elif top.rank is OV:
            saw["over"] = True
    assert saw == {"7": True, "ace": True, "over": True}


def test_different_seeds_deal_differently():
    assert Game(0, 0).debug_state()["hands"] != Game(42, 0).debug_state()["hands"]


def test_match_by_suit_by_rank_and_over_on_anything():
    by_suit = C(H, KI)
    by_rank = C(B, S9)
    dead = C(L, S8)
    over = C(A, OV)
    game = Game.from_position(
        hands=[[by_suit, by_rank, dead, over], [C(L, KI)]],
        discard=[C(H, S9)],
        draw=[C(B, KI)],
        current_player=0,
    )
    legal = set(game.legal_actions())
    assert play_card_action(by_suit) in legal
    assert play_card_action(by_rank) in legal
    assert play_card_action(dead) not in legal
    assert all(play_over_action(over, suit) in legal for suit in Suit)
    assert DONT_PLAY in legal
    assert game.observation(0)["dont_play"]["label"] == "Draw a card"
    # Holding a playable card does not hide the opponent's lack of information.
    assert game.observation(1)["dont_play"]["label"] is None
    assert game.observation(1)["legal_actions"] == []


def test_only_one_card_per_turn():
    game = Game.from_position(
        hands=[[C(H, S8), C(L, S8)], [C(A, KI)]],
        discard=[C(B, S8)],
        draw=[C(H, KI)],
        current_player=0,
    )
    game.step(play_card_action(C(H, S8)))
    assert game.current == 1
    assert C(L, S8) in game.hands[0]


def test_under_and_king_have_no_special_power():
    game = Game.from_position(
        hands=[[C(H, UN), C(B, KI)], [C(L, S8)]],
        discard=[C(H, S9)],
        draw=[C(A, S10)],
        current_player=0,
    )
    game.step(play_card_action(C(H, UN)))
    assert game.pending_sevens == 0
    assert game.ace_active is False
    assert game.suit_named is False
    assert game.active_suit is H
    assert game.current == 1


def test_over_names_the_suit_and_can_follow_another_over():
    over = C(H, OV)
    leaves_seven = C(L, S7)
    old_suit = C(A, S9)
    another_over = C(B, OV)
    own_suit = C(H, S10)
    game = Game.from_position(
        hands=[[over, C(B, S8)], [leaves_seven, old_suit, another_over, own_suit]],
        discard=[C(A, S10)],
        draw=[C(B, KI), C(L, KI)],
        current_player=0,
    )
    game.step(play_over_action(over, L))
    assert game.active_suit is L
    assert game.suit_named is True
    assert game.discard[-1] == over
    assert game.current == 1
    assert game.observation(1)["named_by_over"] is True
    legal = set(game.legal_actions())
    assert play_card_action(leaves_seven) in legal
    assert play_card_action(old_suit) not in legal
    assert play_card_action(own_suit) not in legal
    assert all(play_over_action(another_over, suit) in legal for suit in Suit)

    game.step(play_over_action(another_over, A))
    assert game.active_suit is A
    assert game.suit_named is True
    assert game.pending_sevens == 0
    assert game.ace_active is False


def test_over_cannot_answer_a_seven_or_an_ace():
    game = Game.from_position(
        hands=[[C(H, S7), C(A, KI)], [C(L, S7), C(B, OV), C(H, KI)]],
        discard=[C(H, S9)],
        draw=[C(B, S10), C(L, S10)],
        current_player=0,
    )
    game.step(play_card_action(C(H, S7)))
    legal = set(game.legal_actions())
    assert legal == {play_card_action(C(L, S7)), DONT_PLAY}
    assert game.observation(1)["dont_play"]["label"] == "Take 2 cards"

    ace = Game.from_position(
        hands=[[C(H, AC), C(H, S8)], [C(L, AC), C(B, OV), C(L, S8)]],
        discard=[C(H, S9)],
        draw=[C(A, KI)],
        current_player=0,
    )
    ace.step(play_card_action(C(H, AC)))
    legal = set(ace.legal_actions())
    assert play_card_action(C(L, AC)) in legal
    assert DONT_PLAY in legal
    assert play_over_action(C(B, OV), H) not in legal
    assert play_card_action(C(L, S8)) not in legal
    assert ace.observation(1)["dont_play"]["label"] == "Stand"


def test_seven_chain_reaches_eight_and_then_behaves_as_a_normal_card():
    h7, l7, a7, b7 = C(H, S7), C(L, S7), C(A, S7), C(B, S7)
    filler = C(H, S8)
    leaves = C(L, S8)
    bells = C(B, S9)
    over = C(H, OV)
    top = C(H, S9)
    used = {h7, l7, a7, b7, filler, leaves, bells, over, top}
    draw = [card for card in full_deck() if card not in used][:10]
    game = Game.from_position(
        hands=[[h7, a7, filler], [l7, b7, leaves, bells, over]],
        discard=[top],
        draw=draw,
        current_player=0,
    )
    game.step(play_card_action(h7))
    assert game.observation(1)["penalty_cards"] == 2
    assert set(game.legal_actions()) == {play_card_action(l7), play_card_action(b7), DONT_PLAY}

    game.step(play_card_action(l7))
    assert game.observation(0)["penalty_cards"] == 4
    game.step(play_card_action(a7))
    assert game.observation(1)["penalty_cards"] == 6
    game.step(play_card_action(b7))
    assert game.observation(0)["penalty_cards"] == 8
    assert game.current == 0
    assert game.legal_actions() == [DONT_PLAY]
    assert game.observation(0)["dont_play"]["label"] == "Take 8 cards"

    game.step(DONT_PLAY)
    assert game.pending_sevens == 0
    assert game.ace_active is False
    assert len(game.hands[0]) == 9
    assert game.current == 1
    assert game.consecutive_passes == 0
    assert game.discard[-1] == b7
    legal = set(game.legal_actions())
    assert play_card_action(bells) in legal
    assert play_card_action(leaves) not in legal
    assert all(play_over_action(over, suit) in legal for suit in Suit)
    assert DONT_PLAY in legal
    assert not game.is_over


def test_may_take_a_seven_penalty_while_holding_a_seven():
    game = Game.from_position(
        hands=[[C(H, S7), C(H, S8)], [C(L, S7), C(B, S7), C(L, S8)]],
        discard=[C(H, S9)],
        draw=[C(A, KI), C(B, KI), C(L, KI), C(A, S10)],
        current_player=0,
    )
    game.step(play_card_action(C(H, S7)))
    assert play_card_action(C(L, S7)) in game.legal_actions()
    assert DONT_PLAY in game.legal_actions()
    game.step(DONT_PLAY)
    assert game.pending_sevens == 0
    assert C(L, S7) in game.hands[1]
    assert len(game.hands[1]) == 5
    assert game.current == 0
    assert game.consecutive_passes == 0
    # The seven is used up: a matching non-seven is playable, and so is an Over.
    assert game.ace_active is False


def test_zero_card_penalty_is_still_allowed_and_counts_as_a_pass():
    seven = C(H, S7)
    game = Game.from_position(
        hands=[[seven, C(B, KI)], [C(A, S8)]],
        discard=[C(L, S7)],
        draw=[],
        pending_sevens=1,
        current_player=0,
    )
    assert play_card_action(seven) in game.legal_actions()
    assert DONT_PLAY in game.legal_actions()
    assert game.observation(0)["dont_play"]["label"] == "Pass"
    game.step(DONT_PLAY)
    assert game.pending_sevens == 0
    assert len(game.hands[0]) == 2
    assert game.consecutive_passes == 1
    assert game.log[-1]["kind"] == "pass"
    assert game.log[-1]["forgiven"] is True
    assert not game.is_over
    assert game.legal_actions() == [DONT_PLAY]
    game.step(DONT_PLAY)
    assert game.is_over
    assert game.result == {"winner": 1, "reason": "stuck"}


def test_standing_on_an_ace_returns_the_turn_and_uses_the_ace_up():
    game = Game.from_position(
        hands=[[C(H, AC), C(H, S8), C(L, S9)], [C(L, AC), C(L, S8)]],
        discard=[C(H, S9)],
        draw=[C(B, KI), C(B, S10)],
        current_player=0,
    )
    game.step(play_card_action(C(H, AC)))
    assert game.ace_active is True
    assert game.current == 1
    assert set(game.legal_actions()) == {play_card_action(C(L, AC)), DONT_PLAY}
    assert game.observation(0)["legal_actions"] == []

    game.step(DONT_PLAY)
    assert game.ace_active is False
    assert game.consecutive_passes == 0
    assert game.current == 0
    assert play_card_action(C(H, S8)) in game.legal_actions()
    assert play_card_action(C(L, S9)) not in game.legal_actions()
    assert game.observation(0)["dont_play"]["label"] == "Draw a card"
    assert game.observation(0)["penalty_cards"] == 0


def test_ace_chain_then_stand():
    game = Game.from_position(
        hands=[[C(H, AC), C(H, S8), C(L, S9)], [C(L, AC), C(L, S8)]],
        discard=[C(H, S9)],
        draw=[C(B, KI), C(B, S10)],
        current_player=0,
    )
    game.step(play_card_action(C(H, AC)))
    game.step(play_card_action(C(L, AC)))
    assert game.ace_active is True
    assert game.current == 0
    assert game.legal_actions() == [DONT_PLAY]
    assert game.observation(0)["dont_play"]["label"] == "Stand"
    game.step(DONT_PLAY)
    assert game.ace_active is False
    assert game.current == 1
    assert play_card_action(C(L, S8)) in game.legal_actions()
    assert DONT_PLAY in game.legal_actions()


def test_drawing_is_allowed_while_holding_a_playable_card_and_stays_secret():
    secret = C(B, AC)
    playable = C(H, KI)
    game = Game.from_position(
        hands=[[playable], [C(L, S8)]],
        discard=[C(H, S9)],
        draw=[C(A, S10), secret],
        current_player=0,
    )
    assert play_card_action(playable) in game.legal_actions()
    assert DONT_PLAY in game.legal_actions()
    game.step(DONT_PLAY)
    assert game.current == 1
    assert secret in game.hands[0]
    assert playable in game.hands[0]
    assert secret.name in json.dumps(game.observation(0))
    assert secret.name not in json.dumps(game.observation(1))


def test_must_play_when_nothing_can_be_drawn():
    playable = C(H, KI)
    dead = C(L, S8)
    game = Game.from_position(
        hands=[[playable, dead], [C(A, S10)]],
        discard=[C(H, S9)],
        draw=[],
        current_player=0,
    )
    legal = set(game.legal_actions())
    assert play_card_action(playable) in legal
    assert play_card_action(dead) not in legal
    assert DONT_PLAY not in legal
    assert game.observation(0)["dont_play"]["legal"] is False


def test_two_passes_with_nothing_to_draw_end_the_game():
    fewer = Game.from_position(
        hands=[[C(L, S8)], [C(A, S10), C(B, UN), C(A, KI)]],
        discard=[C(H, S9)],
        draw=[],
        current_player=0,
    )
    assert fewer.observation(0)["dont_play"]["label"] == "Pass"
    fewer.step(DONT_PLAY)
    assert not fewer.is_over
    assert fewer.consecutive_passes == 1
    fewer.step(DONT_PLAY)
    assert fewer.result == {"winner": 0, "reason": "stuck"}

    drawn = Game.from_position(
        hands=[[C(L, S8)], [C(A, S10)]],
        discard=[C(H, S9)],
        draw=[],
        current_player=0,
    )
    drawn.step(DONT_PLAY)
    drawn.step(DONT_PLAY)
    assert drawn.result == {"winner": None, "reason": "stuck"}


def test_a_play_resets_the_pass_count():
    game = Game.from_position(
        hands=[[C(L, S8)], [C(H, S10), C(A, KI)]],
        discard=[C(H, S9)],
        draw=[],
        current_player=0,
    )
    game.step(DONT_PLAY)
    assert game.consecutive_passes == 1
    assert not game.is_over
    game.step(play_card_action(C(H, S10)))
    assert game.consecutive_passes == 0
    assert not game.is_over
    # Playing buried the old top, so it can be drawn. The pass streak is gone.
    assert game.current == 0
    assert DONT_PLAY in game.legal_actions()
    assert game.observation(0)["dont_play"]["label"] == "Draw a card"


def test_reshuffle_keeps_the_top_card_and_follows_the_seed():
    under = [C(H, S8), C(L, S8), C(A, S8), C(B, S8)]
    top = C(H, S9)

    def make(seed: int) -> Game:
        return Game.from_position(
            hands=[[C(H, KI)], [C(L, KI)]],
            discard=[*under, top],
            draw=[],
            seed=seed,
            current_player=0,
        )

    game = make(99)
    game.step(DONT_PLAY)
    assert game.discard == [top]
    drawn = game.hands[0][-1]
    assert drawn in under
    assert set(game.draw) | {drawn} == set(under)
    assert len(game.draw) == 3
    assert any(event["kind"] == "shuffle" for event in game.log)
    again = make(99)
    again.step(DONT_PLAY)
    assert again.debug_state() == game.debug_state()
    assert again.log == game.log

    orders = set()
    for seed in range(99, 110):
        other = make(seed)
        other.step(DONT_PLAY)
        orders.add(tuple(card.id for card in other.draw))
    assert len(orders) > 1


def test_penalty_draws_the_pile_before_reshuffling():
    first = C(B, S10)
    under = [C(H, S8), C(L, S8), C(A, S8), C(B, S8), C(H, S10)]
    top = C(L, S7)

    def make() -> Game:
        return Game.from_position(
            hands=[[C(A, KI)], [C(B, KI)]],
            discard=[*under, top],
            draw=[first],
            pending_sevens=2,
            seed=0,
            current_player=0,
        )

    game = make()
    assert game.observation(0)["dont_play"]["label"] == "Take 4 cards"
    game.step(DONT_PLAY)
    assert game.hands[0][0] == C(A, KI)
    assert game.hands[0][1] == first
    assert len(game.hands[0]) == 5
    assert game.discard == [top]
    assert len(game.draw) == 2
    assert set(game.hands[0][2:]) | set(game.draw) == set(under)
    assert game.pending_sevens == 0
    other = make()
    other.step(DONT_PLAY)
    assert other.debug_state() == game.debug_state()


def test_penalty_larger_than_the_pack_is_forgiven():
    top = C(H, S7)
    game = Game.from_position(
        hands=[[C(L, KI)], [C(A, KI)]],
        discard=[C(B, S9), top],
        draw=[C(B, S10)],
        pending_sevens=3,
        current_player=0,
    )
    assert game.available_to_draw() == 2
    assert game.observation(0)["dont_play"]["label"] == "Take 2 cards"
    game.step(DONT_PLAY)
    assert len(game.hands[0]) == 3
    assert game.pending_sevens == 0
    assert game.consecutive_passes == 0
    assert game.discard == [top]
    assert game.draw == []
    assert game.current == 1
    assert not game.is_over


def test_penalty_label_uses_singular_for_one_card():
    game = Game.from_position(
        hands=[[C(L, KI)], [C(A, KI)]],
        discard=[C(H, S7)],
        draw=[C(B, S10)],
        pending_sevens=2,
        current_player=0,
    )
    assert game.observation(0)["dont_play"]["label"] == "Take 1 card"


@pytest.mark.parametrize("rank", [S7, AC, KI])
def test_last_card_wins_immediately_even_when_it_is_special(rank):
    card = C(H, rank)
    opponent = [C(L, S8), C(A, S9), C(B, S10)]
    draw = [C(A, S8), C(B, S9), C(L, S10), C(H, S8)]
    game = Game.from_position(
        hands=[[card], list(opponent)],
        discard=[C(H, S9)],
        draw=list(draw),
        current_player=0,
    )
    game.step(play_card_action(card))
    assert game.result == {"winner": 0, "reason": "empty_hand"}
    assert game.hands[1] == opponent
    assert game.draw == draw
    assert game.pending_sevens == 0
    assert game.ace_active is False
    assert game.legal_actions() == []
    with pytest.raises(IllegalAction):
        game.step(DONT_PLAY)


def test_inspecting_a_position_does_not_change_it():
    game = Game.from_position(
        hands=[[C(H, KI)], [C(L, KI)]],
        discard=[C(H, S8), C(L, S8), C(A, S9)],
        draw=[],
        seed=0,
    )
    before = game.debug_state()
    logged = list(game.log)
    for _ in range(4):
        game.legal_actions()
        game.observation(0)
        game.observation(1)
    assert game.debug_state() == before
    assert game.log == logged


def test_same_seed_replays_exactly_even_if_someone_looks():
    watched = Game(42, 1)
    blind = Game(42, 1)
    assert watched.debug_state() == blind.debug_state()
    rng = random.Random(42)
    steps = 0
    while not watched.is_over:
        watched.observation(0)
        watched.observation(1)
        watched.legal_actions()
        action = rng.choice(watched.legal_actions())
        watched.step(action)
        blind.step(action)
        steps += 1
        if steps > 5000:
            raise AssertionError("game did not finish")
    assert watched.debug_state() == blind.debug_state()
    assert watched.log == blind.log


def test_illegal_action_changes_nothing():
    game = Game(42, 0)
    before = game.debug_state()
    legal = set(game.legal_actions())
    bad = next(action for action in range(NUM_ACTIONS) if action not in legal)
    with pytest.raises(IllegalAction):
        game.step(bad)
    with pytest.raises(IllegalAction):
        game.step(True)  # type: ignore[arg-type]
    assert game.debug_state() == before


def test_observation_shows_only_what_that_player_may_know():
    game = Game(42, 0)
    for viewer in (0, 1):
        obs = game.observation(viewer)
        assert set(obs) == VIEW_KEYS
        hidden = {card.name for card in game.hands[1 - viewer]}
        hidden -= {card.name for card in game.hands[viewer]}
        hidden.discard(game.discard[-1].name)
        blob = json.dumps(obs)
        for name in hidden:
            assert name not in blob
        assert json.dumps([card.id for card in game.draw]) not in blob
        assert obs["opponent_hand_size"] == len(game.hands[1 - viewer])
        assert obs["draw_pile_size"] == len(game.draw)
        assert obs["your_turn"] is (game.current == viewer)
        if obs["your_turn"]:
            assert obs["legal_actions"] == game.legal_actions()
            covered = {play["id"] for card in obs["hand"] for play in card["actions"]}
            for action in obs["legal_actions"]:
                if action == DONT_PLAY:
                    assert obs["dont_play"]["legal"] is True
                else:
                    assert action in covered
        else:
            assert obs["legal_actions"] == []
            assert obs["dont_play"] == {
                "id": DONT_PLAY,
                "kind": None,
                "label": None,
                "legal": False,
            }


def test_under_cards_are_the_two_beneath_the_top():
    game = Game.from_position(
        hands=[[C(H, S7)], [C(L, S8)]],
        discard=[C(A, S9), C(B, S10), C(H, KI)],
    )
    assert [card["id"] for card in game.observation(0)["under_cards"]] == [
        C(A, S9).id,
        C(B, S10).id,
    ]
    thin = Game.from_position(
        hands=[[C(H, S7)], [C(L, S8)]],
        discard=[C(H, KI)],
    )
    assert thin.observation(0)["under_cards"] == []


def test_from_position_rejects_a_duplicated_card():
    duplicated = C(H, S7)
    with pytest.raises(ValueError):
        Game.from_position(hands=[[duplicated], [C(L, S8)]], discard=[duplicated])


def test_record_roundtrip_and_cli(tmp_path):
    rng = random.Random(0)
    game = Game(42, 0)
    steps = 0
    while not game.is_over:
        game.observation(0)
        game.observation(1)
        game.step(rng.choice(game.legal_actions()))
        steps += 1
        if steps > 5000:
            raise AssertionError("game did not finish")
    path = tmp_path / "game.json"
    save_record(game, path, names=["Ada", "Bea"])
    record = load_record(path)
    clone = replay(record)
    assert clone.debug_state() == game.debug_state()
    assert clone.log == game.log
    assert record["names"] == ["Ada", "Bea"]
    assert record["opening_hand"] == OPENING_HAND == 6
    assert format_event(game.log[0], record["names"]).startswith("Opening card:")

    completed = subprocess.run(
        [sys.executable, "-m", "prsi.replay", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Opening card:" in completed.stdout
    assert "Moves:" in completed.stdout


def test_record_without_opening_hand_replays_a_four_card_deal():
    record = {"version": 2, "seed": 42, "starting_player": 1, "actions": []}
    assert opening_hand_of(record) == 4
    dealt = replay(record)
    four = Game(42, 1, opening_hand=4)
    assert [len(hand) for hand in dealt.hands] == [4, 4]
    assert dealt.hands == four.hands
    assert dealt.discard == four.discard
    assert dealt.draw == four.draw
    assert len(dealt.draw) == 23


def test_replay_cli_rejects_a_missing_file():
    completed = subprocess.run(
        [sys.executable, "-m", "prsi.replay", "no-such-game.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
