"""The table: two humans, a secret hand, a rematch, and a saved game."""

import asyncio
import json

from fastapi.testclient import TestClient

from prsi.cards import full_deck
from prsi.record import load_record, replay
from server.app import create_app
from server.room import Room


class FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


def _public_names(view):
    names = {card["name"] for card in view["hand"]}
    if view["top_card"]:
        names.add(view["top_card"]["name"])
    names.update(card["name"] for card in view["played"])
    for event in view["log"]:
        card = event.get("card")
        if card:
            names.add(card["name"])
    return names


def _assert_no_secrets(messages):
    deck_names = [card.name for card in full_deck()]
    for message in messages:
        if message.get("type") != "state" or message.get("phase") == "waiting":
            continue
        public = _public_names(message["view"])
        blob = json.dumps(message)
        leaked = [name for name in deck_names if name in blob and name not in public]
        assert leaked == []


async def _pump(room, ada, bea):
    for _ in range(20000):
        if room.phase == "over":
            return
        game = room.game
        if room.phase == "playing" and game is not None and not game.is_over:
            player = room.seats[game.current]
            if player is not None and player.expected_token:
                player.submit(game.legal_actions()[0], player.expected_token)
        await asyncio.sleep(0)
    _assert_no_secrets(ada.messages)
    _assert_no_secrets(bea.messages)
    raise AssertionError("the game never finished")


async def _scenario(tmp_path):
    room = Room("TEST01", tmp_path)
    ada_ws, bea_ws = FakeSocket(), FakeSocket()
    ada, err = await room.join("Ada", ada_ws)
    assert err is None and ada is not None
    full, err = await room.join("Bea", bea_ws)
    assert err is None and full is not None
    extra, err = await room.join("Cara", FakeSocket())
    assert extra is None and err == "Room is full."
    again, err = await room.rejoin(ada.token, FakeSocket())
    assert again is ada and err is None
    missing, err = await room.rejoin("not-a-token", FakeSocket())
    assert missing is None and err

    await _pump(room, ada_ws, bea_ws)
    assert room.phase == "over"
    assert room.game is not None and room.game.is_over
    assert room.wins[0] + room.wins[1] + room.draws == 1
    first = room.game
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1
    record = load_record(saved[0])
    assert replay(record).debug_state() == first.debug_state()
    assert record["names"] == ["Ada", "Bea"]

    await room.vote_rematch(0)
    await room.vote_rematch(1)
    for _ in range(1000):
        await asyncio.sleep(0)
        if room.game is not first and room.phase == "playing":
            break
    else:
        raise AssertionError("rematch did not start")
    assert room.game.starting_player == 1 - first.starting_player
    assert room.wins[0] + room.wins[1] + room.draws == 1

    await _pump(room, ada_ws, bea_ws)
    assert room.wins[0] + room.wins[1] + room.draws == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_two_players_finish_rematch_and_save(tmp_path):
    asyncio.run(asyncio.wait_for(_scenario(tmp_path), timeout=30))


def test_browser_entry_and_wrong_code(tmp_path):
    room = Room("TEST01", tmp_path)
    with TestClient(create_app(room)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Room code" in page.text
        assert "Reset score" in page.text
        assert "Resign" in page.text
        assert "app.js" in page.text
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "join", "code": "nope", "name": "Ada"})
            message = socket.receive_json()
            assert message["type"] == "error"
            assert "Wrong room code" in message["message"]


def _toasts(ws):
    return [message["message"] for message in ws.messages if message.get("type") == "toast"]


async def _until(predicate, label):
    for _ in range(5000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(label)


async def _stop(room):
    task = room._task
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _reset_and_resign(tmp_path):
    room = Room("TEST01", tmp_path, host_token="secret")
    ada_ws, bea_ws = FakeSocket(), FakeSocket()
    ada, err = await room.join("Ada", ada_ws, "secret")
    assert err is None and ada is not None and ada.is_host
    assert await room.reset_score(ada) == "The game has not started."
    bea, err = await room.join("Bea", bea_ws)
    assert err is None and bea is not None
    assert await room.reset_score(bea) == "Only the host can reset the score."

    await _until(lambda: room.phase == "playing" and room.game is not None, "the hand did not start")
    first = room.game
    starter = first.starting_player
    first_seed = next(message["seed"] for message in reversed(ada_ws.messages) if message.get("phase") == "playing")
    assert first_seed == str(first.seed)
    assert await room.reset_score(ada) is None
    await _until(
        lambda: room.phase == "playing" and room.game is not None and room.game is not first,
        "reset did not deal",
    )
    assert room.wins == [0, 0] and room.draws == 0
    assert room.game.starting_player == 1 - starter
    dealt_seed = next(message["seed"] for message in reversed(ada_ws.messages) if message.get("phase") == "playing")
    assert dealt_seed == str(room.game.seed) and dealt_seed != first_seed
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1
    reset_record = load_record(saved[0])
    assert reset_record["result"] == {"winner": None, "reason": "reset"}
    assert replay(reset_record).result is None
    await _until(lambda: _toasts(bea_ws) == ["The host reset the score and dealt a new hand."], "Bea got no reset toast")
    assert _toasts(ada_ws) == []

    await _pump(room, ada_ws, bea_ws)
    assert room.wins[0] + room.wins[1] + room.draws == 1
    finished = set(tmp_path.glob("*.json"))
    assert len(finished) == 2
    follow = room.next_starter
    assert await room.reset_score(ada) is None
    await _until(
        lambda: _toasts(bea_ws).count("The host reset the score and dealt a new hand.") == 2
        and room.phase == "playing"
        and room.wins == [0, 0]
        and room.draws == 0,
        "the score was not cleared",
    )
    assert set(tmp_path.glob("*.json")) == finished
    assert room.game.starting_player == follow

    before_resign = set(tmp_path.glob("*.json"))
    await room.resign(bea)
    await _until(lambda: room.phase == "over", "resign did not finish the hand")
    assert room.wins[ada.seat] == 1
    assert room.wins[bea.seat] == 0
    assert room.game is not None and room.game.result == {"winner": ada.seat, "reason": "resign"}
    assert await room.resign(ada) == "The hand is already over."
    added = set(tmp_path.glob("*.json")) - before_resign
    assert len(added) == 1
    resign_record = load_record(added.pop())
    assert resign_record["result"] == {"winner": ada.seat, "reason": "resign"}
    await _until(lambda: _toasts(ada_ws) == ["Bea resigned."], "Ada was not told about the resignation")
    assert "Bea resigned." not in _toasts(bea_ws)
    await _stop(room)


def test_reset_score_and_resign(tmp_path):
    asyncio.run(asyncio.wait_for(_reset_and_resign(tmp_path), timeout=30))
