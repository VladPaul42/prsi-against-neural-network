"""The bot at the web table: modes, privacy, responsiveness, saved records, review."""

import asyncio
import concurrent.futures
import json
import time

import pytest
from fastapi.testclient import TestClient

from prsi.cards import full_deck
from prsi.record import load_record, replay
from server import bot as server_bot
from server.app import create_app
from server.bot import BotPlayer
from server.review import analyse_game, game_frames, read_status, spec_version
from server.room import Room

FAST_BOT = {"kind": "ismcts", "iterations": 30, "label": "test-bot"}


@pytest.fixture(autouse=True, scope="module")
def _stop_workers():
    yield
    server_bot.shutdown()


class FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


def make_room(tmp_path, spec=FAST_BOT) -> Room:
    def factory(seat, on_panel):
        bot = BotPlayer(seat, spec, on_panel)
        # The live cap defaults to 100000. Tests that name an iteration budget
        # use that as the cap, so a game does not imagine a hundred thousand deals.
        if spec.get("iterations"):
            bot.search_limit = spec["iterations"]
        return bot

    return Room("TEST01", tmp_path, host_token="secret-host", bot_factory=factory,
                bot_note="test bot", bot_spec=spec)


def public_names(view):
    names = {card["name"] for card in view["hand"]}
    if view["top_card"]:
        names.add(view["top_card"]["name"])
    names.update(card["name"] for card in view["played"])
    for event in view["log"]:
        card = event.get("card")
        if card:
            names.add(card["name"])
    return names


def assert_friend_sees_only_their_seat(messages, seat):
    deck = [card.name for card in full_deck()]
    for message in messages:
        assert message.get("type") != "panel", "the friend received a bot panel"
        assert not message.get("spectating")
        assert "panel" not in message and "host" not in message
        if message.get("type") != "state" or message.get("phase") == "waiting":
            continue
        assert message["seat"] == seat
        blob = json.dumps(message)
        public = public_names(message["view"])
        assert [name for name in deck if name in blob and name not in public] == []


async def pump(room, humans, games=1, timeout=120.0):
    """Click the first legal move for every human until ``games`` games are over."""
    start = time.time()
    finished = room.wins[0] + room.wins[1] + room.draws
    while room.wins[0] + room.wins[1] + room.draws < finished + games:
        if time.time() - start > timeout:
            raise AssertionError("the game did not finish")
        game = room.game
        if room.phase == "playing" and game is not None and not game.is_over:
            player = room.seats[game.current]
            if isinstance(player, BotPlayer) and player.confirm_moves:
                host = room.host()
                if host is not None:
                    await room.confirm_move(host, room.move_id)
            elif player in humans and player.expected_token:
                player.submit(game.legal_actions()[0], player.expected_token)
        await asyncio.sleep(0.01)


async def _mode3_scenario(tmp_path):
    room = make_room(tmp_path)
    host_ws, friend_ws = FakeSocket(), FakeSocket()
    host, err = await room.join("Host", host_ws, "secret-host")
    assert err is None and host.is_host and host.seat == 0
    friend, err = await room.join("Friend", friend_ws)
    assert err is None and friend.seat == 1
    assert await room.set_mode(friend, 3) == "Only the host can change the game mode."
    assert await room.set_mode(host, 3) is None
    assert room.mode == 3
    assert isinstance(room.seats[0], BotPlayer) and room.seats[1] is friend
    assert host.seat == -1
    await pump(room, [friend], games=1)
    assert_friend_sees_only_their_seat(friend_ws.messages, 1)
    panels = [m for m in host_ws.messages if m.get("type") == "panel"]
    watching = [m for m in host_ws.messages if m.get("spectating")]
    assert panels and watching
    assert all(m["seat"] == 0 for m in watching)
    final = [m for m in panels if m["panel"]["final"]][-1]["panel"]
    assert {"win_chance", "iterations", "moves", "unseen"} <= set(final)
    # Saved with who sat where.
    saved = sorted(tmp_path.glob("*.json"))
    record = load_record(saved[-1])
    assert record["version"] == 2 and record["mode"] == 3
    assert record["seats"][0]["kind"] == "bot" and record["seats"][1] == {"kind": "human", "name": "Friend"}
    assert replay(record).result == record["result"]
    # The bot already wants a rematch; the friend decides.
    assert room.rematch[0] is True


def test_mode3_friend_never_sees_the_bot(tmp_path):
    asyncio.run(asyncio.wait_for(_mode3_scenario(tmp_path), timeout=180))


async def _mode4_and_responsiveness(tmp_path):
    room = make_room(tmp_path, {"kind": "ismcts", "seconds": 0.6, "label": "slow-bot"})
    host_ws = FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    await room.set_mode(host, 4)
    assert all(isinstance(seat, BotPlayer) for seat in room.seats)
    # While the bots think, the event loop keeps running.
    worst = 0.0
    deadline = time.time() + 4.0
    while time.time() < deadline:
        before = time.perf_counter()
        await asyncio.sleep(0.01)
        worst = max(worst, time.perf_counter() - before - 0.01)
    assert worst < 0.25, f"event loop blocked for {worst:.3f}s"
    assert room.game is not None and room.game.actions, "the bots did not move"
    # Switching mode ends the match and resets the score.
    await room.set_mode(host, 2)
    assert room.wins == [0, 0] and room.draws == 0
    assert room.seats[host.seat] is host


def test_bot_vs_bot_keeps_the_server_responsive(tmp_path):
    asyncio.run(asyncio.wait_for(_mode4_and_responsiveness(tmp_path), timeout=120))


async def _mode4_bar_reaches_only_the_host(tmp_path):
    room = make_room(tmp_path, {"kind": "ismcts", "seconds": 0.5, "label": "bar-bot"})
    host_ws, friend_ws = FakeSocket(), FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    await room.join("Friend", friend_ws)
    await room.set_mode(host, 4)
    deadline = time.time() + 25
    bar = None
    while time.time() < deadline:
        found = [message for message in host_ws.messages if message.get("type") == "bar"]
        if found:
            bar = found[-1]
            break
        await asyncio.sleep(0.05)
    assert bar is not None, "the host did not receive a bar"
    assert 0.0 <= bar["chance"] <= 1.0
    assert any(
        message.get("spectating") and message.get("phase") == "playing" and message.get("move") == bar["move"]
        for message in host_ws.messages
    )
    assert not [message for message in friend_ws.messages if message.get("type") == "bar" or message.get("bar")]
    await room.set_mode(host, 1)


async def _mode4_final_bar_is_exact(tmp_path):
    room = make_room(tmp_path)
    host_ws = FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    await room.set_mode(host, 4)
    await pump(room, [], games=1)
    final = [message for message in host_ws.messages if message.get("spectating") and message.get("phase") == "over"]
    assert final
    assert final[-1]["bar"]["move"] == final[-1]["move"]
    winner = room.game.result["winner"]
    expected = 0.5 if winner is None else (1.0 if winner == 0 else 0.0)
    assert final[-1]["bar"]["chance"] == expected


def test_bot_vs_bot_bar_reaches_only_the_host(tmp_path):
    asyncio.run(asyncio.wait_for(_mode4_bar_reaches_only_the_host(tmp_path), timeout=40))


def test_bot_vs_bot_final_bar_is_exact(tmp_path):
    asyncio.run(asyncio.wait_for(_mode4_final_bar_is_exact(tmp_path), timeout=120))


async def _mode4_bar_streams_samples(tmp_path, monkeypatch):
    """Two running averages reach the host before the cap, and nobody else."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr("server.bot.executor", lambda: pool)

    def fake_score(snapshot, spec, seed, progress=None, stop=None, token=None):
        progress.put_nowait((token, 0.25, 1, 2))
        time.sleep(0.45)
        if stop is not None and stop.is_set():
            return None
        progress.put_nowait((token, 0.75, 2, 2))
        return {"chance": 0.75, "kind": "random", "samples": 2, "cap": 2}

    monkeypatch.setattr("server.review.score_snapshot", fake_score)

    async def hold(self, view):
        await asyncio.sleep(30)
        return view["legal_actions"][0]

    monkeypatch.setattr(BotPlayer, "choose_action", hold)
    room = make_room(tmp_path)
    host = None
    try:
        host_ws, friend_ws = FakeSocket(), FakeSocket()
        host, _ = await room.join("Host", host_ws, "secret-host")
        await room.join("Friend", friend_ws)
        await room.set_mode(host, 4)
        deadline = time.time() + 10
        bars = []
        while time.time() < deadline:
            bars = [message for message in host_ws.messages if message.get("type") == "bar"]
            counts = [message.get("samples") for message in bars]
            if 1 in counts and 2 in counts:
                break
            await asyncio.sleep(0.05)
        counts = [message.get("samples") for message in bars]
        assert 1 in counts and 2 in counts, counts
        assert counts == sorted(c for c in counts if c is not None)
        assert bars[counts.index(1)]["chance"] == 0.25
        assert bars[counts.index(1)]["cap"] == 2
        assert bars[counts.index(2)]["chance"] == 0.75
        assert bars[counts.index(2)]["samples"] == 2
        assert not [
            message for message in friend_ws.messages if message.get("type") == "bar" or message.get("bar")
        ]
    finally:
        if host is not None:
            await room.set_mode(host, 1)
        pool.shutdown(wait=True, cancel_futures=True)


def test_bot_vs_bot_bar_streams_samples(tmp_path, monkeypatch):
    asyncio.run(asyncio.wait_for(_mode4_bar_streams_samples(tmp_path, monkeypatch), timeout=20))


async def _mode4_next_game(tmp_path):
    room = make_room(tmp_path)
    host_ws = FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    await room.set_mode(host, 4)
    await pump(room, [], games=1)
    await asyncio.sleep(0.2)
    assert room.phase == "over", "bot vs bot must wait for the host"
    first = room.game
    await room.next_game(host)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if room.game is not first:
            break
    assert room.game is not first
    record = load_record(sorted(tmp_path.glob("*.json"))[-1])
    assert [seat["kind"] for seat in record["seats"]] == ["bot", "bot"]
    assert any(m.get("spectating") and m["seat"] == 1 for m in host_ws.messages)


def test_bot_vs_bot_waits_for_the_host(tmp_path):
    asyncio.run(asyncio.wait_for(_mode4_next_game(tmp_path), timeout=180))


async def _mode2(tmp_path):
    room = make_room(tmp_path)
    host_ws, friend_ws = FakeSocket(), FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    friend, _ = await room.join("Friend", friend_ws)
    await room.set_mode(host, 2)
    assert friend.seat == -1 and room.seats[host.seat] is host
    late, err = await room.join("Late", FakeSocket())
    assert late is None and "bot" in err
    assert await room.confirm_move(host, room.move_id) == "The bot is not waiting for confirmation."
    await pump(room, [host], games=1)
    assert not [m for m in host_ws.messages if m.get("type") == "panel" or m.get("spectating")]
    assert [m for m in friend_ws.messages if m.get("notice")]
    # Back to two people: the friend gets their seat again, the score starts over.
    await room.set_mode(host, 1)
    assert room.seats[friend.home_seat] is friend and room.wins == [0, 0]


def test_mode2_shows_no_evaluations(tmp_path):
    asyncio.run(asyncio.wait_for(_mode2(tmp_path), timeout=180))


def test_review_needs_the_host_link_and_analyses_a_game(tmp_path, monkeypatch):
    monkeypatch.setattr("server.review.review_spec", lambda spec: {**spec, "line_limit": 4, "seed": 42})
    monkeypatch.setattr("server.review.EVAL_SAMPLES", 2)
    room = make_room(tmp_path)
    asyncio.run(asyncio.wait_for(_finish_one_mode3_game(room), timeout=180))
    name = sorted(tmp_path.glob("*.json"))[-1].name
    frames = game_frames(tmp_path / name)
    assert len(frames["frames"]) == len(frames["record"]["actions"]) + 1
    assert all(len(f["hands"]) == 2 for f in frames["frames"])
    analyse_game(str(tmp_path), name, FAST_BOT)
    status = read_status(tmp_path, name, spec_version(FAST_BOT))
    assert status["status"] == "done"
    moves = status["result"]["moves"]
    assert len(moves) == len(frames["record"]["actions"])
    forced_at = {frame["index"] for frame in frames["frames"] if frame.get("legal_count") == 1}
    for move in moves:
        assert move["bot_action"] in [m["action"] for m in move["moves"]]
        assert move["iterations"] == 4 * len(move["moves"])
        assert sum(option["visits"] for option in move["moves"]) == move["iterations"]
        assert "mistake" not in move
        assert 0.0 <= move["win_chance"] <= 1.0
        assert move["forced"] == (move["index"] in forced_at)
        if move["forced"]:
            assert len(move["moves"]) == 1
            assert move["iterations"] == 4
            assert move["bot_action"] == move["actual"]
            assert move["loss"] == 0
    assert "mistake_margin" not in status["result"]
    chances = status["result"]["eval"]
    assert len(chances) == len(frames["frames"])
    assert status["result"]["eval_kind"] == "random"
    assert all(0.0 <= chance <= 1.0 for chance in chances)
    winner = frames["record"]["result"]["winner"]
    if winner is None:
        assert chances[-1] == 0.5
    else:
        assert chances[-1] == (1.0 if winner == 0 else 0.0)
    with TestClient(create_app(room)) as client:
        assert client.get("/api/review/games").status_code == 403
        assert client.get("/api/review/games", params={"host": "wrong"}).status_code == 403
        listing = client.get("/api/review/games", params={"host": "secret-host"}).json()
        assert listing["games"][0]["file"] == name
        game = client.get("/api/review/game", params={"host": "secret-host", "file": name}).json()
        assert game["analysis"]["status"] == "done"
        assert client.get("/api/review/game", params={"host": "secret-host", "file": "../x.json"}).status_code == 404
        assert client.get("/review").status_code == 200


async def _finish_one_mode3_game(room):
    host, _ = await room.join("Host", FakeSocket(), "secret-host")
    friend, _ = await room.join("Friend", FakeSocket())
    await room.set_mode(host, 3)
    await pump(room, [friend], games=1)


async def _mode3_waits_for_confirm(tmp_path):
    room = make_room(tmp_path, {"kind": "ismcts", "iterations": 40, "label": "capped"})
    host_ws, friend_ws = FakeSocket(), FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    friend, _ = await room.join("Friend", friend_ws)
    await room.set_mode(host, 3)
    deadline = time.time() + 70
    while time.time() < deadline:
        game = room.game
        if room.phase == "playing" and game is not None and not game.is_over:
            player = room.seats[game.current]
            if isinstance(player, BotPlayer):
                break
            if player is friend and player.expected_token:
                player.submit(game.legal_actions()[0], player.expected_token)
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the bot never got a turn")
    bot = room.seats[room.game.current]
    assert bot.confirm_moves and bot.search_limit == 40
    before = len(room.game.actions)
    paused = None
    while time.time() < deadline:
        found = [m["panel"] for m in host_ws.messages if m.get("type") == "panel" and m["panel"].get("paused")]
        if found:
            paused = found[-1]
            break
        await asyncio.sleep(0.02)
    assert paused is not None, "the search did not stop at the cap"
    assert paused["final"] is False and paused["iterations"] == 40
    await asyncio.sleep(0.2)
    assert len(room.game.actions) == before, "the bot played without confirmation"
    assert "host" in (await room.confirm_move(friend, room.move_id)).lower()
    assert await room.confirm_move(host, room.move_id - 1)
    assert len(room.game.actions) == before
    assert await room.confirm_move(host, room.move_id) is None
    while time.time() < deadline and len(room.game.actions) == before:
        await asyncio.sleep(0.02)
    assert len(room.game.actions) == before + 1
    assert_friend_sees_only_their_seat(friend_ws.messages, friend.seat)


def test_mode3_waits_until_the_host_confirms(tmp_path):
    asyncio.run(asyncio.wait_for(_mode3_waits_for_confirm(tmp_path), timeout=120))


async def _timed_cap_holds_the_move(tmp_path):
    room = make_room(
        tmp_path,
        {"kind": "ismcts", "iterations": 1, "seconds": 0.45, "label": "capped"},
    )
    host_ws = FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    await room.set_mode(host, 4)
    deadline = time.time() + 20
    final = None
    while time.time() < deadline:
        found = [
            message["panel"]
            for message in host_ws.messages
            if message.get("type") == "panel" and message["panel"].get("final")
        ]
        if found and room.game is not None:
            final = found[-1]
            break
        await asyncio.sleep(0.01)
    assert final is not None, "the finished evaluation was not published"
    assert final["iterations"] == 1
    assert final["seconds"] < 0.2
    assert room.game.actions == [], "the bot played when the search stopped"
    held_at = time.perf_counter()
    while time.perf_counter() - held_at < 2 and not room.game.actions:
        await asyncio.sleep(0.01)
    waited = time.perf_counter() - held_at
    assert room.game.actions, "the bot did not play when the clock elapsed"
    assert waited > 0.25


def test_timed_cap_holds_the_move(tmp_path):
    asyncio.run(asyncio.wait_for(_timed_cap_holds_the_move(tmp_path), timeout=30))


async def _mode3_ignores_the_clock(tmp_path):
    # A short clock would stop the search early if it were applied. The cap is
    # large enough that reaching it takes longer than that clock.
    room = make_room(
        tmp_path,
        {"kind": "ismcts", "iterations": 300, "seconds": 0.05, "label": "capped"},
    )
    host_ws, friend_ws = FakeSocket(), FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    friend, _ = await room.join("Friend", friend_ws)
    await room.set_mode(host, 3)
    deadline = time.time() + 20
    while time.time() < deadline:
        game = room.game
        if room.phase == "playing" and game is not None and not game.is_over:
            player = room.seats[game.current]
            if isinstance(player, BotPlayer):
                break
            if player is friend and player.expected_token:
                player.submit(game.legal_actions()[0], player.expected_token)
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the bot never got a turn")
    before = len(room.game.actions)
    turn_started = time.perf_counter()
    paused = None
    while time.time() < deadline:
        found = [m["panel"] for m in host_ws.messages if m.get("type") == "panel" and m["panel"].get("paused")]
        if found:
            paused = found[-1]
            break
        await asyncio.sleep(0.02)
    assert paused is not None, "the search did not stop at the cap"
    assert paused["iterations"] == 300 and paused["final"] is False
    assert time.perf_counter() - turn_started < 10
    assert len(room.game.actions) == before, "the bot played without confirmation"
    assert await room.confirm_move(host, room.move_id) is None
    while time.time() < deadline and len(room.game.actions) == before:
        await asyncio.sleep(0.02)
    assert len(room.game.actions) == before + 1
    assert time.perf_counter() - turn_started < 10


def test_mode3_ignores_the_clock(tmp_path):
    asyncio.run(asyncio.wait_for(_mode3_ignores_the_clock(tmp_path), timeout=30))


async def _resign_bot_and_reject_bot_vs_bot(tmp_path):
    room = make_room(tmp_path)
    host_ws, friend_ws = FakeSocket(), FakeSocket()
    host, _ = await room.join("Host", host_ws, "secret-host")
    friend, _ = await room.join("Friend", friend_ws)
    await room.set_mode(host, 3)
    for _ in range(2000):
        if room.phase == "playing" and room.game is not None:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("friend vs bot did not start")
    bot_seat = next(seat for seat, player in enumerate(room.seats) if isinstance(player, BotPlayer))
    assert await room.resign(host) is None
    for _ in range(2000):
        if room.phase == "over" and any(message.get("type") == "toast" for message in friend_ws.messages):
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("the bot's resignation did not finish the hand")
    assert room.game.result == {"winner": 1 - bot_seat, "reason": "resign"}
    assert room.wins[1 - bot_seat] == 1 and room.wins[bot_seat] == 0
    assert room.rematch[bot_seat] is True and room.rematch[1 - bot_seat] is False
    toasts = [message["message"] for message in friend_ws.messages if message.get("type") == "toast"]
    assert toasts == ["The bot resigned."]
    assert not [message for message in host_ws.messages if message.get("type") == "toast"]

    await room.set_mode(host, 4)
    assert await room.resign(host) == "Nobody is playing this seat."
    task = room._task
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_host_can_resign_the_bot_but_not_bot_vs_bot(tmp_path):
    asyncio.run(asyncio.wait_for(_resign_bot_and_reject_bot_vs_bot(tmp_path), timeout=30))
