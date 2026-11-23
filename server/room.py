"""One table for two players.

The room keeps the cards. Browsers only ever receive the view for their own
seat, and a move is applied only when its action number is legal.

Game modes (only the host - whoever opens the private host link - can change
them; changing the mode starts a new match with a fresh score):

1. Two people. Works exactly as it always has.
2. The host against the bot. No hints or evaluations. The bot thinks for
   its time limit, and also stops at the imagined-game cap. Hitting the cap
   does not play the card early.
3. The friend against the bot, which sits in the host's seat. The host
   watches that seat: its hand and the bot's evaluation panel, and confirms
   each of the bot's moves. The bot searches until then, or until its
   imagined-game cap, and only plays once the host confirms. The think
   clock does not apply.
4. The bot against itself. The host watches both seats and both panels,
   plus a win-chance bar for the bottom seat. The bar updates as finishes
   complete, up to the sample cap, and starts again when the position
   changes. Each bot uses the same time limit and imagined-game cap as
   "Me vs the bot". The next game starts when the host asks for it.

Bot hands and panels are sent only to the host's sockets.
"""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from pathlib import Path

from prsi.actions import DONT_PLAY
from prsi.game import Game
from prsi.record import save_record
from server.players import HumanPlayer, Player

log = logging.getLogger("prsi.server")

MODES = (1, 2, 3, 4)
MODE_NAMES = {
    1: "Me vs my friend",
    2: "Me vs the bot",
    3: "My friend vs the bot",
    4: "Bot vs bot",
}


def clean_name(raw: object, fallback: str) -> str:
    text = "".join(
        ch for ch in str(raw or "") if ch.isprintable() and ch not in "<>&"
    )
    text = " ".join(text.split())
    return text[:20] or fallback


class Room:
    def __init__(
        self,
        code: str,
        games_dir: Path,
        *,
        host_token: str | None = None,
        bot_factory=None,
        bot_note: str = "",
        bot_spec: dict | None = None,
    ):
        self.code = code.strip().upper()
        self.games_dir = Path(games_dir)
        self.seats: list[Player | None] = [None, None]
        self.spectators: list[tuple[object, int]] = []
        self.humans: list[HumanPlayer] = []
        self.mode = 1
        self.host_token = host_token or secrets.token_urlsafe(18)
        # bot_factory(seat, on_panel) -> Player. None means no bot is available.
        self.bot_factory = bot_factory
        self.bot_note = bot_note
        # The bot's description (see bot.agents), used for the review page.
        self.bot_spec = bot_spec
        self.bots: dict[int, Player] = {}
        self.panels: dict[int, dict] = {}
        self.wins = [0, 0]
        self.draws = 0
        self.game: Game | None = None
        self.phase = "waiting"  # waiting, playing, over
        self.rematch = [False, False]
        self.next_starter: int | None = None
        self.move_id = 0
        self.seq = 0
        self._rng = random.SystemRandom()
        self._lock = asyncio.Lock()
        self._rematch_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Move id of the bot turn the host may confirm, and the token of that
        # turn. The token keeps a cancelled game from clearing the next one.
        self._confirm_move_id: int | None = None
        self._confirm_token = 0
        self._confirm_token_open: int | None = None
        # The live bar scores one position at a time and publishes a running
        # mean until the sample cap. ``_bar_epoch`` changes with each deal so
        # a late score cannot land on the next game. ``_bar_stop`` abandons
        # the worker between finishes.
        self._bar_epoch = 0
        self._bar_key = None
        self._bar_ready: dict | None = None
        self._bar_latest: dict | None = None
        self._bar_running = False
        self._bar_stop = None
        self._bar_queue = None
        # Bumped when the current hand is thrown away, so a late move cannot
        # land on the deal that replaces it.
        self._deal_gen = 0
        # Set when a player resigns, so the seat that is choosing a move stops.
        self._turn_halt = asyncio.Event()
        # One-shot lines sent with the next publish, then forgotten.
        self._toasts: list[tuple[list, str]] = []

    def matches(self, code: object) -> bool:
        return str(code or "").strip().upper() == self.code

    def is_host_token(self, token: object) -> bool:
        return isinstance(token, str) and bool(token) and secrets.compare_digest(token, self.host_token)

    def add_spectator(self, ws: object, seat: int) -> None:
        """Watch one seat (its hand and the public table)."""
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        self.spectators.append((ws, seat))

    # ------------------------------------------------------------ people

    def host(self) -> HumanPlayer | None:
        for player in self.humans:
            if player.is_host:
                return player
        return None

    def _friends(self) -> list[HumanPlayer]:
        return [player for player in self.humans if not player.is_host]

    async def join(self, name: object, ws, host: object = None) -> tuple[HumanPlayer | None, str | None]:
        async with self._lock:
            is_host = self.is_host_token(host)
            if is_host:
                existing = self.host()
                if existing is not None:
                    existing.sockets.add(ws)
                    return existing, None
                if self.mode == 1 and len(self.humans) >= 2:
                    return None, "Room is full."
            else:
                if self.mode in (2, 4):
                    return None, "The host is playing against the bot right now. Try again later."
                if self.mode == 1 and len(self.humans) >= 2:
                    return None, "Room is full."
                if self.mode == 3 and self._friends():
                    return None, "Room is full."
            player = HumanPlayer(clean_name(name, f"Player {len(self.humans) + 1}"), -1)
            player.is_host = is_host
            player.sockets.add(ws)
            self.humans.append(player)
            self._arrange_locked()
            self._ensure_task_locked()
        return player, None

    async def rejoin(self, token: object, ws) -> tuple[HumanPlayer | None, str | None]:
        async with self._lock:
            for player in self.humans:
                if isinstance(token, str) and secrets.compare_digest(player.token, token):
                    player.sockets.add(ws)
                    return player, None
        return None, "This session has expired. Enter the room code to join."

    def _arrange_locked(self) -> None:
        """Put people and bots in seats for the current mode."""
        mode = self.mode
        host = self.host()
        friends = self._friends()
        new: list[Player | None] = [None, None]

        def place(player: HumanPlayer, prefer: int) -> None:
            seat = prefer if prefer in (0, 1) and new[prefer] is None else (0 if new[0] is None else 1)
            new[seat] = player
            player.home_seat = seat

        if mode == 1:
            for player in self.humans:
                place(player, player.home_seat)
        elif mode == 2:
            if host is not None:
                place(host, host.home_seat)
            bot_seat = 1 if new[1] is None else 0
            new[bot_seat] = self._bot_locked(bot_seat)
        elif mode == 3:
            if friends:
                place(friends[0], friends[0].home_seat)
            if new[0] is None and new[1] is None:
                prefer = host.home_seat if host is not None and host.home_seat in (0, 1) else 0
                bot_seat = prefer
            else:
                bot_seat = 0 if new[0] is None else 1
            new[bot_seat] = self._bot_locked(bot_seat)
        else:
            new = [self._bot_locked(0), self._bot_locked(1)]
        for player in self.humans:
            player.seat = player.home_seat if player in new else -1
        self.bots = {seat: player for seat, player in enumerate(new) if player is not None and not isinstance(player, HumanPlayer)}
        self.seats = new

    def _bot_locked(self, seat: int) -> Player:
        bot = self.bots.get(seat)
        if bot is None:
            bot = self.bot_factory(seat, self._on_panel)
        if hasattr(bot, "confirm_moves"):
            # Bots are reused across modes. Only the friend's game waits.
            bot.confirm_moves = self.mode == 3
        bot.name = ("Bot A", "Bot B")[seat] if self.mode == 4 else "Bot"
        return bot

    def watched_seats(self) -> list[int]:
        """Seats the host watches as a spectator in this mode."""
        if self.mode == 3:
            return [seat for seat, player in enumerate(self.seats) if player is not None and not isinstance(player, HumanPlayer)]
        if self.mode == 4:
            return [0, 1]
        return []

    # ------------------------------------------------------------ host controls

    async def set_mode(self, player: HumanPlayer, mode: object) -> str | None:
        if not getattr(player, "is_host", False):
            return "Only the host can change the game mode."
        if mode not in MODES:
            return "Unknown game mode."
        if mode != 1 and self.bot_factory is None:
            return "The bot is not available."
        async with self._lock:
            if mode == self.mode:
                return None
            task = self._task
            self._task = None
            if task is not None:
                task.cancel()
            self.mode = mode
            self.wins = [0, 0]
            self.draws = 0
            self.game = None
            self.phase = "waiting"
            self.rematch = [False, False]
            self.next_starter = None
            self.panels = {}
            self._confirm_move_id = None
            self._confirm_token_open = None
            self._stop_bar_locked()
            self._bar_epoch += 1
            self._bar_key = None
            self._bar_ready = None
            self._bar_latest = None
            self._deal_gen += 1
            self._turn_halt.clear()
            self._rematch_event.clear()
            self._arrange_locked()
            self._ensure_task_locked()
        await self.publish()
        return None

    async def confirm_move(self, player: HumanPlayer, move: object) -> str | None:
        """Host accepts the bot's current move in "My friend vs the bot"."""
        if not getattr(player, "is_host", False):
            return "Only the host can confirm the bot's move."
        if self.mode != 3:
            return "The bot is not waiting for confirmation."
        if type(move) is not int:
            return "That confirmation is out of date."
        async with self._lock:
            if self._confirm_move_id is None or self.game is None:
                return "The bot is not choosing a move."
            if move != self._confirm_move_id:
                return "That confirmation is out of date."
            bot = self.seats[self.game.current]
            if not getattr(bot, "confirm_moves", False) or not hasattr(bot, "confirm"):
                return "The bot is not choosing a move."
            bot.confirm()
        return None

    async def next_game(self, player: HumanPlayer) -> str | None:
        """Bot vs bot: the host starts the next game."""
        if not getattr(player, "is_host", False):
            return "Only the host can start the next game."
        async with self._lock:
            if self.phase != "over":
                return None
            self.rematch = [True, True]
            self._rematch_event.set()
        await self.publish()
        return None

    async def reset_score(self, player: HumanPlayer) -> str | None:
        """Zero the match and deal again. The host only.

        An unfinished hand is saved with no winner and does not change the score.
        The drive task is cancelled after the lock is released, and awaited, so
        a move it already chose cannot step the new deal.
        """
        if not getattr(player, "is_host", False):
            return "Only the host can reset the score."
        async with self._lock:
            if self.seats[0] is None or self.seats[1] is None or self.phase == "waiting":
                return "The game has not started."
            task = self._task
            self._task = None
            if self.phase == "playing" and self.game is not None and self.game.is_over:
                self._finish_locked()
            if self.phase == "playing" and self.game is not None and not self.game.is_over:
                self.next_starter = 1 - self.game.starting_player
                self._save_record_locked({"winner": None, "reason": "reset"})
            self.wins = [0, 0]
            self.draws = 0
            self.game = None
            self.phase = "waiting"
            self.rematch = [False, False]
            self.panels = {}
            self._confirm_move_id = None
            self._confirm_token_open = None
            self._stop_bar_locked()
            self._bar_epoch += 1
            self._bar_key = None
            self._bar_ready = None
            self._bar_latest = None
            self._deal_gen += 1
            self._turn_halt.clear()
            # Wake a loop that is sitting between hands so it can notice the
            # table is no longer waiting for a rematch.
            self._rematch_event.set()
            if self.mode in (1, 3):
                self._queue_toast_locked(
                    self._friends(),
                    "The host reset the score and dealt a new hand.",
                )
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            self._ensure_task_locked()
        return None

    async def resign(self, player: HumanPlayer) -> str | None:
        """The caller's seat loses this hand. In friend-vs-bot, the host resigns the bot."""
        async with self._lock:
            if self.mode == 4:
                return "Nobody is playing this seat."
            if self.phase == "over" or (self.game is not None and self.game.is_over):
                return "The hand is already over."
            if self.phase != "playing" or self.game is None:
                return "You can't resign right now."
            loser = self._resign_seat_locked(player)
            if isinstance(loser, str):
                return loser
            if self.game.is_over:
                return "The hand is already over."
            self.game.concede(loser)
            seated = self.seats[loser]
            if isinstance(seated, HumanPlayer):
                text = f"{seated.name} resigned."
            else:
                text = "The bot resigned."
            others = [human for human in self.humans if human is not player]
            self._queue_toast_locked(others, text)
            self._turn_halt.set()
        return None

    def _resign_seat_locked(self, player: HumanPlayer) -> int | str:
        if player.seat in (0, 1) and self.seats[player.seat] is player:
            return player.seat
        if getattr(player, "is_host", False) and self.mode == 3 and player.seat == -1:
            for seat, seated in enumerate(self.seats):
                if seated is not None and not isinstance(seated, HumanPlayer):
                    return seat
        return "You are watching this game."

    async def submit_action(self, seat: int, action: int, move: int) -> str | None:
        async with self._lock:
            if seat not in (0, 1):
                return "You are watching this game."
            player = self.seats[seat]
            if not isinstance(player, HumanPlayer):
                return "This seat is not waiting for a click."
            if self.phase != "playing" or self.game is None or self.game.is_over:
                return "The game is not waiting for a move."
            if self.game.current != seat:
                return "It's not your turn."
            if move != self.move_id:
                return "That move is out of date. Try again."
            player.submit(action, move)
        return None

    async def vote_rematch(self, seat: int) -> None:
        async with self._lock:
            if seat not in (0, 1) or self.phase != "over" or self.seats[seat] is None:
                return
            self.rematch[seat] = True
            if self.rematch[0] and self.rematch[1]:
                self._rematch_event.set()
        await self.publish()

    async def publish(self) -> None:
        async with self._lock:
            plan = self._plan_locked()
        await self._emit(plan)

    # ------------------------------------------------------------ the game loop

    def _ensure_task_locked(self) -> None:
        if self.seats[0] is None or self.seats[1] is None or self.phase != "waiting":
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        try:
            while True:
                opened = await self._begin_and_open()
                if opened is None:
                    return
                if not await self._play_until_over(opened):
                    return
                if not await self._wait_rematch():
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("The game loop stopped.")

    async def _begin_and_open(self):
        async with self._lock:
            if self.seats[0] is None or self.seats[1] is None:
                self.phase = "waiting"
                self.game = None
                plan = self._plan_locked()
                opened = None
            else:
                self._begin_game_locked()
                opened = self._arm_turn_locked()
                plan = self._plan_locked()
        await self._emit(plan)
        return opened

    async def _play_until_over(self, opened) -> bool:
        """Play ``opened`` until the hand ends.

        Return False when this loop must stop because the hand was replaced
        (a reset or a mode change), so it does not sit in the rematch wait
        and deal a second game on top of the new one.
        """
        while opened is not None:
            seat, player, view, gen, game = opened
            token = self._confirm_token
            action = await self._take_action(player, view)
            stepped = False
            async with self._lock:
                # This decision is over. A late click must not confirm the next
                # turn, and a cancelled game must not clear the new one's token.
                if self._confirm_token_open == token:
                    self._confirm_move_id = None
                    self._confirm_token_open = None
                if self.game is not game or self._deal_gen != gen or self.phase != "playing":
                    return False
                if action is None and not self.game.is_over:
                    self._turn_halt.clear()
                    continue
                if action is None or self.game.is_over or self.game.current != seat:
                    break
                if action in self.game.legal_actions():
                    self.game.step(action)
                    stepped = True
                    if self.game.is_over:
                        break
            if not stepped:
                continue
            opened = await self._open_turn()
        async with self._lock:
            if self.phase != "playing":
                return False
            self._finish_locked()
            plan = self._plan_locked()
        await self._emit(plan)
        return self.phase == "over"

    async def _take_action(self, player: Player, view: dict) -> int | None:
        """The seat's move, or None when a resignation ended the hand first."""
        choose = asyncio.create_task(player.choose_action(view))
        halt = asyncio.create_task(self._turn_halt.wait())
        try:
            done, pending = await asyncio.wait(
                {choose, halt}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            choose.cancel()
            halt.cancel()
            await asyncio.gather(choose, halt, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if choose not in done or choose.cancelled():
            return None
        return choose.result()

    async def _open_turn(self):
        async with self._lock:
            if self.game is None or self.game.is_over or self.phase != "playing":
                return None
            opened = self._arm_turn_locked()
            plan = self._plan_locked()
        await self._emit(plan)
        return opened

    async def _wait_rematch(self) -> bool:
        """Wait until both seats want another hand.

        Return False when the table left the post-hand phase, so a reset can
        take over the deal instead of this loop starting one too.
        """
        while True:
            async with self._lock:
                if self.phase != "over":
                    return False
                if self.rematch[0] and self.rematch[1]:
                    return True
                self._rematch_event.clear()
            await self._rematch_event.wait()

    def _begin_game_locked(self) -> None:
        starter = self.next_starter
        if starter is None:
            starter = self._rng.randrange(2)
        seed = self._rng.randrange(2**63)
        self.game = Game(seed, starter)
        self.phase = "playing"
        self.rematch = [False, False]
        self._turn_halt.clear()
        self._rematch_event.clear()
        self.move_id = 0
        self.panels = {}
        self._stop_bar_locked()
        self._bar_epoch += 1
        self._bar_key = None
        self._bar_ready = None

    def _arm_turn_locked(self):
        assert self.game is not None
        seat = self.game.current
        player = self.seats[seat]
        assert player is not None
        self.move_id += 1
        self._confirm_token += 1
        if isinstance(player, HumanPlayer):
            player.expected_token = self.move_id
        if getattr(player, "confirm_moves", False) and hasattr(player, "begin_turn"):
            player.begin_turn()
            self._confirm_move_id = self.move_id
            self._confirm_token_open = self._confirm_token
        else:
            self._confirm_move_id = None
            self._confirm_token_open = None
        view = self.game.observation(seat)
        return seat, player, view, self._deal_gen, self.game

    def _finish_locked(self) -> None:
        if self.phase != "playing" or self.game is None or not self.game.is_over:
            return
        # The finished cards are a new position. A new move id keeps a score
        # of the position before the last play from filling the final bar.
        if self.mode == 4:
            self.move_id += 1
        winner = self.game.result["winner"] if self.game.result else None
        if winner is None:
            self.draws += 1
        else:
            self.wins[winner] += 1
        self.next_starter = 1 - self.game.starting_player
        self._save_record_locked()
        self.phase = "over"
        if self.mode in (2, 3):
            # The bot always wants a rematch; in mode 4 the host decides.
            for seat in self.bots:
                self.rematch[seat] = True

    def _save_record_locked(self, result: dict | None = None) -> None:
        if self.game is None:
            return
        names = [self._seat_name(index) for index in range(2)]
        seats = [self._seat_info(index) for index in range(2)]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = self.games_dir / f"{stamp}_{self.game.seed}.json"
        if path.exists():
            path = self.games_dir / f"{stamp}_{self.game.seed}_{self.seq}.json"
        try:
            save_record(
                self.game, path, names=names, seats=seats, mode=self.mode, result=result
            )
        except OSError:
            log.exception("Could not save the game to %s", path)

    def _seat_name(self, seat: int) -> str:
        player = self.seats[seat]
        if isinstance(player, HumanPlayer):
            return player.name
        if player is not None and getattr(player, "name", None):
            return player.name
        return f"Player {seat + 1}"

    def _seat_info(self, seat: int) -> dict:
        player = self.seats[seat]
        if isinstance(player, HumanPlayer):
            return {"kind": "human", "name": player.name}
        if player is not None and hasattr(player, "describe"):
            return player.describe()
        return {"kind": "unknown", "name": self._seat_name(seat)}

    # ------------------------------------------------------------ bot panels

    async def _on_panel(self, seat: int, panel: dict) -> None:
        async with self._lock:
            if self.bots.get(seat) is None:
                return
            self.panels[seat] = panel
            host = self.host()
            if host is None or seat not in self.watched_seats():
                return
            message = {"type": "panel", "seat": seat, "move": self.move_id, "panel": panel}
            sockets = list(host.sockets)
        for ws in sockets:
            try:
                await ws.send_json(message)
            except Exception:
                host.sockets.discard(ws)

    # ------------------------------------------------------------ messages

    def _plan_locked(self) -> dict:
        notifies = []
        deliveries = []
        for seat, player in enumerate(self.seats):
            if player is None:
                continue
            view = self._view_locked(seat)
            notifies.append((player, view))
            if not isinstance(player, HumanPlayer):
                continue
            message = self._message_locked(seat, view)
            if player.is_host:
                message.update(self._host_fields_locked())
            for ws in list(player.sockets):
                deliveries.append((ws, message, player))
        for ws, seat in list(self.spectators):
            if seat not in (0, 1):
                continue
            deliveries.append((ws, self._message_locked(seat, self._view_locked(seat)), None))
        host = self.host()
        if host is not None and host.seat == -1:
            watched = self.watched_seats()
            for seat in watched:
                message = self._message_locked(seat, self._view_locked(seat))
                message.update(self._host_fields_locked())
                message["spectating"] = True
                message["watching"] = watched
                message["panel"] = self.panels.get(seat)
                for ws in list(host.sockets):
                    deliveries.append((ws, message, host))
            if not watched:
                message = self._benched_message_locked()
                message.update(self._host_fields_locked())
                for ws in list(host.sockets):
                    deliveries.append((ws, message, host))
        for friend in self._friends():
            if friend.seat != -1:
                continue
            message = self._benched_message_locked()
            for ws in list(friend.sockets):
                deliveries.append((ws, message, friend))
        plan = {"notifies": notifies, "deliveries": deliveries, "toasts": self._toasts}
        self._toasts = []
        plan["bar"] = self._prepare_bar_locked(deliveries)
        return plan

    def _prepare_bar_locked(self, deliveries: list) -> dict | None:
        """Attach a finished bar, or return a position for the worker to score.

        Called once per plan, while the room lock is held. The same position
        is not scored twice; a later plan for it reuses the chance already found.
        """
        if self.mode != 4 or self.game is None or not self.bot_spec:
            return None
        key = (self._bar_epoch, len(self.game.actions), self.game.is_over)
        if key == self._bar_key:
            if self._bar_ready is not None and self._bar_ready["move"] == self.move_id:
                self._attach_bar(deliveries, self._bar_ready)
            return None
        self._bar_key = key
        self._bar_ready = None
        if self.game.is_over:
            self._stop_bar_locked()
            winner = self.game.result["winner"] if self.game.result else None
            ready = {
                "move": self.move_id,
                "chance": _chance_for_winner(winner),
                "kind": "network" if self.bot_spec.get("path") else "random",
            }
            self._bar_ready = ready
            self._attach_bar(deliveries, ready)
            return None
        from server.review import bar_snapshot

        return {
            "move": self.move_id,
            "epoch": self._bar_epoch,
            "snapshot": bar_snapshot(self.game),
            "seed": (self.move_id * 1_000_003 + int(self.game.seed or 0)) & 0xFFFFFFFFFFFF,
            "spec": _bar_spec(self.bot_spec),
        }

    def _attach_bar(self, deliveries: list, payload: dict) -> None:
        for _ws, message, _owner in deliveries:
            if message.get("spectating") and message.get("mode") == 4:
                message["bar"] = dict(payload)

    def _host_fields_locked(self) -> dict:
        return {
            "host": True,
            "mode": self.mode,
            "modes": [{"id": mode, "name": MODE_NAMES[mode]} for mode in MODES],
            "bot_available": self.bot_factory is not None,
            "bot_note": self.bot_note,
        }

    def _benched_message_locked(self) -> dict:
        self.seq += 1
        return {
            "type": "state",
            "seq": self.seq,
            "seat": -1,
            "code": self.code,
            "names": [self._seat_name(0), self._seat_name(1)],
            "connected": [True, True],
            "score": {"wins": list(self.wins), "draws": self.draws},
            "rematch": list(self.rematch),
            "phase": "waiting",
            "move": self.move_id,
            "notice": "The host is playing with the bot right now. You'll get your seat back "
                      "when they switch to a two-player game.",
            "view": _lobby_view(-1),
        }

    def _view_locked(self, seat: int) -> dict:
        if self.game is None:
            return _lobby_view(seat)
        return self.game.observation(seat)

    def _message_locked(self, seat: int, view: dict) -> dict:
        self.seq += 1
        names = []
        connected = []
        for player in self.seats:
            if isinstance(player, HumanPlayer):
                names.append(player.name)
                connected.append(bool(player.sockets))
            else:
                names.append(getattr(player, "name", None) or (player.__class__.__name__ if player else None))
                connected.append(player is not None)
        return {
            "type": "state",
            "seq": self.seq,
            "seat": seat,
            "code": self.code,
            "names": names,
            "connected": connected,
            "score": {"wins": list(self.wins), "draws": self.draws},
            "rematch": list(self.rematch),
            "phase": self.phase,
            "move": self.move_id,
            "seed": None if self.game is None else str(self.game.seed),
            "view": view,
        }

    async def _emit(self, plan: dict) -> None:
        for player, view in plan["notifies"]:
            await player.notify(view)
        for ws, message, owner in plan["deliveries"]:
            try:
                await ws.send_json(message)
            except Exception:
                if isinstance(owner, HumanPlayer):
                    owner.sockets.discard(ws)
                else:
                    self.spectators = [
                        item for item in self.spectators if item[0] is not ws
                    ]
        for sockets, message in plan.get("toasts") or []:
            await self._send_toast(sockets, message)
        request = plan.get("bar")
        if request:
            await self._schedule_bar(request)

    def _queue_toast_locked(self, humans: list[HumanPlayer], message: str) -> None:
        sockets = []
        for human in humans:
            sockets.extend(list(human.sockets))
        if sockets:
            self._toasts.append((sockets, message))

    async def _send_toast(self, sockets: list, message: str) -> None:
        for ws in sockets:
            try:
                await ws.send_json({"type": "toast", "message": message})
            except Exception:
                for human in self.humans:
                    human.sockets.discard(ws)

    def _stop_bar_locked(self) -> None:
        """Ask the in-flight bar to finish the sample it is on and then stop."""
        stop = self._bar_stop
        if stop is not None:
            stop.set()

    def _bar_still_locked(self, epoch: int, move: int) -> bool:
        """True while ``move`` is still the position on the table."""
        return (
            self.mode == 4
            and self._bar_epoch == epoch
            and self.move_id == move
            and self._bar_latest is None
            and self.game is not None
            and not self.game.is_over
        )

    async def _schedule_bar(self, request: dict) -> None:
        """Score ``request`` in a worker. A newer position replaces it; nothing queues up."""
        async with self._lock:
            self._stop_bar_locked()
            self._bar_latest = request
            if self._bar_running:
                return
            self._bar_running = True
        asyncio.create_task(self._run_bar())

    async def _run_bar(self) -> None:
        from server.bot import executor
        from server.review import score_snapshot

        loop = asyncio.get_running_loop()
        try:
            while True:
                async with self._lock:
                    request = self._bar_latest
                    self._bar_latest = None
                    if request is None or self.mode != 4:
                        self._bar_running = False
                        return
                    move = request["move"]
                    epoch = request["epoch"]
                    snapshot = request["snapshot"]
                    spec = request["spec"]
                    seed = request["seed"]
                queue, stop = await loop.run_in_executor(None, _bar_channel, self)
                async with self._lock:
                    if not self._bar_still_locked(epoch, move):
                        continue
                    self._bar_stop = stop
                token = (epoch, move)
                try:
                    future = loop.run_in_executor(
                        executor(), score_snapshot, snapshot, spec, seed, queue, stop, token
                    )
                    while not future.done():
                        await asyncio.wait({future}, timeout=0.2)
                        latest = await loop.run_in_executor(None, _latest_bar, queue, token)
                        if latest is not None:
                            _token, chance, played, cap = latest
                            await self._publish_live_bar(epoch, move, chance, played, cap, spec)
                    result = future.result()
                except Exception:
                    log.exception("The evaluation bar failed.")
                    result = None
                if result is not None and not stop.is_set():
                    await self._publish_live_bar(
                        epoch, move, result["chance"], result["samples"], result["cap"], spec
                    )
                async with self._lock:
                    if self._bar_latest is None or self.mode != 4:
                        self._bar_running = False
                        return
        except Exception:
            log.exception("The evaluation bar stopped.")
            self._bar_running = False

    async def _publish_live_bar(self, epoch, move, chance, samples, cap, spec) -> None:
        """Send one running average to the host, and keep it for a later refresh."""
        async with self._lock:
            if not self._bar_still_locked(epoch, move):
                return
            payload = {
                "move": move,
                "chance": chance,
                "kind": "network" if spec.get("path") else "random",
                "samples": int(samples),
                "cap": int(cap),
            }
            self._bar_ready = payload
            host = self.host()
            sockets = list(host.sockets) if host is not None else []
        message = {"type": "bar", **payload}
        for ws in sockets:
            try:
                await ws.send_json(message)
            except Exception:
                if host is not None:
                    host.sockets.discard(ws)


def _bar_channel(room: "Room"):
    """A queue plus a fresh stop event. Drains reports left by the previous position."""
    from server.bot import _drain, manager

    mgr = manager()
    if room._bar_queue is None:
        room._bar_queue = mgr.Queue()
    else:
        _drain(room._bar_queue)
    return room._bar_queue, mgr.Event()


def _latest_bar(progress_queue, token):
    """The newest report for ``token``, discarding anything older still in the queue."""
    import queue as queue_module

    latest = None
    try:
        while True:
            item = progress_queue.get_nowait()
            if item[0] == token:
                latest = item
    except (queue_module.Empty, EOFError, OSError):
        pass
    return latest


def _chance_for_winner(winner: int | None) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == 0 else 0.0


def _bar_spec(spec: dict) -> dict:
    """The few fields the worker needs. The search budget stays behind."""
    slim = {"kind": spec.get("kind", "random"), "seed": int(spec.get("seed") or 0)}
    if spec.get("path"):
        slim["path"] = spec["path"]
    if spec.get("bar_samples") is not None:
        slim["bar_samples"] = int(spec["bar_samples"])
    return slim


def _lobby_view(seat: int) -> dict:
    return {
        "you": seat,
        "your_turn": False,
        "hand": [],
        "top_card": None,
        "under_cards": [],
        "active_suit": None,
        "active_suit_name": None,
        "penalty_cards": 0,
        "ace_active": False,
        "named_by_over": False,
        "played": [],
        "opponent_hand_size": 0,
        "draw_pile_size": 0,
        "discard_size": 0,
        "current": None,
        "legal_actions": [],
        "dont_play": {"id": DONT_PLAY, "kind": None, "label": None, "legal": False},
        "log": [],
        "result": None,
    }
