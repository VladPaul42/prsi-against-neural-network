"""A seat at the table.

A person connected through a websocket is a ``HumanPlayer``. The bot
(``server.bot.BotPlayer``) is another ``Player``: it is given the same view
and returns an action number. When the bot plays the host's side, the host's
browser watches that seat; the friend never receives it.
"""

from __future__ import annotations

import asyncio
import secrets
from abc import ABC, abstractmethod


class Player(ABC):
    async def notify(self, view: dict) -> None:
        """Called with this seat's view after every change."""

    @abstractmethod
    async def choose_action(self, view: dict) -> int:
        """Return one action number from ``view["legal_actions"]``."""


class HumanPlayer(Player):
    def __init__(self, name: str, seat: int):
        self.name = name
        # -1 while a bot mode keeps this person out of the game.
        self.seat = seat
        # The seat this person goes back to when they play again.
        self.home_seat = seat
        self.is_host = False
        self.token = secrets.token_urlsafe(24)
        self.sockets: set = set()
        self.queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()
        self.expected_token: int | None = None
        self.latest_view: dict | None = None

    def submit(self, action: int, move: int) -> None:
        self.queue.put_nowait((action, move))

    async def notify(self, view: dict) -> None:
        self.latest_view = view

    async def choose_action(self, view: dict) -> int:
        self.latest_view = view
        legal = set(view["legal_actions"])
        token = self.expected_token
        while True:
            action, move = await self.queue.get()
            if move != token:
                continue
            if type(action) is int and action in legal:
                return action
            await self.send_error("That move isn't allowed.")

    async def send_error(self, message: str) -> None:
        dead = []
        for ws in list(self.sockets):
            try:
                await ws.send_json({"type": "error", "message": message})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)
