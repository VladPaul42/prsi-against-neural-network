"""Start the table: python -m server"""

from __future__ import annotations

import argparse
import os
import re
import secrets
from pathlib import Path

import uvicorn

from server.app import create_app
from server.review import EVAL_SAMPLES, REVIEW_LINE_LIMIT
from server.room import Room

ROOT = Path(__file__).resolve().parent.parent
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Play Prší with a friend or the bot. You run this; they open the link."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--code",
        default=None,
        help="Room code. Defaults to a random one, or the PRSI_ROOM_CODE environment variable.",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Network folder for the bot, e.g. runs/league/gen_003. "
        "Defaults to the best generation in runs/league; without one the bot is plain ISMCTS.",
    )
    parser.add_argument("--think", type=float, default=5.0, help="Seconds the bot thinks per move.")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=100_000,
        help="Most imagined games a bot will search on one move. "
        'In "Me vs the bot" and "Bot vs bot" the card is played when --think '
        "elapses; if this cap is reached first the bot waits out the rest of "
        'that time. In "My friend vs the bot" the card is played when the host '
        "confirms.",
    )
    parser.add_argument(
        "--review-line-limit",
        type=int,
        default=REVIEW_LINE_LIMIT,
        help="Imagined games the saved-game review gives each legal card.",
    )
    parser.add_argument(
        "--bar-samples",
        type=int,
        default=EVAL_SAMPLES,
        help="Finishes the evaluation bar plays from each position, "
        "on the review page and in Bot vs bot. "
        "The Bot vs bot bar updates as those finishes complete.",
    )
    parser.add_argument("--no-bot", action="store_true", help="Two-player games only.")
    args = parser.parse_args(argv)
    if args.review_line_limit < 1:
        raise SystemExit("The review line limit must be at least 1.")
    if args.bar_samples < 1:
        raise SystemExit("The bar sample count must be at least 1.")
    raw = args.code or os.environ.get("PRSI_ROOM_CODE") or _generate_code()
    code = _normalize_code(raw)
    bot_factory = None
    bot_note = ""
    spec = None
    if not args.no_bot:
        from server.bot import BotPlayer, choose_bot, warm_up

        if args.search_limit < 1:
            raise SystemExit("The search limit must be at least 1.")
        spec, bot_note = choose_bot(args.network, args.think)
        spec["review_line_limit"] = args.review_line_limit
        spec["bar_samples"] = args.bar_samples

        def bot_factory(seat, on_panel):
            bot = BotPlayer(seat, spec, on_panel)
            bot.search_limit = args.search_limit
            return bot

        warm_up(spec)
    room = Room(code, ROOT / "games", bot_factory=bot_factory, bot_note=bot_note, bot_spec=spec)
    app = create_app(room, host=args.host, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _generate_code() -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(6))


def _normalize_code(raw: str) -> str:
    code = str(raw).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{4,12}", code):
        raise SystemExit("The room code must be 4 to 12 letters or digits.")
    return code


if __name__ == "__main__":
    main()
